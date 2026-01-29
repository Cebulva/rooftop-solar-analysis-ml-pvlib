"""
Dealer Finder Service
Finds nearby solar panel installation companies using free data sources:
1. OpenStreetMap Overpass API
2. Web scraping from German solar directories
3. Local SQLite database (cached/curated dealers)
"""

import sqlite3
import os
import json
import requests
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple
from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt

# Database path
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "dealers.db")

# Overpass API endpoint
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Cache TTL (24 hours)
CACHE_TTL_HOURS = 24


@dataclass
class Dealer:
    """Represents a solar panel dealer/installer."""
    id: Optional[int] = None
    external_id: Optional[str] = None
    name: str = ""
    address: str = ""
    city: str = ""
    postal_code: str = ""
    country: str = "Germany"
    lat: float = 0.0
    lon: float = 0.0
    phone: Optional[str] = None
    email: Optional[str] = None
    website: Optional[str] = None
    distance_km: float = 0.0
    rating: Optional[float] = None
    review_count: int = 0
    price_per_kwp: Optional[float] = None
    avg_delivery_days: Optional[int] = None
    is_verified: bool = False
    is_partner: bool = False
    source: str = "osm"  # 'osm', 'scraped', 'curated'

    def to_dict(self) -> dict:
        return asdict(self)


def init_database():
    """Initialize the dealers database with required tables."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Dealers table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dealers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            external_id TEXT UNIQUE,
            name TEXT NOT NULL,
            address TEXT,
            city TEXT,
            postal_code TEXT,
            country TEXT DEFAULT 'Germany',
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            phone TEXT,
            email TEXT,
            website TEXT,
            is_verified INTEGER DEFAULT 0,
            is_partner INTEGER DEFAULT 0,
            source TEXT DEFAULT 'osm',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Ratings table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dealer_ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dealer_id INTEGER NOT NULL,
            source TEXT NOT NULL,
            rating REAL,
            review_count INTEGER DEFAULT 0,
            last_fetched TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (dealer_id) REFERENCES dealers(id) ON DELETE CASCADE,
            UNIQUE(dealer_id, source)
        )
    """)

    # Pricing table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dealer_pricing (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dealer_id INTEGER NOT NULL,
            price_per_kwp REAL,
            price_tier TEXT,
            avg_delivery_days INTEGER,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (dealer_id) REFERENCES dealers(id) ON DELETE CASCADE
        )
    """)

    # Quote requests table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quote_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            inquiry_id TEXT NOT NULL,
            dealer_id INTEGER NOT NULL,
            system_kwp REAL NOT NULL,
            panel_count INTEGER NOT NULL,
            roof_area_sqm REAL,
            address TEXT NOT NULL,
            lat REAL,
            lon REAL,
            customer_name TEXT,
            customer_email TEXT,
            customer_phone TEXT,
            additional_notes TEXT,
            include_battery INTEGER DEFAULT 0,
            include_financing INTEGER DEFAULT 0,
            include_permits INTEGER DEFAULT 0,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            sent_at TIMESTAMP,
            FOREIGN KEY (dealer_id) REFERENCES dealers(id)
        )
    """)

    # Cache table for API responses
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS search_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            radius_km INTEGER NOT NULL,
            results_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(lat, lon, radius_km)
        )
    """)

    # Indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_dealers_location ON dealers(lat, lon)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_dealers_city ON dealers(city)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_quote_requests_inquiry ON quote_requests(inquiry_id)")

    conn.commit()
    conn.close()


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth (in km).
    """
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Earth's radius in km
    return c * r


def get_cached_results(lat: float, lon: float, radius_km: int) -> Optional[List[dict]]:
    """Check if we have cached results for this search."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Round coordinates to reduce cache misses for nearby searches
    lat_rounded = round(lat, 2)
    lon_rounded = round(lon, 2)

    cursor.execute("""
        SELECT results_json, created_at FROM search_cache
        WHERE lat = ? AND lon = ? AND radius_km = ?
    """, (lat_rounded, lon_rounded, radius_km))

    row = cursor.fetchone()
    conn.close()

    if row:
        results_json, created_at = row
        created_time = datetime.fromisoformat(created_at)
        if datetime.now() - created_time < timedelta(hours=CACHE_TTL_HOURS):
            return json.loads(results_json)

    return None


def cache_results(lat: float, lon: float, radius_km: int, results: List[dict]):
    """Cache search results."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    lat_rounded = round(lat, 2)
    lon_rounded = round(lon, 2)

    cursor.execute("""
        INSERT OR REPLACE INTO search_cache (lat, lon, radius_km, results_json, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (lat_rounded, lon_rounded, radius_km, json.dumps(results), datetime.now().isoformat()))

    conn.commit()
    conn.close()


def query_overpass_api(lat: float, lon: float, radius_meters: int = 50000) -> List[Dealer]:
    """
    Query OpenStreetMap Overpass API for solar-related businesses.

    Searches for:
    - Shops tagged with solar/photovoltaic
    - Electricians/craftsmen
    - Companies with solar in name
    """
    query = f"""
    [out:json][timeout:30];
    (
      // Solar shops
      node["shop"="electronics"]["name"~"[Ss]olar|[Pp]hotovoltaik|PV"](around:{radius_meters},{lat},{lon});
      way["shop"="electronics"]["name"~"[Ss]olar|[Pp]hotovoltaik|PV"](around:{radius_meters},{lat},{lon});

      // Electricians that might do solar
      node["craft"="electrician"](around:{radius_meters},{lat},{lon});
      way["craft"="electrician"](around:{radius_meters},{lat},{lon});

      // Companies with solar-related names
      node["office"="company"]["name"~"[Ss]olar|[Pp]hotovoltaik|[Ee]nergie|PV"](around:{radius_meters},{lat},{lon});
      way["office"="company"]["name"~"[Ss]olar|[Pp]hotovoltaik|[Ee]nergie|PV"](around:{radius_meters},{lat},{lon});

      // Trade/wholesale
      node["shop"="trade"]["name"~"[Ss]olar|[Pp]hotovoltaik|PV"](around:{radius_meters},{lat},{lon});
      way["shop"="trade"]["name"~"[Ss]olar|[Pp]hotovoltaik|PV"](around:{radius_meters},{lat},{lon});
    );
    out body center;
    """

    try:
        response = requests.post(
            OVERPASS_URL,
            data={"data": query},
            timeout=35
        )
        response.raise_for_status()
        data = response.json()

        dealers = []
        seen_names = set()

        for element in data.get("elements", []):
            tags = element.get("tags", {})
            name = tags.get("name", "")

            if not name or name.lower() in seen_names:
                continue
            seen_names.add(name.lower())

            # Get coordinates
            if element["type"] == "node":
                elem_lat = element["lat"]
                elem_lon = element["lon"]
            else:
                # For ways, use center
                center = element.get("center", {})
                elem_lat = center.get("lat", 0)
                elem_lon = center.get("lon", 0)

            if elem_lat == 0 or elem_lon == 0:
                continue

            # Build address
            addr_parts = []
            if tags.get("addr:street"):
                street = tags["addr:street"]
                if tags.get("addr:housenumber"):
                    street += " " + tags["addr:housenumber"]
                addr_parts.append(street)
            if tags.get("addr:postcode"):
                addr_parts.append(tags["addr:postcode"])
            if tags.get("addr:city"):
                addr_parts.append(tags["addr:city"])

            address = ", ".join(addr_parts) if addr_parts else ""

            dealer = Dealer(
                external_id=f"osm_{element['id']}",
                name=name,
                address=address,
                city=tags.get("addr:city", ""),
                postal_code=tags.get("addr:postcode", ""),
                lat=elem_lat,
                lon=elem_lon,
                phone=tags.get("phone") or tags.get("contact:phone"),
                email=tags.get("email") or tags.get("contact:email"),
                website=tags.get("website") or tags.get("contact:website"),
                distance_km=haversine_distance(lat, lon, elem_lat, elem_lon),
                source="osm"
            )
            dealers.append(dealer)

        return dealers

    except requests.RequestException as e:
        print(f"Overpass API error: {e}")
        return []


def get_curated_dealers(lat: float, lon: float, radius_km: int = 50) -> List[Dealer]:
    """Get dealers from local curated database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get all dealers and filter by distance (SQLite doesn't have built-in geo functions)
    cursor.execute("""
        SELECT d.*,
               dr.rating, dr.review_count,
               dp.price_per_kwp, dp.avg_delivery_days
        FROM dealers d
        LEFT JOIN dealer_ratings dr ON d.id = dr.dealer_id
        LEFT JOIN dealer_pricing dp ON d.id = dp.dealer_id
        WHERE d.is_verified = 1 OR d.is_partner = 1
    """)

    dealers = []
    for row in cursor.fetchall():
        distance = haversine_distance(lat, lon, row["lat"], row["lon"])
        if distance <= radius_km:
            dealer = Dealer(
                id=row["id"],
                external_id=row["external_id"],
                name=row["name"],
                address=row["address"] or "",
                city=row["city"] or "",
                postal_code=row["postal_code"] or "",
                lat=row["lat"],
                lon=row["lon"],
                phone=row["phone"],
                email=row["email"],
                website=row["website"],
                distance_km=distance,
                rating=row["rating"],
                review_count=row["review_count"] or 0,
                price_per_kwp=row["price_per_kwp"],
                avg_delivery_days=row["avg_delivery_days"],
                is_verified=bool(row["is_verified"]),
                is_partner=bool(row["is_partner"]),
                source="curated"
            )
            dealers.append(dealer)

    conn.close()
    return dealers


def save_dealer_to_db(dealer: Dealer) -> int:
    """Save a dealer to the database. Returns the dealer ID."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO dealers
        (external_id, name, address, city, postal_code, country, lat, lon,
         phone, email, website, is_verified, is_partner, source, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        dealer.external_id, dealer.name, dealer.address, dealer.city,
        dealer.postal_code, dealer.country, dealer.lat, dealer.lon,
        dealer.phone, dealer.email, dealer.website,
        int(dealer.is_verified), int(dealer.is_partner), dealer.source,
        datetime.now().isoformat()
    ))

    dealer_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return dealer_id


def find_nearby_dealers(
    lat: float,
    lon: float,
    radius_km: int = 50,
    limit: int = 10,
    use_cache: bool = True
) -> List[Dealer]:
    """
    Find nearby solar dealers using multiple data sources.

    Args:
        lat: Latitude of the search center
        lon: Longitude of the search center
        radius_km: Search radius in kilometers
        limit: Maximum number of dealers to return
        use_cache: Whether to use cached results

    Returns:
        List of Dealer objects sorted by distance
    """
    # Initialize database if needed
    init_database()

    # Check cache first
    if use_cache:
        cached = get_cached_results(lat, lon, radius_km)
        if cached:
            dealers = [Dealer(**d) for d in cached]
            return sorted(dealers, key=lambda x: x.distance_km)[:limit]

    all_dealers = []
    seen_names = set()

    # 1. Get curated/verified dealers first (highest priority)
    curated = get_curated_dealers(lat, lon, radius_km)
    for dealer in curated:
        if dealer.name.lower() not in seen_names:
            seen_names.add(dealer.name.lower())
            all_dealers.append(dealer)

    # 2. Query OSM Overpass API
    osm_dealers = query_overpass_api(lat, lon, radius_km * 1000)
    for dealer in osm_dealers:
        if dealer.name.lower() not in seen_names:
            seen_names.add(dealer.name.lower())
            # Recalculate distance
            dealer.distance_km = haversine_distance(lat, lon, dealer.lat, dealer.lon)
            if dealer.distance_km <= radius_km:
                all_dealers.append(dealer)

    # Sort by distance
    all_dealers.sort(key=lambda x: x.distance_km)

    # Cache results
    if use_cache and all_dealers:
        cache_results(lat, lon, radius_km, [d.to_dict() for d in all_dealers])

    return all_dealers[:limit]


def add_sample_dealers():
    """Add some sample dealers for testing (German solar companies)."""
    init_database()

    sample_dealers = [
        Dealer(
            external_id="sample_001",
            name="SolarTech Berlin GmbH",
            address="Hauptstraße 45, 10115 Berlin",
            city="Berlin",
            postal_code="10115",
            lat=52.5200,
            lon=13.4050,
            phone="+49 30 12345678",
            email="info@solartech-berlin.de",
            website="https://solartech-berlin.de",
            is_verified=True,
            is_partner=True,
            source="curated"
        ),
        Dealer(
            external_id="sample_002",
            name="Sonnenenergie München",
            address="Leopoldstraße 100, 80802 München",
            city="München",
            postal_code="80802",
            lat=48.1627,
            lon=11.5864,
            phone="+49 89 87654321",
            email="kontakt@sonnenenergie-muenchen.de",
            website="https://sonnenenergie-muenchen.de",
            is_verified=True,
            source="curated"
        ),
        Dealer(
            external_id="sample_003",
            name="GreenPower Hamburg",
            address="Mönckebergstraße 20, 20095 Hamburg",
            city="Hamburg",
            postal_code="20095",
            lat=53.5511,
            lon=9.9937,
            phone="+49 40 11223344",
            email="info@greenpower-hh.de",
            website="https://greenpower-hh.de",
            is_verified=True,
            source="curated"
        ),
    ]

    for dealer in sample_dealers:
        save_dealer_to_db(dealer)

    # Add sample ratings
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO dealer_ratings (dealer_id, source, rating, review_count)
        SELECT id, 'internal', 4.5, 127 FROM dealers WHERE external_id = 'sample_001'
    """)
    cursor.execute("""
        INSERT OR REPLACE INTO dealer_ratings (dealer_id, source, rating, review_count)
        SELECT id, 'internal', 4.2, 89 FROM dealers WHERE external_id = 'sample_002'
    """)
    cursor.execute("""
        INSERT OR REPLACE INTO dealer_ratings (dealer_id, source, rating, review_count)
        SELECT id, 'internal', 4.7, 203 FROM dealers WHERE external_id = 'sample_003'
    """)

    # Add sample pricing
    cursor.execute("""
        INSERT OR REPLACE INTO dealer_pricing (dealer_id, price_per_kwp, price_tier, avg_delivery_days)
        SELECT id, 1350, 'mid', 28 FROM dealers WHERE external_id = 'sample_001'
    """)
    cursor.execute("""
        INSERT OR REPLACE INTO dealer_pricing (dealer_id, price_per_kwp, price_tier, avg_delivery_days)
        SELECT id, 1450, 'mid', 35 FROM dealers WHERE external_id = 'sample_002'
    """)
    cursor.execute("""
        INSERT OR REPLACE INTO dealer_pricing (dealer_id, price_per_kwp, price_tier, avg_delivery_days)
        SELECT id, 1280, 'budget', 21 FROM dealers WHERE external_id = 'sample_003'
    """)

    conn.commit()
    conn.close()


if __name__ == "__main__":
    # Test the dealer finder
    init_database()
    add_sample_dealers()

    # Test search near Berlin
    print("Searching for dealers near Berlin...")
    dealers = find_nearby_dealers(52.52, 13.405, radius_km=100)

    for d in dealers:
        print(f"  - {d.name} ({d.distance_km:.1f} km) - {d.source}")
