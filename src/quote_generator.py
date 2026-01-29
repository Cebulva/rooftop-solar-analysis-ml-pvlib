"""
Quote Generator Service
Handles creating, storing, and sending quote requests to dealers.
"""

import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from src.dealer_finder import DB_PATH, init_database, Dealer
from src.dealer_grader import GradedDealer
from src.email_service import QuoteEmailData, send_quote_email, send_confirmation_email


@dataclass
class QuoteRequest:
    """Represents a quote request."""
    id: Optional[int] = None
    inquiry_id: str = ""
    dealer_id: int = 0
    dealer_name: str = ""
    dealer_email: str = ""
    system_kwp: float = 0.0
    panel_count: int = 0
    roof_area_sqm: float = 0.0
    address: str = ""
    lat: float = 0.0
    lon: float = 0.0
    customer_name: str = ""
    customer_email: str = ""
    customer_phone: Optional[str] = None
    additional_notes: Optional[str] = None
    include_battery: bool = False
    include_financing: bool = False
    include_permits: bool = False
    status: str = "pending"
    created_at: Optional[str] = None
    sent_at: Optional[str] = None


def create_quote_request(
    graded_dealer: GradedDealer,
    solar_results: Dict[str, Any],
    consumption_inputs: Dict[str, Any],
    location: Dict[str, Any],
    customer_info: Dict[str, Any],
    inquiry_id: str
) -> QuoteRequest:
    """
    Create a quote request from solar analysis results.

    Args:
        graded_dealer: The selected dealer with grades
        solar_results: Results from Stage 3b (system_kwp, panel_count, etc.)
        consumption_inputs: User consumption data
        location: Location data (lat, lon, address)
        customer_info: Customer contact information
        inquiry_id: The inquiry ID

    Returns:
        QuoteRequest object
    """
    return QuoteRequest(
        inquiry_id=inquiry_id,
        dealer_id=graded_dealer.dealer.id or 0,
        dealer_name=graded_dealer.name,
        dealer_email=graded_dealer.email or "",
        system_kwp=solar_results.get("system_kwp", 0),
        panel_count=solar_results.get("panel_count", 0),
        roof_area_sqm=solar_results.get("usable_roof_area_m2", 0),
        address=location.get("address", ""),
        lat=location.get("lat", 0),
        lon=location.get("lon", 0),
        customer_name=customer_info.get("name", ""),
        customer_email=customer_info.get("email", ""),
        customer_phone=customer_info.get("phone"),
        additional_notes=customer_info.get("notes"),
        include_battery=customer_info.get("include_battery", False),
        include_financing=customer_info.get("include_financing", False),
        include_permits=customer_info.get("include_permits", False)
    )


def save_quote_request(quote: QuoteRequest) -> int:
    """
    Save a quote request to the database.

    Returns:
        The ID of the saved quote request
    """
    init_database()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO quote_requests (
            inquiry_id, dealer_id, system_kwp, panel_count, roof_area_sqm,
            address, lat, lon, customer_name, customer_email, customer_phone,
            additional_notes, include_battery, include_financing, include_permits,
            status, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        quote.inquiry_id,
        quote.dealer_id,
        quote.system_kwp,
        quote.panel_count,
        quote.roof_area_sqm,
        quote.address,
        quote.lat,
        quote.lon,
        quote.customer_name,
        quote.customer_email,
        quote.customer_phone,
        quote.additional_notes,
        int(quote.include_battery),
        int(quote.include_financing),
        int(quote.include_permits),
        "pending",
        datetime.now().isoformat()
    ))

    quote_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return quote_id


def update_quote_status(quote_id: int, status: str, sent_at: Optional[str] = None):
    """Update the status of a quote request."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if sent_at:
        cursor.execute("""
            UPDATE quote_requests SET status = ?, sent_at = ? WHERE id = ?
        """, (status, sent_at, quote_id))
    else:
        cursor.execute("""
            UPDATE quote_requests SET status = ? WHERE id = ?
        """, (status, quote_id))

    conn.commit()
    conn.close()


def get_quote_requests_for_inquiry(inquiry_id: str) -> List[QuoteRequest]:
    """Get all quote requests for a specific inquiry."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT qr.*, d.name as dealer_name, d.email as dealer_email
        FROM quote_requests qr
        LEFT JOIN dealers d ON qr.dealer_id = d.id
        WHERE qr.inquiry_id = ?
        ORDER BY qr.created_at DESC
    """, (inquiry_id,))

    quotes = []
    for row in cursor.fetchall():
        quotes.append(QuoteRequest(
            id=row["id"],
            inquiry_id=row["inquiry_id"],
            dealer_id=row["dealer_id"],
            dealer_name=row["dealer_name"] or "",
            dealer_email=row["dealer_email"] or "",
            system_kwp=row["system_kwp"],
            panel_count=row["panel_count"],
            roof_area_sqm=row["roof_area_sqm"],
            address=row["address"],
            lat=row["lat"],
            lon=row["lon"],
            customer_name=row["customer_name"],
            customer_email=row["customer_email"],
            customer_phone=row["customer_phone"],
            additional_notes=row["additional_notes"],
            include_battery=bool(row["include_battery"]),
            include_financing=bool(row["include_financing"]),
            include_permits=bool(row["include_permits"]),
            status=row["status"],
            created_at=row["created_at"],
            sent_at=row["sent_at"]
        ))

    conn.close()
    return quotes


def send_quote_to_dealer(
    quote: QuoteRequest,
    roof_tilt: float,
    roof_orientation: str,
    estimated_production: float
) -> tuple[bool, str]:
    """
    Send the quote request email to the dealer.

    Args:
        quote: The quote request to send
        roof_tilt: Roof tilt angle in degrees
        roof_orientation: Roof orientation (e.g., "South-West")
        estimated_production: Estimated annual production in kWh

    Returns:
        Tuple of (success, message)
    """
    if not quote.dealer_email:
        return False, "Dealer email not available. Please contact them directly."

    # Create email data
    email_data = QuoteEmailData(
        dealer_name=quote.dealer_name,
        dealer_email=quote.dealer_email,
        system_kwp=quote.system_kwp,
        panel_count=quote.panel_count,
        roof_area_sqm=quote.roof_area_sqm,
        roof_tilt=roof_tilt,
        roof_orientation=roof_orientation,
        estimated_production=estimated_production,
        installation_address=quote.address,
        lat=quote.lat,
        lon=quote.lon,
        customer_name=quote.customer_name,
        customer_email=quote.customer_email,
        customer_phone=quote.customer_phone,
        additional_notes=quote.additional_notes,
        include_battery=quote.include_battery,
        include_financing=quote.include_financing,
        include_permits=quote.include_permits,
        inquiry_id=quote.inquiry_id
    )

    # Send to dealer
    success, message = send_quote_email(email_data)

    if success:
        # Update quote status
        update_quote_status(quote.id, "sent", datetime.now().isoformat())

        # Send confirmation to customer
        send_confirmation_email(
            quote.customer_email,
            quote.customer_name,
            quote.dealer_name,
            quote.system_kwp
        )

    return success, message


def process_quote_submission(
    graded_dealer: GradedDealer,
    solar_results: Dict[str, Any],
    final_analysis: Dict[str, Any],
    location: Dict[str, Any],
    customer_info: Dict[str, Any],
    inquiry_id: str
) -> tuple[bool, str, Optional[int]]:
    """
    Complete quote submission process: save and send.

    Args:
        graded_dealer: Selected dealer
        solar_results: Stage 3b results
        final_analysis: Stage 4 analysis
        location: Location data
        customer_info: Customer contact info
        inquiry_id: Inquiry ID

    Returns:
        Tuple of (success, message, quote_id)
    """
    # Create quote request
    quote = create_quote_request(
        graded_dealer=graded_dealer,
        solar_results=solar_results,
        consumption_inputs={},  # Not needed for quote
        location=location,
        customer_info=customer_info,
        inquiry_id=inquiry_id
    )

    # Save to database
    quote_id = save_quote_request(quote)
    quote.id = quote_id

    # Get additional data for email
    roof_tilt = solar_results.get("tilt_angle", 35)
    roof_orientation = location.get("orientation", "South")

    # Calculate estimated production
    production = final_analysis.get("production", {})
    specific_yield = production.get("specific_yield_kwh_kwp", 950)
    estimated_production = quote.system_kwp * specific_yield

    # Send email
    success, message = send_quote_to_dealer(
        quote=quote,
        roof_tilt=roof_tilt,
        roof_orientation=roof_orientation,
        estimated_production=estimated_production
    )

    if not success:
        # Update status to indicate send failure
        update_quote_status(quote_id, "send_failed")

    return success, message, quote_id


if __name__ == "__main__":
    # Test quote creation
    from src.dealer_finder import find_nearby_dealers, add_sample_dealers
    from src.dealer_grader import get_top_dealers

    init_database()
    add_sample_dealers()

    # Find dealers
    dealers = find_nearby_dealers(52.52, 13.405, radius_km=100)
    graded = get_top_dealers(dealers, system_kwp=5.0, limit=1)

    if graded:
        # Create test quote
        quote = create_quote_request(
            graded_dealer=graded[0],
            solar_results={
                "system_kwp": 5.2,
                "panel_count": 13,
                "usable_roof_area_m2": 32.5,
                "tilt_angle": 35,
                "azimuth": 210
            },
            consumption_inputs={},
            location={
                "lat": 52.52,
                "lon": 13.405,
                "address": "Hauptstraße 123, 12345 Berlin"
            },
            customer_info={
                "name": "Max Mustermann",
                "email": "max@example.com",
                "phone": "+49 170 1234567",
                "notes": "Interested in battery storage as well.",
                "include_battery": True,
                "include_financing": True
            },
            inquiry_id="INQ-001"
        )

        print(f"Created quote for dealer: {quote.dealer_name}")
        print(f"System: {quote.system_kwp} kWp, {quote.panel_count} panels")
        print(f"Customer: {quote.customer_name} ({quote.customer_email})")

        # Save quote
        quote_id = save_quote_request(quote)
        print(f"Saved with ID: {quote_id}")
