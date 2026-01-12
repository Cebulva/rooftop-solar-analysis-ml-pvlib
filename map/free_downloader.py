import mercantile
import requests
import io
from PIL import Image 
import math

def get_free_satellite_image(lat, lon, zoom=19, filename="my_property.png"):
    """
    Downloads the satellite tile for a specific location from Esri World Imagery.
    """
    
    # 1. Convert Lat/Lon to Tile Coordinates (X, Y, Z)
    # mercantile.tile(lng, lat, zoom) -> Note the order: Longitude first!
    tile = mercantile.tile(lon, lat, zoom)
    
    # 2. Construct the URL
    # This is the same URL your Folium map uses
    url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{tile.y}/{tile.x}"
    
    print(f"Fetching tile: {url}")
    
    # 3. Download the image
    headers = {"User-Agent": "Student-Solar-Project/1.0"}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        # 4. Save the image
        image_data = response.content
        image = Image.open(io.BytesIO(image_data))
        image.save(filename)
        print(f"✅ Saved free satellite image to {filename}")
        return filename
    else:
        print("❌ Failed to download tile.")
        return None

def calculate_gsd(lat, zoom):
    """
    Calculates the Ground Sample Distance (meters per pixel).
    """
    # Convert latitude to radians
    lat_rad = math.radians(lat)
    
    # Earth circumference constant / 2^zoom
    # 156543.03 is derived from (2 * pi * 6378137) / 256
    resolution = (156543.03 * math.cos(lat_rad)) / (2 ** zoom)
    
    return resolution