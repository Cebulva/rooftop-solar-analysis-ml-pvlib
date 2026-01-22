import numpy as np
import cv2
import math
import pandas as pd
import pvlib

import pvlib.solarposition as solarposition
import pvlib.irradiance as irradiance
import pvlib.atmosphere as atmosphere
from pvlib.clearsky import ineichen

def calculate_global_gsd(lat, zoom):
    """Calculates meters per pixel for any latitude and zoom level."""
    equatorial_circumference = 40075016.686
    gsd = (equatorial_circumference * math.cos(math.radians(lat))) / (2**(zoom + 8))
    return gsd

def get_masked_roof_array(zoom_img, polygon_pts):
    pts = np.array(polygon_pts, dtype=np.int32)
    mask = np.zeros(zoom_img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    roof_cutout = cv2.bitwise_and(zoom_img, zoom_img, mask=mask)
    return mask, roof_cutout

def analyze_roof_texture(roof_cutout, mask, threshold=15.0):
    gray_roof = cv2.cvtColor(roof_cutout, cv2.COLOR_BGR2GRAY)
    relevant_intensities = gray_roof[mask > 0]
    if len(relevant_intensities) == 0:
        return "Flat", 0.0
    std_dev = np.std(relevant_intensities)
    roof_type = "Pitched" if std_dev > threshold else "Flat"
    return roof_type, std_dev

def draw_azimuth_arrow(img, azimuth_deg):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    length = min(h, w) // 4
    canvas = img.copy()
    # North Reference
    cv2.arrowedLine(canvas, center, (center[0], center[1] - length), (255, 255, 255), 2)
    # Azimuth Arrow
    theta = np.radians(azimuth_deg - 90)
    end_x = int(center[0] + length * np.cos(theta))
    end_y = int(center[1] + length * np.sin(theta))
    cv2.arrowedLine(canvas, center, (end_x, end_y), (0, 0, 255), 2)
    return canvas

def calculate_solar_potential(lat, lon, tilt, azimuth):
    """Calculates the peak solar energy potential (W/m2)."""
    # June 21st, 2026 at Noon
    times = pd.date_range('2026-06-21 12:00:00', periods=1, freq='H', tz='Europe/Berlin')
    
    # 1. Sun Position
    solpos = solarposition.get_solarposition(times, lat, lon)
    
    # 2. Extra-terrestrial Radiation (Updated to match your documentation)
    # Note: If get_extra_radiation still fails, try irradiance.get_total_extraradiation
    dni_extra = irradiance.get_extra_radiation(times)
    
    # 3. Air Mass & Pressure
    airmass = atmosphere.get_relative_airmass(solpos['apparent_zenith'])
    pressure = atmosphere.alt2pres(0) 
    am_abs = atmosphere.get_absolute_airmass(airmass, pressure)
    
    # 4. Clear Sky Model
    cs = ineichen(solpos['apparent_zenith'], am_abs, linke_turbidity=3.0, dni_extra=dni_extra)
    
    # 5. Total Irradiance on the Tilted Surface
    total_irrad = irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        solar_zenith=solpos['apparent_zenith'],
        solar_azimuth=solpos['azimuth'],
        dni=cs['dni'],
        ghi=cs['ghi'],
        dhi=cs['dhi']
    )
    
    return float(total_irrad['poa_global'].iloc[0])

def get_sunny_polygon_mask(roof_only, mask, threshold_offset=20):
    """
    Finds the sunny area by identifying the dominant brightness peak 
    within the specific roof geometry.
    """
    # 1. Convert to grayscale
    gray = cv2.cvtColor(roof_only, cv2.COLOR_BGR2GRAY)
    
    # 2. Ensure mask is uint8 for OpenCV compatibility
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
        
    # 3. Apply a Gaussian Blur to reduce pixel noise (e.g. tile textures)
    # This helps in identifying 'areas' rather than individual pixels
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 4. Find the 'Peak' brightness inside the ROOF MASK only
    # This ignores bright cars or roads outside the roof
    hist = cv2.calcHist([blurred], [0], mask, [256], [0, 256])
    brightest_significant_val = np.argmax(hist)
    
    # 5. Dynamic Thresholding
    # We define 'Sunny' as anything near that peak brightness
    _, sun_mask = cv2.threshold(blurred, brightest_significant_val - threshold_offset, 255, cv2.THRESH_BINARY)
    
    # 6. CRITICAL: Bitwise AND with the original mask
    # This ensures that even if the threshold is low, we only show results on the ROOF
    final_sun_mask = cv2.bitwise_and(sun_mask, mask)
    
    return final_sun_mask