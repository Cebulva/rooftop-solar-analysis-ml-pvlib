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

def calculate_azimuth(img, polygon_pts):
    pts = np.array(polygon_pts, dtype=np.int32)
    rect = cv2.minAreaRect(pts)
    (x, y), (w, h), angle = rect
    
    # Favor short axis for slope
    azimuth = angle if w > h else angle + 90
    azimuth = azimuth % 360

    # Brightness heuristic (point away from shadow)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_i, w_i = gray.shape
    if np.mean(gray[h_i//2:, :]) > np.mean(gray[0:h_i//2, :]):
        if azimuth < 90 or azimuth > 270: azimuth = (azimuth + 180) % 360
    else:
        if 90 < azimuth < 270: azimuth = (azimuth + 180) % 360
    return float(azimuth)

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

def get_sunny_polygon_mask(roof_only, threshold_val=180):
    """
    Creates a binary mask of the brightest parts of the roof (sun vs shadow).
    """
    gray = cv2.cvtColor(roof_only, cv2.COLOR_BGR2GRAY)
    _, sun_mask = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
    return sun_mask