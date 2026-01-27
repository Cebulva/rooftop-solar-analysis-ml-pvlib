import numpy as np
import cv2
import math
import pandas as pd
import pvlib

import pvlib.solarposition as solarposition
import pvlib.irradiance as irradiance
import pvlib.atmosphere as atmosphere
from pvlib.clearsky import ineichen

# ==========================================
# ⚙️ ROOF DETECTION CONFIGURATION (Tweak these)
# ==========================================
MIN_FLAT_ROOF_COVERAGE = 0.75  # 75% - If usable area is above this, likely flat
TEXTURE_VARIANCE_THRESHOLD = 15.0  # Standard deviation threshold for texture analysis
MIN_BRIGHTNESS_DIFF = 30  # Minimum brightness difference to detect pitched roof
# ==========================================

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

def analyze_roof_geometry(roof_cutout, mask, sun_mask):
    """
    Enhanced roof type detection using multiple heuristics:
    1. Usable area percentage (flat roofs have uniform lighting)
    2. Brightness distribution analysis (pitched roofs have bright/dark sides)
    3. Texture variance (tiles vs. flat materials)
    
    Returns: (roof_type, confidence_score, debug_info)
    """
    gray_roof = cv2.cvtColor(roof_cutout, cv2.COLOR_BGR2GRAY)
    
    # Cast masks to uint8
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if sun_mask.dtype != np.uint8:
        sun_mask = sun_mask.astype(np.uint8)
    
    # Extract relevant pixels
    all_roof_intensities = gray_roof[mask > 0]
    sunny_intensities = gray_roof[sun_mask > 0]
    
    if len(all_roof_intensities) == 0:
        return "Flat", 0.0, {"reason": "No roof pixels detected"}
    
    # ============================================
    # HEURISTIC 1: Usable Area Coverage Ratio
    # ============================================
    total_roof_pixels = np.sum(mask > 0)
    usable_roof_pixels = np.sum(sun_mask > 0)
    coverage_ratio = usable_roof_pixels / total_roof_pixels if total_roof_pixels > 0 else 0.0
    
    # High coverage = likely flat (uniform lighting)
    # Low coverage = likely pitched (one side in shadow)
    flat_score_from_coverage = coverage_ratio
    
    # ============================================
    # HEURISTIC 2: Brightness Distribution
    # ============================================
    mean_intensity = np.mean(all_roof_intensities)
    max_intensity = np.max(all_roof_intensities)
    min_intensity = np.min(all_roof_intensities)
    brightness_range = max_intensity - min_intensity
    
    # Pitched roofs typically have one bright side and one darker side
    # resulting in larger brightness range
    pitched_score_from_brightness = min(brightness_range / 255.0, 1.0)
    
    # ============================================
    # HEURISTIC 3: Texture Variance
    # ============================================
    # Use sunny area for texture analysis to avoid shadow influence
    analysis_mask = sun_mask if usable_roof_pixels > 100 else mask
    texture_intensities = gray_roof[analysis_mask > 0]
    
    if len(texture_intensities) > 0:
        std_dev = np.std(texture_intensities)
    else:
        std_dev = 0.0
    
    pitched_score_from_texture = 1.0 if std_dev > TEXTURE_VARIANCE_THRESHOLD else 0.0
    
    # ============================================
    # DECISION LOGIC
    # ============================================
    debug_info = {
        "coverage_ratio": coverage_ratio,
        "brightness_range": brightness_range,
        "std_dev": std_dev,
        "flat_score_coverage": flat_score_from_coverage,
        "pitched_score_brightness": pitched_score_from_brightness,
        "pitched_score_texture": pitched_score_from_texture
    }
    
    # PRIMARY CHECK: Coverage ratio (most reliable indicator)
    if coverage_ratio >= MIN_FLAT_ROOF_COVERAGE:
        roof_type = "Flat"
        confidence = flat_score_from_coverage
        debug_info["reason"] = f"High coverage ({coverage_ratio:.1%}) indicates uniform lighting"
    
    # SECONDARY CHECK: Significant brightness difference
    elif brightness_range >= MIN_BRIGHTNESS_DIFF:
        roof_type = "Pitched"
        confidence = pitched_score_from_brightness
        debug_info["reason"] = f"Brightness range ({brightness_range:.0f}) indicates shadow on one side"
    
    # TERTIARY CHECK: Texture analysis
    elif std_dev > TEXTURE_VARIANCE_THRESHOLD:
        roof_type = "Pitched"
        confidence = pitched_score_from_texture
        debug_info["reason"] = f"High texture variance ({std_dev:.1f}) suggests tiles/shingles"
    
    # DEFAULT: Assume flat if uncertain
    else:
        roof_type = "Flat"
        confidence = 0.5
        debug_info["reason"] = "Uncertain - defaulting to Flat"
    
    return roof_type, confidence, debug_info

def analyze_roof_texture(roof_cutout, mask, sun_mask=None, threshold=TEXTURE_VARIANCE_THRESHOLD):
    """
    DEPRECATED: Kept for backward compatibility.
    Use analyze_roof_geometry() for better detection.
    """
    gray_roof = cv2.cvtColor(roof_cutout, cv2.COLOR_BGR2GRAY)
    
    # Use sun_mask if provided, otherwise fallback to the full roof mask
    analysis_mask = sun_mask if sun_mask is not None else mask
    
    # Cast to uint8 to avoid OpenCV errors
    if analysis_mask.dtype != np.uint8:
        analysis_mask = analysis_mask.astype(np.uint8)
        
    relevant_intensities = gray_roof[analysis_mask > 0]
    
    if len(relevant_intensities) == 0:
        return "Flat", 0.0
        
    std_dev = np.std(relevant_intensities)
    
    # Low variance = Flat (bitumen/concrete/gravel)
    # High variance = Pitched (tiles/shingles)
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