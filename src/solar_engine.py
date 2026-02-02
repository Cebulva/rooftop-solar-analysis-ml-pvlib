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
MIN_FLAT_ROOF_COVERAGE = 0.60  # 60% - If usable area is above this, likely flat
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


def calculate_optimal_azimuth(lat, lon, tilt):
    """
    Calculates the optimal panel azimuth angle that maximizes annual solar irradiance.

    Uses pvlib to simulate irradiance across the year and finds the azimuth
    that produces maximum energy output.

    Args:
        lat: Latitude of the location
        lon: Longitude of the location
        tilt: Panel tilt angle in degrees

    Returns:
        Optimal azimuth angle in degrees (0° = North, 90° = East, 180° = South, 270° = West)
    """
    # For quick calculation, test key dates throughout the year
    # Spring equinox, Summer solstice, Fall equinox, Winter solstice
    test_dates = [
        '2026-03-20 12:00:00',  # Spring equinox
        '2026-06-21 12:00:00',  # Summer solstice
        '2026-09-22 12:00:00',  # Fall equinox
        '2026-12-21 12:00:00',  # Winter solstice
    ]

    times = pd.DatetimeIndex(test_dates).tz_localize('Europe/Berlin')

    # Get sun position for all test dates
    solpos = solarposition.get_solarposition(times, lat, lon)
    dni_extra = irradiance.get_extra_radiation(times)
    airmass = atmosphere.get_relative_airmass(solpos['apparent_zenith'])
    pressure = atmosphere.alt2pres(0)
    am_abs = atmosphere.get_absolute_airmass(airmass, pressure)
    cs = ineichen(solpos['apparent_zenith'], am_abs, linke_turbidity=3.0, dni_extra=dni_extra)

    best_azimuth = 180  # Default to South (optimal for Northern Hemisphere)
    best_total_irradiance = 0

    # Test azimuth angles in 10° increments
    for azimuth in range(0, 360, 10):
        total_irrad = irradiance.get_total_irradiance(
            surface_tilt=tilt,
            surface_azimuth=azimuth,
            solar_zenith=solpos['apparent_zenith'],
            solar_azimuth=solpos['azimuth'],
            dni=cs['dni'],
            ghi=cs['ghi'],
            dhi=cs['dhi']
        )

        # Sum irradiance across all test dates (weighted average for annual)
        total = total_irrad['poa_global'].sum()

        if total > best_total_irradiance:
            best_total_irradiance = total
            best_azimuth = azimuth

    # Fine-tune around the best angle (±10° in 1° steps)
    fine_start = max(0, best_azimuth - 10)
    fine_end = min(360, best_azimuth + 10)

    for azimuth in range(fine_start, fine_end):
        total_irrad = irradiance.get_total_irradiance(
            surface_tilt=tilt,
            surface_azimuth=azimuth,
            solar_zenith=solpos['apparent_zenith'],
            solar_azimuth=solpos['azimuth'],
            dni=cs['dni'],
            ghi=cs['ghi'],
            dhi=cs['dhi']
        )

        total = total_irrad['poa_global'].sum()

        if total > best_total_irradiance:
            best_total_irradiance = total
            best_azimuth = azimuth

    print(f"\n☀️ OPTIMAL AZIMUTH CALCULATION:")
    print(f"   Location: {lat:.4f}°, {lon:.4f}°")
    print(f"   Tilt: {tilt}°")
    print(f"   Optimal Azimuth: {best_azimuth}° ({'South' if 135 <= best_azimuth <= 225 else 'North' if best_azimuth < 45 or best_azimuth > 315 else 'East' if 45 <= best_azimuth < 135 else 'West'})")
    print(f"   Max Irradiance Sum: {best_total_irradiance:.0f} W/m²")

    return float(best_azimuth)

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

def create_solar_panel_sprite(width_px, height_px, azimuth):
    """
    Creates a realistic solar panel sprite with orientation

    Args:
        width_px: Width in pixels
        height_px: Height in pixels (already projected based on tilt)
        azimuth: Rotation angle in degrees

    Returns:
        Rotated RGBA image of solar panel
    """
    # Create base panel with realistic solar cell appearance
    panel = np.ones((int(height_px), int(width_px), 4), dtype=np.uint8) * 255

    # Dark blue/black base for solar cells
    panel[:, :, 0] = 25   # Blue
    panel[:, :, 1] = 35   # Green
    panel[:, :, 2] = 60   # Red
    panel[:, :, 3] = 255  # Alpha

    # Add cell grid pattern (6x10 cells typical for modern panels)
    cells_h = 6
    cells_w = 10
    cell_h = height_px / cells_h
    cell_w = width_px / cells_w

    # Draw cell borders (silver/gray lines)
    for i in range(1, cells_h):
        y = int(i * cell_h)
        cv2.line(panel, (0, y), (int(width_px), y), (180, 180, 180, 255), 1)

    for j in range(1, cells_w):
        x = int(j * cell_w)
        cv2.line(panel, (x, 0), (x, int(height_px)), (180, 180, 180, 255), 1)

    # Add frame border
    cv2.rectangle(panel, (0, 0), (int(width_px)-1, int(height_px)-1),
                  (60, 60, 60, 255), 2)

    # Add slight gradient to simulate light reflection
    gradient = np.linspace(0.8, 1.0, int(height_px))
    for i in range(3):
        panel[:, :, i] = (panel[:, :, i] * gradient[:, np.newaxis]).astype(np.uint8)

    # Rotate panel to match azimuth
    # OpenCV rotation: positive = counter-clockwise
    # Azimuth: 0° = North, 90° = East, 180° = South, 270° = West
    # We need to rotate the sprite so it "faces" the azimuth direction
    center = (width_px / 2, height_px / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, -azimuth, 1.0)

    # Calculate new bounding box to prevent clipping
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((height_px * sin) + (width_px * cos))
    new_h = int((height_px * cos) + (width_px * sin))

    # Adjust rotation matrix for new size
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(panel, rotation_matrix, (new_w, new_h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0, 0))

    return rotated

def overlay_panel_sprite(base_img, panel_polygon, sprite):
    """
    Overlays a solar panel sprite onto the base image at the polygon location

    Args:
        base_img: Background image (BGR)
        panel_polygon: Shapely Polygon defining panel location
        sprite: RGBA sprite image

    Returns:
        Updated base image with panel overlaid
    """
    # Get polygon bounds
    coords = np.array(panel_polygon.exterior.coords[:-1], dtype=np.float32)
    x, y, w, h = cv2.boundingRect(coords.astype(np.int32))

    # Resize sprite to match polygon size
    resized_sprite = cv2.resize(sprite, (w, h), interpolation=cv2.INTER_LINEAR)

    # Ensure bounds are within image
    if y < 0 or x < 0 or y+h > base_img.shape[0] or x+w > base_img.shape[1]:
        return base_img

    # Extract the region of interest
    roi = base_img[y:y+h, x:x+w]

    # Separate alpha channel
    sprite_bgr = resized_sprite[:, :, :3]
    alpha = resized_sprite[:, :, 3] / 255.0

    # Blend the sprite with the background
    for c in range(3):
        roi[:, :, c] = (alpha * sprite_bgr[:, :, c] +
                        (1 - alpha) * roi[:, :, c]).astype(np.uint8)

    base_img[y:y+h, x:x+w] = roi

    return base_img

def find_contiguous_sequences(row, gsd, panel_w=1.76, panel_spacing=0.05):
    """
    Finds contiguous sequences of panels within a row.

    Panels are considered contiguous if their centroid distance is within
    expected spacing (panel_width + panel_spacing + tolerance).

    Args:
        row: List of panel polygons sorted by X-coordinate
        gsd: Ground Sample Distance (meters per pixel)
        panel_w: Panel width in meters (default 1.76m)
        panel_spacing: Gap between panels in meters (default 0.05m)

    Returns:
        List of lists, where each inner list is a contiguous sequence
    """
    import math

    if not row:
        return []

    if len(row) == 1:
        return [row]

    # Calculate expected spacing between panel centroids
    # Normal spacing = panel width + gap between panels
    expected_spacing_m = panel_w + panel_spacing  # ~1.81m
    expected_spacing_px = expected_spacing_m / gsd

    # Allow 50% tolerance for slight variations (e.g., rotation, placement)
    max_gap_threshold = expected_spacing_px * 1.5

    print(f"      Gap detection: max distance = {max_gap_threshold:.1f}px (expected: {expected_spacing_px:.1f}px)")

    sequences = []
    current_sequence = [row[0]]

    for i in range(1, len(row)):
        prev_centroid = row[i-1].centroid
        curr_centroid = row[i].centroid

        # Calculate actual distance
        distance = math.sqrt((curr_centroid.x - prev_centroid.x)**2 +
                           (curr_centroid.y - prev_centroid.y)**2)

        if distance <= max_gap_threshold:  # Adjacent
            current_sequence.append(row[i])
        else:  # Gap detected (chimney, vent, etc.)
            print(f"         GAP detected: {distance:.1f}px > {max_gap_threshold:.1f}px threshold")
            sequences.append(current_sequence)
            current_sequence = [row[i]]

    # Add final sequence
    if current_sequence:
        sequences.append(current_sequence)

    return sequences


def generate_optimal_grid(sunny_mask, gsd, azimuth, tilt, target_count,
                         panel_w=1.76, panel_h=1.13, edge_margin=0.30,
                         panel_spacing=0.05, orientation="Portrait"):
    """
    Generates panel grid optimized for target count by trying multiple positions.

    For target_count <= 10:
    - Tries multiple horizontal and vertical offsets
    - Evaluates each configuration for contiguity
    - Returns best layout

    For target_count > 10:
    - Uses standard maximum capacity grid (no optimization needed)

    Args:
        sunny_mask: Binary mask of sunny area
        gsd: Ground Sample Distance (meters per pixel)
        azimuth: Panel orientation in degrees
        tilt: Panel tilt angle in degrees
        target_count: Desired number of panels
        panel_w: Physical panel width in meters
        panel_h: Physical panel height in meters
        edge_margin: Minimum distance from roof edge in meters
        panel_spacing: Gap between panels in meters
        orientation: "Portrait" or "Landscape"

    Returns:
        tuple: (panels_flat, rows_structure, contiguity_score, warning_msg)
    """
    from src.panel_optimization import select_panels_from_grid, score_contiguity

    # For large arrays, skip optimization (performance)
    if target_count > 10:
        print(f"\n⚡ LARGE ARRAY MODE (>{10} panels) - Using standard serpentine placement")
        panels_flat, rows_structure = generate_panel_grid(
            sunny_mask, gsd, azimuth, tilt, panel_w, panel_h,
            edge_margin, panel_spacing, orientation
        )
        if not panels_flat:
            return [], [], 0, "No valid panel positions found"

        # Select panels and score
        selected, selected_rows, score, warning = select_panels_from_grid(
            panels_flat, rows_structure, target_count
        )
        print(f"\n   ✅ MODE: SERPENTINE (Multi-row, {len(selected_rows)} rows)")
        return selected, selected_rows, score, warning

    # For small arrays (≤10 panels), optimize placement
    print(f"\n🔍 OPTIMIZING PLACEMENT for {target_count} panels...")

    # Generate ONE grid covering entire sunny area (no offset)
    all_panels, all_rows = generate_panel_grid(
        sunny_mask, gsd, azimuth, tilt, panel_w, panel_h,
        edge_margin, panel_spacing, orientation
    )

    if not all_panels:
        return [], [], 0, "No valid panel positions found"

    total_capacity = len(all_panels)
    print(f"   Total capacity: {total_capacity} panels across {len(all_rows)} rows")

    if total_capacity < target_count:
        print(f"   ⚠️ Cannot fit {target_count} panels (max: {total_capacity})")
        selected, selected_rows, score, warning = select_panels_from_grid(
            all_panels, all_rows, total_capacity
        )
        return selected, selected_rows, score, warning

    # Try to find best placement by testing different row combinations
    best_score = -9999
    best_config = None
    best_warning = None

    # Strategy 1: Try to find single-row placement
    print(f"\n   🔍 STRATEGY 1: Searching for SINGLE-ROW placement...")
    for row_idx, row in enumerate(all_rows):
        # Find contiguous sequences within this row
        contiguous_sequences = find_contiguous_sequences(row, gsd, panel_w=panel_w, panel_spacing=panel_spacing)

        # Show sequence details
        seq_info = ", ".join([f"{len(seq)} panels" for seq in contiguous_sequences])
        print(f"   Row {row_idx + 1}: {len(row)} total panels → {len(contiguous_sequences)} sequences ({seq_info})")

        # Check each contiguous sequence
        for seq_idx, sequence in enumerate(contiguous_sequences):
            if len(sequence) >= target_count:
                # This sequence can fit all panels contiguously!
                selected = sequence[:target_count]
                selected_rows = [selected]
                score, is_acceptable, warning = score_contiguity(selected, selected_rows, target_count)

                print(f"      ✓ Sequence {seq_idx + 1} can fit {target_count} panels!")
                print(f"      Score={score}, contiguous={is_acceptable}")

                # This should always be acceptable now since we're using contiguous sequences
                if is_acceptable and score > best_score:
                    best_score = score
                    best_config = (selected, selected_rows)
                    best_warning = warning

                    # Perfect single-row contiguous placement - use it immediately
                    print(f"\n   ✅ MODE: SINGLE-ROW (Row {row_idx + 1}, Sequence {seq_idx + 1})")
                    print(f"   All {target_count} panels in ONE continuous row")
                    return selected, selected_rows, score, warning

    # Strategy 2: If no single row works, use serpentine (multi-row) placement
    if best_config is None or best_score < 0:
        print(f"\n   🔍 STRATEGY 2: No single-row possible, using SERPENTINE...")

        # Use ALL contiguous sequences as separate rows (not just longest from each row)
        # This maximizes capacity while maintaining no-gaps guarantee
        contiguous_rows = []
        for row in all_rows:
            sequences = find_contiguous_sequences(row, gsd, panel_w=panel_w, panel_spacing=panel_spacing)
            # Add ALL sequences as separate rows
            for sequence in sequences:
                if sequence:  # Skip empty sequences
                    contiguous_rows.append(sequence)

        print(f"   Total contiguous sequences available: {len(contiguous_rows)}")
        total_capacity = sum(len(seq) for seq in contiguous_rows)
        print(f"   Total panel capacity: {total_capacity} panels")

        selected, selected_rows, score, warning = select_panels_from_grid(
            all_panels, contiguous_rows, target_count
        )

        print(f"\n   ✅ MODE: SERPENTINE (Multi-row)")
        print(f"   {len(selected_rows)} rows used, {len(selected)} total panels")
        return selected, selected_rows, score, warning

    # Return best configuration found (if any single-row config exists but wasn't perfect)
    if best_config:
        print(f"\n   ✅ MODE: SINGLE-ROW (Best available)")
        print(f"   {len(best_config[1])} rows, score={best_score}")
        return best_config[0], best_config[1], best_score, best_warning
    else:
        # Shouldn't reach here, but fallback to serpentine
        print(f"\n   ⚠️ No valid configuration found, falling back to serpentine...")
        selected, selected_rows, score, warning = select_panels_from_grid(
            all_panels, all_rows, target_count
        )
        return selected, selected_rows, score, warning

def generate_panel_grid(sunny_mask, gsd, azimuth, tilt, panel_w=1.76, panel_h=1.13,
                        edge_margin=0.30, panel_spacing=0.05, orientation="Portrait",
                        x_offset=0.0, y_offset=0.0):
    """
    Generates panels row-by-row for simple serpentine wiring.

    Args:
        sunny_mask: Binary mask of sunny area
        gsd: Ground Sample Distance (meters per pixel)
        azimuth: Panel orientation in degrees
        tilt: Panel tilt angle in degrees (0° = flat, 38° = typical pitched)
        panel_w: Physical panel width in meters (default 1.76m)
        panel_h: Physical panel height in meters (default 1.13m)
        edge_margin: Minimum distance from roof edge in meters (default 0.30m)
        panel_spacing: Gap between panels in meters (default 0.05m)
        orientation: "Portrait" (vertical) or "Landscape" (horizontal)
        x_offset: Horizontal offset in meters (for grid shifting)
        y_offset: Vertical offset in meters (for grid shifting)

    Returns:
        Tuple: (all_panels_flat, rows_structure)
            - all_panels_flat: Flat list of all panels
            - rows_structure: List of lists, each inner list is a row of panels
    """
    from shapely.geometry import Polygon
    from shapely import affinity
    from src.geometry_utils import mask_to_polygon

    sunny_pts = mask_to_polygon(sunny_mask)
    if not sunny_pts:
        return []

    # Handle orientation (swap dimensions for landscape)
    if orientation == "Landscape":
        actual_w = panel_h  # 1.13m wide
        actual_h = panel_w  # 1.76m tall
    else:  # Portrait
        actual_w = panel_w  # 1.76m wide
        actual_h = panel_h  # 1.13m tall

    # Width is NOT affected by tilt in top-down view
    projected_w = actual_w
    projected_h = actual_h * math.cos(math.radians(tilt))

    # Convert to pixels
    pw_px = projected_w / gsd
    ph_px = projected_h / gsd
    edge_margin_px = edge_margin / gsd
    spacing_px = panel_spacing / gsd

    # Create sunny polygon with edge margin buffer
    sunny_poly = Polygon(sunny_pts).buffer(-edge_margin_px)

    if sunny_poly.is_empty or sunny_poly.area < (pw_px * ph_px):
        return []

    center = sunny_poly.centroid

    # Rotate polygon to align with azimuth
    aligned_poly = affinity.rotate(sunny_poly, -azimuth, origin=center)

    minx, miny, maxx, maxy = aligned_poly.bounds

    # Apply grid offsets (in pixels)
    x_offset_px = x_offset / gsd
    y_offset_px = y_offset / gsd
    minx += x_offset_px
    miny += y_offset_px

    # Step sizes include spacing
    step_x = pw_px + spacing_px
    step_y = ph_px + spacing_px

    # Generate panels ROW BY ROW
    all_rows = []

    y = miny
    while y + ph_px <= maxy:
        row_panels = []
        x = minx

        while x + pw_px <= maxx:
            # Create panel rectangle
            p = Polygon([
                (x, y),
                (x + pw_px, y),
                (x + pw_px, y + ph_px),
                (x, y + ph_px)
            ])

            # Check if panel fits
            if aligned_poly.contains(p.buffer(-0.5)):
                row_panels.append(p)

            x += step_x

        # Keep row if it has at least 1 panel
        if len(row_panels) >= 1:
            all_rows.append(row_panels)

        y += step_y

    if not all_rows:
        return [], []

    # Rotate all panels back to real-world orientation (keeping row structure)
    rotated_rows = []
    for row in all_rows:
        rotated_row = [affinity.rotate(p, azimuth, origin=center) for p in row]
        rotated_rows.append(rotated_row)

    # Flatten for validation
    all_panels_flat = []
    for row in rotated_rows:
        all_panels_flat.extend(row)

    # Final validation: verify entire panel footprint is within sunny mask
    valid_rows = []
    for row in rotated_rows:
        valid_row = []
        for panel in row:
            panel_coords = np.array(panel.exterior.coords[:-1], dtype=np.int32)

            # Get bounding box of panel
            min_px = int(np.min(panel_coords[:, 0]))
            max_px = int(np.max(panel_coords[:, 0]))
            min_py = int(np.min(panel_coords[:, 1]))
            max_py = int(np.max(panel_coords[:, 1]))

            # Bounds check
            if min_px < 0 or min_py < 0 or max_py >= sunny_mask.shape[0] or max_px >= sunny_mask.shape[1]:
                continue

            # Rasterize panel polygon onto a local mask and compare with sunny mask
            local_h = max_py - min_py + 1
            local_w = max_px - min_px + 1
            local_panel_mask = np.zeros((local_h, local_w), dtype=np.uint8)
            local_coords = panel_coords - np.array([min_px, min_py])
            cv2.fillConvexPoly(local_panel_mask, local_coords, 255)

            local_sunny = sunny_mask[min_py:max_py + 1, min_px:max_px + 1]

            panel_pixels = int(np.sum(local_panel_mask > 0))
            overlap_pixels = int(np.sum((local_panel_mask > 0) & (local_sunny > 0)))

            # Allow at most 2 pixels of rounding error from coordinate conversion
            if panel_pixels > 0 and (panel_pixels - overlap_pixels) <= 2:
                valid_row.append(panel)

        # CRITICAL FIX: Split rows with gaps into separate contiguous sequences
        # This prevents "9 panels" being reported when they have gaps
        if valid_row:
            # Sort by X coordinate first
            valid_row.sort(key=lambda p: p.centroid.x)

            # Find contiguous sequences within this row
            # Pass GSD for accurate gap detection
            contiguous_sequences = find_contiguous_sequences(valid_row, gsd, panel_w=actual_w, panel_spacing=panel_spacing)

            # Add each contiguous sequence as a separate row
            for sequence in contiguous_sequences:
                if len(sequence) >= 1:
                    valid_rows.append(sequence)

    # Flatten valid panels
    valid_panels_flat = []
    for row in valid_rows:
        valid_panels_flat.extend(row)

    print(f"\n📐 GRID GENERATION:")
    print(f"   Contiguous rows created: {len(valid_rows)}")
    print(f"   (Each row is guaranteed to have NO gaps)")
    for i, row in enumerate(valid_rows):
        # Calculate physical span
        if len(row) > 1:
            x_coords = [p.centroid.x for p in row]
            span = max(x_coords) - min(x_coords)
            print(f"      Row {i+1}: {len(row)} panels (contiguous, span: {span:.0f}px)")
        else:
            print(f"      Row {i+1}: {len(row)} panel")
    print(f"   Total panels: {len(valid_panels_flat)}")

    return valid_panels_flat, valid_rows