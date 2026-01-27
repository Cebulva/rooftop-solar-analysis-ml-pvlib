import streamlit as st
import numpy as np
import cv2
import math
from shapely.geometry import Polygon
from shapely import affinity
import ui_components as ui

# Updated Imports
from src.solar_engine import (
    get_masked_roof_array, 
    analyze_roof_geometry,  # NEW: Enhanced detection
    analyze_roof_texture,   # DEPRECATED: Kept for fallback
    draw_azimuth_arrow,
    get_sunny_polygon_mask,
    calculate_solar_potential,
    calculate_global_gsd,
    # Import configuration constants from solar_engine (single source of truth)
    MIN_FLAT_ROOF_COVERAGE,
    TEXTURE_VARIANCE_THRESHOLD,
    MIN_BRIGHTNESS_DIFF
)
from src.geometry_utils import calculate_azimuth, mask_to_polygon

# ==========================================
# ⚙️ UI CONFIGURATION (Tweak these)
# ==========================================
# Detection thresholds are configured in solar_engine.py
# Only UI-specific settings here
SHOW_DEBUG_INFO = True          # Display detection reasoning in UI
DEFAULT_PITCHED_TILT = 38.0     # Default tilt angle for pitched roofs (degrees)
DEFAULT_FLAT_TILT = 10.0        # Default tilt angle for flat roofs (degrees, optimal mounting)

# Solar Panel String Configuration
MIN_PANELS_PER_STRING = 2       # Minimum panels in series (inverter requirement)
MAX_PANELS_PER_STRING = 20      # Maximum panels in series (voltage limit)
TYPICAL_MIN_STRING = 8          # Typical minimum for efficiency
TYPICAL_MAX_STRING = 15         # Typical maximum for efficiency
# ==========================================

def update_azimuth():
    st.session_state.data["user_azimuth"] = float(st.session_state.az_slider_widget)
    # Recalculate irradiance when azimuth changes
    recalculate_irradiance()

def update_threshold():
    st.session_state.data["sun_threshold"] = int(st.session_state.sun_slider_widget)

def recalculate_irradiance():
    """Real-time irradiance calculation when azimuth or tilt changes"""
    if "final_poly" in st.session_state.data:
        lat = st.session_state.data.get("confirmed_lat")
        lon = st.session_state.data.get("confirmed_lon")
        user_azimuth = st.session_state.data.get("user_azimuth", 180)
        user_tilt = st.session_state.data.get("user_tilt", 38)
        
        irrad_val = calculate_solar_potential(lat, lon, user_tilt, user_azimuth)
        st.session_state.data["current_irradiance"] = irrad_val

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

def generate_panel_grid(sunny_mask, gsd, azimuth, tilt, panel_w=1.76, panel_h=1.13):
    """
    Creates a grid of panels (Modern Trina/Jinko Glass-Glass dimensions).
    Adjusts ONLY the VISUAL HEIGHT based on tilt angle (Cosine Projection).
    Width remains constant as it's unaffected by tilt in top-down view.
    
    CRITICAL: Panels must be FULLY contained within the sunny_mask (usable space).
    
    Args:
        sunny_mask: Binary mask of sunny area
        gsd: Ground Sample Distance (meters per pixel)
        azimuth: Panel orientation in degrees
        tilt: Panel tilt angle in degrees (0° = flat, 38° = typical pitched)
        panel_w: Physical panel width in meters (default 1.76m)
        panel_h: Physical panel height in meters (default 1.13m)
    """
    sunny_pts = mask_to_polygon(sunny_mask)
    if not sunny_pts:
        return []
    
    # CRITICAL FIX: Width is NOT affected by tilt in top-down view
    # Only height appears compressed due to viewing angle
    projected_w = panel_w  # Width stays constant! ✓
    projected_h = panel_h * math.cos(math.radians(tilt))  # Height compressed by cosine
    
    # Convert meters to pixels FIRST (before any rotation)
    pw_px = projected_w / gsd  # Width in pixels (constant)
    ph_px = projected_h / gsd  # Height in pixels (varies with tilt)
    
    # Create the sunny polygon with a small buffer to avoid edge issues
    sunny_poly = Polygon(sunny_pts).buffer(-1.0)  # Slightly more conservative buffer
    
    if sunny_poly.is_empty or sunny_poly.area < (pw_px * ph_px):
        return []  # Not enough space for even one panel
    
    # Get the oriented bounding box (rotated rectangle that fits the sunny area)
    # This helps place panels more efficiently in non-rectangular sunny areas
    center = sunny_poly.centroid
    
    # Rotate polygon to align with azimuth (0° = panels facing north)
    aligned_poly = affinity.rotate(sunny_poly, -azimuth, origin=center)
    
    minx, miny, maxx, maxy = aligned_poly.bounds
    
    # Generate grid with small spacing between panels (0.05m = 5cm gap for maintenance)
    spacing = 0.05 / gsd  # Convert 5cm to pixels
    step_x = pw_px + spacing
    step_y = ph_px + spacing
    
    aligned_panels = []
    
    # Grid generation: row by row, left to right
    y = miny
    while y + ph_px <= maxy:
        x = minx
        while x + pw_px <= maxx:
            # Create panel rectangle
            p = Polygon([
                (x, y), 
                (x + pw_px, y), 
                (x + pw_px, y + ph_px), 
                (x, y + ph_px)
            ])
            
            # Check if panel fits within the aligned polygon
            # Use a small buffer to be slightly more permissive (accounts for numerical precision)
            if aligned_poly.contains(p.buffer(-0.5)):
                aligned_panels.append(p)
            
            x += step_x
        y += step_y
    
    if not aligned_panels:
        return []
    
    # Rotate all panels back to real-world orientation
    rotated_panels = [affinity.rotate(p, azimuth, origin=center) for p in aligned_panels]
    
    # FINAL VALIDATION: Verify panels are fully within the sunny mask
    # This catches any edge cases from rotation or numerical imprecision
    valid_panels = []
    
    for panel in rotated_panels:
        panel_coords = np.array(panel.exterior.coords[:-1], dtype=np.int32)
        
        # Sample multiple points across the panel (not just corners)
        # This is more robust for rotated panels
        is_valid = True
        
        # Check corners
        for corner in panel_coords:
            x, y = int(corner[0]), int(corner[1])
            
            # Boundary check
            if x < 0 or y < 0 or y >= sunny_mask.shape[0] or x >= sunny_mask.shape[1]:
                is_valid = False
                break
            
            # Mask check
            if sunny_mask[y, x] == 0:
                is_valid = False
                break
        
        # Additionally check center point (catches panels that span across sunny/shadow boundary)
        if is_valid:
            center_x = int(np.mean(panel_coords[:, 0]))
            center_y = int(np.mean(panel_coords[:, 1]))
            
            if (center_x < 0 or center_y < 0 or 
                center_y >= sunny_mask.shape[0] or center_x >= sunny_mask.shape[1] or
                sunny_mask[center_y, center_x] == 0):
                is_valid = False
        
        if is_valid:
            valid_panels.append(panel)
    
    return valid_panels

def show():
    st.header("Step 3b: Solar And Irradiance Analysis")

    if "final_poly" not in st.session_state.data:
        st.warning("Please complete the roof refinement in Step 2 first.")
        return

    # Back button
    if st.button("⬅️ Back to Questionnaire", key="back_to_step3a"):
        st.session_state.step = 3
        st.rerun()

    res = st.session_state.data["res"]
    lat = st.session_state.data["confirmed_lat"]
    lon = st.session_state.data["confirmed_lon"]

    # 1. INITIALIZATION & DATA RETRIEVAL
    # Get the recommendation from the questionnaire (Stage 3a)
    recommended_limit = st.session_state.data.get("recommended_count", 20)
    consumption_inputs = st.session_state.data.get("consumption_inputs", {})
    annual_kwh = consumption_inputs.get("annual_kwh", 3500)
    breakdown = consumption_inputs.get("breakdown", {})

    # Show consumption summary and recommendation from Stage 3a
    with st.container(border=True):
        st.subheader("📊 Your Estimated Annual Consumption")
        col_cons1, col_cons2 = st.columns([2, 1])

        with col_cons1:
            for item, kwh in breakdown.items():
                if item != 'Total':
                    st.write(f"- {item}: {kwh:,} kWh")

        with col_cons2:
            st.metric("Total", f"{annual_kwh:,} kWh/year")
            recommended_kwp = (recommended_limit * 440) / 1000
            st.metric("Recommended", f"{recommended_limit} panels ({recommended_kwp:.1f} kWp)")

    # Initialize the target count in session state if not present
    if "target_panel_count" not in st.session_state.data:
        st.session_state.data["target_panel_count"] = recommended_limit

    # 2. Geometry and Masking
    mask, roof_only = get_masked_roof_array(res["zoom_img"], st.session_state.data["final_poly"])
    gsd = calculate_global_gsd(lat, zoom=19) 
    pixel_area_m2 = gsd ** 2
    
    total_area_m2 = np.sum(mask > 0) * pixel_area_m2
    
    current_threshold = st.session_state.data.get("sun_threshold", 25)
    sun_mask = get_sunny_polygon_mask(roof_only, mask, threshold_offset=current_threshold)
    usable_area_m2 = np.sum(sun_mask > 0) * pixel_area_m2

    # 3. Analysis - IMPROVED ROOF TYPE DETECTION
    if "auto_roof_type" not in st.session_state.data:
        # Use enhanced detection with multiple heuristics
        detected_type, confidence, debug_info = analyze_roof_geometry(roof_only, mask, sun_mask)
        auto_azimuth = calculate_azimuth(st.session_state.data["final_poly"], img=roof_only)
        
        # Set default tilt based on roof type
        default_tilt = DEFAULT_PITCHED_TILT if detected_type == "Pitched" else DEFAULT_FLAT_TILT
        
        st.session_state.data.update({
            "auto_roof_type": detected_type,
            "user_azimuth": float(auto_azimuth),
            "user_tilt": default_tilt,
            "detection_confidence": confidence,
            "detection_debug": debug_info
        })
        
        # Display detection reasoning in console/logs
        print(f"\n🏠 ROOF TYPE DETECTION:")
        print(f"   Result: {detected_type} (Confidence: {confidence:.1%})")
        print(f"   Reason: {debug_info.get('reason', 'N/A')}")
        print(f"   Coverage Ratio: {debug_info.get('coverage_ratio', 0):.1%}")
        print(f"   Brightness Range: {debug_info.get('brightness_range', 0):.0f}")
        print(f"   Texture StdDev: {debug_info.get('std_dev', 0):.1f}")
        print(f"   Default Tilt: {default_tilt}°")

    current_azimuth = st.session_state.data["user_azimuth"]
    user_tilt = st.session_state.data["user_tilt"]

    # Calculate real-time irradiance if not already calculated
    if "current_irradiance" not in st.session_state.data:
        recalculate_irradiance()

    current_irradiance = st.session_state.data.get("current_irradiance", 0)

    # 4. Generate Panel Grid (Limited by Target Count)
    all_possible_panels = generate_panel_grid(sun_mask, gsd, current_azimuth, user_tilt)
    
    # Debug logging
    print(f"\n🔧 PANEL GENERATION DEBUG:")
    print(f"   GSD: {gsd:.4f} m/pixel")
    print(f"   Panel dimensions: {1.76}m × {1.13 * math.cos(math.radians(user_tilt)):.2f}m (projected)")
    print(f"   Panel size in pixels: {1.76/gsd:.1f} × {(1.13 * math.cos(math.radians(user_tilt)))/gsd:.1f}")
    print(f"   Azimuth: {current_azimuth}°")
    print(f"   Tilt: {user_tilt}°")
    print(f"   Total panels generated: {len(all_possible_panels)}")
    
    # Apply the limit chosen by the user or the questionnaire
    limit = st.session_state.data["target_panel_count"]
    panels = all_possible_panels[:limit] 
    selected_count = len(panels)

    # 5. UI Layout
    col_main, col_R = st.columns([4, 1.5])
    
    with col_main:
        display_img = roof_only.copy()
        
        # Yellow Mask Overlay
        mask_overlay = np.zeros_like(display_img)
        mask_overlay[sun_mask > 0] = (0, 255, 255)
        display_img = cv2.addWeighted(display_img, 1.0, mask_overlay, 0.3, 0)
        
        display_img = draw_azimuth_arrow(display_img, current_azimuth)
        
        # Create solar panel sprite (dimensions in pixels based on panel size and tilt)
        # IMPORTANT: Width is constant in top-down view, only height is affected by tilt
        panel_w_px = 1.76 / gsd  # Standard panel width in pixels (CONSTANT)
        panel_h_px = (1.13 * math.cos(math.radians(user_tilt))) / gsd  # Projected height (VARIABLE)
        
        panel_sprite = create_solar_panel_sprite(panel_w_px, panel_h_px, current_azimuth)
        
        # Overlay each panel as a sprite instead of green rectangle
        for p in panels:
            display_img = overlay_panel_sprite(display_img, p, panel_sprite)

        st.image(display_img, use_container_width=True)
        
        with st.expander("🛠️ Analysis And Adjustments", expanded=True):
            # Panel Count Slider with String Validation
            max_capacity = len(all_possible_panels)
            
            # Enforce minimum panel count for series string
            min_installable = max(MIN_PANELS_PER_STRING, 1)
            
            # Show warning if roof capacity is below minimum string requirement
            if max_capacity < MIN_PANELS_PER_STRING:
                st.error(f"⚠️ Roof capacity ({max_capacity} panels) is below the minimum string requirement "
                        f"({MIN_PANELS_PER_STRING} panels in series). Installation not viable with current settings.")
            elif max_capacity < TYPICAL_MIN_STRING:
                st.warning(f"⚠️ Roof capacity ({max_capacity} panels) is below typical minimum "
                          f"({TYPICAL_MIN_STRING} panels). Consider adjusting settings or using micro-inverters.")
            
            # Determine default target count (capped by string limits)
            default_target = min(
                st.session_state.data.get("target_panel_count", recommended_limit),
                max_capacity,
                MAX_PANELS_PER_STRING
            )
            
            # Ensure default meets minimum requirement
            default_target = max(default_target, min_installable)
            
            current_count = st.slider(
                "Number of Panels to Install", 
                min_value=min_installable, 
                max_value=min(max_capacity, MAX_PANELS_PER_STRING), 
                value=int(default_target), 
                key="panel_slider_widget",
                help=(f"String limits: Min {MIN_PANELS_PER_STRING}, Max {MAX_PANELS_PER_STRING}. "
                      f"Typical: {TYPICAL_MIN_STRING}-{TYPICAL_MAX_STRING} panels per string. "
                      f"Questionnaire recommended: {recommended_limit}"),
                on_change=lambda: st.session_state.data.update({
                    "target_panel_count": st.session_state.panel_slider_widget
                })
            )
            
            # Visual feedback on string configuration
            if current_count < TYPICAL_MIN_STRING:
                st.caption(f"⚠️ Below typical minimum ({TYPICAL_MIN_STRING}). May require special inverter configuration.")
            elif current_count > TYPICAL_MAX_STRING:
                st.caption(f"ℹ️ Above typical maximum ({TYPICAL_MAX_STRING}). May require multiple strings.")
            else:
                st.caption(f"✅ Within typical range ({TYPICAL_MIN_STRING}-{TYPICAL_MAX_STRING} panels).")

            c1, c2 = st.columns(2)
            selected_type = c1.selectbox("Roof Form", ["Pitched", "Flat"], 
                                       index=0 if st.session_state.data["auto_roof_type"] == "Pitched" else 1)
            
            # Show detection confidence and reasoning if enabled
            if SHOW_DEBUG_INFO and "detection_debug" in st.session_state.data:
                debug = st.session_state.data["detection_debug"]
                confidence = st.session_state.data.get("detection_confidence", 0)
                
                c2.metric("Detection", f"{confidence:.0%}", 
                         help=debug.get("reason", "Auto-detected roof type"))
                
                # Expandable debug details
                with st.expander("🔍 Detection Details", expanded=False):
                    st.write(f"**Reasoning:** {debug.get('reason', 'N/A')}")
                    st.write(f"**Coverage Ratio:** {debug.get('coverage_ratio', 0):.1%} "
                            f"(Flat if ≥ {MIN_FLAT_ROOF_COVERAGE:.0%})")
                    st.write(f"**Brightness Range:** {debug.get('brightness_range', 0):.0f}/255 "
                            f"(Pitched if ≥ 30)")
                    st.write(f"**Texture Variance:** {debug.get('std_dev', 0):.1f} "
                            f"(Pitched if > 15)")
            
            # Re-run if type changes to update tilt
            if selected_type != st.session_state.data["auto_roof_type"]:
                st.session_state.data["auto_roof_type"] = selected_type
                # Set appropriate default tilt for the roof type
                st.session_state.data["user_tilt"] = DEFAULT_PITCHED_TILT if selected_type == "Pitched" else DEFAULT_FLAT_TILT
                recalculate_irradiance()
                st.rerun()

            # Tilt Angle Slider - visible for PITCHED roofs
            if selected_type == "Pitched":
                st.slider(
                    "Panel Tilt Angle (°)", 
                    min_value=10, 
                    max_value=60, 
                    value=int(st.session_state.data["user_tilt"]),
                    key="tilt_slider_widget",
                    help=f"Typical range: 25-45°. Default: {DEFAULT_PITCHED_TILT}°",
                    on_change=lambda: st.session_state.data.update({
                        "user_tilt": float(st.session_state.tilt_slider_widget)
                    }) or recalculate_irradiance()
                )
            else:
                # For flat roofs, show info but don't allow adjustment (optimal mounting angle)
                st.info(f"ℹ️ Flat roof panels use {DEFAULT_FLAT_TILT}° mounting angle for optimal drainage and performance.")
            
            st.slider("Solar Orientation (Azimuth °)", 0, 359, int(current_azimuth), 
                      key="az_slider_widget", on_change=update_azimuth,
                      help="Watch the Irradiance Potential change as you rotate!")
            
            st.slider("Shadow Tolerance (Threshold)", 0, 100, int(current_threshold), 
                      key="sun_slider_widget", on_change=update_threshold)

    with col_R:
        st.markdown("### 📊 Global Metrics")
        st.metric("Total Roof Area", f"{total_area_m2:.1f} m²")
        st.metric("Usable Space", f"{usable_area_m2:.1f} m²")
        
        # String Configuration Display
        string_info = f"{selected_count}"
        if selected_count < MIN_PANELS_PER_STRING:
            string_info += " ⚠️"
        elif selected_count > MAX_PANELS_PER_STRING:
            string_info += " ⚠️"
        
        st.metric("Selected Panels", string_info, 
                 help=f"String limits: {MIN_PANELS_PER_STRING}-{MAX_PANELS_PER_STRING} panels. "
                      f"Typical: {TYPICAL_MIN_STRING}-{TYPICAL_MAX_STRING}")
        
        system_kwp = (selected_count * 440) / 1000
        st.metric("System Size", f"{system_kwp:.2f} kWp")
        
        # String configuration advice
        if selected_count >= MIN_PANELS_PER_STRING and selected_count <= MAX_PANELS_PER_STRING:
            strings_needed = math.ceil(selected_count / TYPICAL_MAX_STRING)
            if strings_needed == 1:
                st.caption(f"✅ Single string configuration")
            else:
                panels_per_string = selected_count // strings_needed
                st.caption(f"ℹ️ Suggest {strings_needed} strings of ~{panels_per_string} panels each")
        
        # NEW: Real-time Irradiance Potential
        st.metric("☀️ Irradiance Potential", 
                  f"{current_irradiance:.0f} W/m²",
                  help="Real-time solar irradiance at current azimuth. Rotate to see changes!")
        
        # Calculate annual energy production estimate
        # Simplified calculation: kWp × irradiance × hours × efficiency
        peak_sun_hours = 4.5  # Average for Central Europe
        system_efficiency = 0.85  # Account for losses
        annual_production = system_kwp * peak_sun_hours * 365 * (current_irradiance / 1000) * system_efficiency
        
        st.metric("Est. Annual Production", 
                  f"{annual_production:,.0f} kWh/year",
                  help="Estimated yearly energy production")
        
        # Coverage percentage
        coverage_pct = (annual_production / annual_kwh * 100) if annual_kwh > 0 else 0
        st.metric("Coverage", 
                  f"{coverage_pct:.0f}%",
                  help="Percentage of your consumption covered by solar")
        
        if st.button("Run Simulation And Generate Report ☀️", type="primary", use_container_width=True):
            # Validate minimum panel count before generating report
            if selected_count < MIN_PANELS_PER_STRING:
                st.error(f"Cannot generate report: Minimum {MIN_PANELS_PER_STRING} panels required for series connection.")
            else:
                st.session_state.data["solar_results"] = {
                    "total_roof_area_m2": total_area_m2,
                    "usable_roof_area_m2": usable_area_m2,
                    "panel_count": selected_count,
                    "system_kwp": system_kwp,
                    "azimuth": current_azimuth,
                    "tilt_angle": user_tilt,
                    "roof_form": st.session_state.data["auto_roof_type"],
                    "irradiance_potential": current_irradiance,
                    "annual_production_kwh": annual_production,
                    "coverage_percentage": coverage_pct,
                    "string_configuration": {
                        "strings_needed": max(1, math.ceil(selected_count / TYPICAL_MAX_STRING)),
                        "panels_per_string": selected_count // max(1, math.ceil(selected_count / TYPICAL_MAX_STRING))
                    }
                }
                st.session_state.step = 5
                st.rerun()