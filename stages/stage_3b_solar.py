import streamlit as st
import numpy as np
import cv2
import math
from shapely.geometry import Polygon
from shapely import affinity
import ui_components as ui


# Solar Engine - Core solar calculations and roof detection
from src.solar_engine import (
    get_masked_roof_array,
    analyze_roof_geometry,  # Enhanced roof type detection
    analyze_roof_texture,   # DEPRECATED: Kept for fallback
    draw_azimuth_arrow,
    get_sunny_polygon_mask,
    calculate_solar_potential,
    calculate_global_gsd,
    create_solar_panel_sprite,  # Panel visualization
    overlay_panel_sprite,
    generate_panel_grid,
    generate_optimal_grid,  # NEW: Grid optimization for small panel counts
    # Import configuration constants from solar_engine (single source of truth)
    MIN_FLAT_ROOF_COVERAGE,
    TEXTURE_VARIANCE_THRESHOLD,
    MIN_BRIGHTNESS_DIFF
)

# Geometry utilities
from src.geometry_utils import calculate_azimuth, mask_to_polygon

# Inquiry management
from src.inquiry_manager import save_inquiry

# Panel optimization - Simple row-by-row selection
from src.panel_optimization import (
    select_panels_from_grid
)

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

# Panel Installation Margins
ROOF_EDGE_MARGIN = 0.25         # Minimum distance from roof edge (meters) - safety margin
PANEL_SPACING = 0.05            # Gap between adjacent panels (meters) - maintenance access

# Panel Orientation Options
# Portrait: 1.76m wide × 1.13m tall (vertical strings, taller than wide)
# Landscape: 1.13m wide × 1.76m tall (horizontal strings, wider than tall)
DEFAULT_PANEL_ORIENTATION = "Portrait"  # "Portrait" or "Landscape"

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

    # Check if shadow tolerance has been confirmed (used throughout the UI)
    shadow_confirmed = st.session_state.data.get("shadow_tolerance_confirmed", False)


    # Show consumption summary only after shadow tolerance is confirmed
    if shadow_confirmed:
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
    
    # Shadow tolerance: negative values = stricter (only brightest), positive = more inclusive
    current_threshold = st.session_state.data.get("sun_threshold", 15)
    
    # CRITICAL FIX: Cache sun_mask to prevent coverage changes
    # Only recalculate sun_mask when threshold actually changes
    
    # Track if this is the first time we're calculating the sun_mask
    is_first_calculation = "cached_sun_threshold" not in st.session_state.data
    cached_threshold = st.session_state.data.get("cached_sun_threshold")
    
    print(f"\n🎭 SUN_MASK CACHE STATUS:")
    print(f"   is_first_calculation: {is_first_calculation}")
    print(f"   cached_threshold: {cached_threshold}")
    print(f"   current_threshold: {current_threshold}")
    print(f"   Will recalculate sun_mask: {is_first_calculation or cached_threshold != current_threshold}")
    
    if is_first_calculation or cached_threshold != current_threshold:
        # Recalculate sun_mask when: (1) first time, or (2) threshold changes
        print(f"   🔄 GENERATING NEW SUN_MASK...")
        sun_mask = get_sunny_polygon_mask(roof_only, mask, threshold_offset=current_threshold)
        st.session_state.data["cached_sun_mask"] = sun_mask.copy()  # Cache it
        st.session_state.data["cached_sun_threshold"] = current_threshold
        print(f"   ✅ Sun_mask generated and cached")
    else:
        # Use cached sun_mask - prevents coverage from changing
        print(f"   ♻️ USING CACHED SUN_MASK")
        sun_mask = st.session_state.data["cached_sun_mask"]
    
    usable_area_m2 = np.sum(sun_mask > 0) * pixel_area_m2

    # 3. Analysis - IMPROVED ROOF TYPE DETECTION
    # Cache detection results to prevent coverage ratio from changing when adjusting other settings
    # Re-run detection only if sun_mask has actually changed (which happens when threshold changes)
    
    # Check if we have valid cached results
    has_cached_results = all([
        "detection_confidence" in st.session_state.data,
        "detection_debug" in st.session_state.data,
        "detected_roof_type" in st.session_state.data
    ])
    
    # Debug logging
    print(f"\n🔍 DETECTION CACHE STATUS:")
    print(f"   has_cached_results: {has_cached_results}")
    print(f"   is_first_calculation: {is_first_calculation}")
    print(f"   cached_threshold: {cached_threshold}")
    print(f"   current_threshold: {current_threshold}")
    print(f"   Will run detection: {not has_cached_results or is_first_calculation or cached_threshold != current_threshold}")
    
    # Re-detect only if: (1) no cache exists yet, or (2) sun_mask was recalculated
    if not has_cached_results or is_first_calculation or cached_threshold != current_threshold:
        # Run detection when cache is missing or sun_mask changed
        print(f"   🔄 RUNNING DETECTION...")
        detected_type, confidence, debug_info = analyze_roof_geometry(roof_only, mask, sun_mask)
        
        # Cache the detection results
        st.session_state.data["detection_confidence"] = confidence
        st.session_state.data["detection_debug"] = debug_info
        st.session_state.data["detected_roof_type"] = detected_type
        print(f"   ✅ Detection complete: {detected_type}, Coverage: {debug_info.get('coverage_ratio', 0):.1%}")
    else:
        # Use cached detection results when other settings change (tilt, azimuth, etc.)
        print(f"   ♻️ USING CACHED DETECTION")
        detected_type = st.session_state.data.get("detected_roof_type", "Pitched")
        confidence = st.session_state.data.get("detection_confidence", 0)
        debug_info = st.session_state.data.get("detection_debug", {})
        print(f"   Cached values: {detected_type}, Coverage: {debug_info.get('coverage_ratio', 0):.1%}")

    # Initialize on first run only
    if "auto_roof_type" not in st.session_state.data:
        auto_azimuth = calculate_azimuth(st.session_state.data["final_poly"], img=roof_only)

        # Set default tilt based on roof type
        default_tilt = DEFAULT_PITCHED_TILT if detected_type == "Pitched" else DEFAULT_FLAT_TILT

        st.session_state.data.update({
            "auto_roof_type": detected_type,
            "user_azimuth": float(auto_azimuth),
            "user_tilt": default_tilt,
            "panel_orientation": DEFAULT_PANEL_ORIENTATION,  # Initialize orientation
            "roof_type_manually_set": False  # Track if user manually changed roof type
        })

        # Display detection reasoning in console/logs
        print(f"\n🏠 ROOF TYPE DETECTION:")
        print(f"   Result: {detected_type} (Confidence: {confidence:.1%})")
        print(f"   Reason: {debug_info.get('reason', 'N/A')}")
        print(f"   Coverage Ratio: {debug_info.get('coverage_ratio', 0):.1%}")
        print(f"   Brightness Range: {debug_info.get('brightness_range', 0):.0f}")
        print(f"   Texture StdDev: {debug_info.get('std_dev', 0):.1f}")
        print(f"   Default Tilt: {default_tilt}°")

    # Auto-update roof type if user hasn't manually changed it
    elif not st.session_state.data.get("roof_type_manually_set", False):
        if st.session_state.data["auto_roof_type"] != detected_type:
            st.session_state.data["auto_roof_type"] = detected_type
            # Update tilt to match new roof type
            st.session_state.data["user_tilt"] = DEFAULT_PITCHED_TILT if detected_type == "Pitched" else DEFAULT_FLAT_TILT
    
    # Get current panel orientation
    current_orientation = st.session_state.data.get("panel_orientation", DEFAULT_PANEL_ORIENTATION)

    current_azimuth = st.session_state.data["user_azimuth"]
    user_tilt = st.session_state.data["user_tilt"]

    # Calculate real-time irradiance based on CURRENT tilt and azimuth
    # This ensures the displayed values always match the current slider positions
    current_irradiance = calculate_solar_potential(lat, lon, user_tilt, current_azimuth)
    st.session_state.data["current_irradiance"] = current_irradiance

    # Calculate actual panel dimensions based on orientation
    if current_orientation == "Landscape":
        display_w = 1.13  # Width when in landscape
        display_h = 1.76 * math.cos(math.radians(user_tilt))  # Projected height
    else:  # Portrait
        display_w = 1.76  # Width when in portrait
        display_h = 1.13 * math.cos(math.radians(user_tilt))  # Projected height

    # Debug logging
    print(f"\n🔧 PANEL GENERATION DEBUG:")
    print(f"   GSD: {gsd:.4f} m/pixel")
    print(f"   Orientation: {current_orientation}")
    print(f"   Panel dimensions: {display_w}m × {display_h:.2f}m (projected)")
    print(f"   Panel size in pixels: {display_w/gsd:.1f} × {display_h/gsd:.1f}")
    print(f"   Edge margin: {ROOF_EDGE_MARGIN}m ({ROOF_EDGE_MARGIN/gsd:.1f} px)")
    print(f"   Azimuth: {current_azimuth}°")
    print(f"   Tilt: {user_tilt}°")

    # Get target count
    limit = st.session_state.data["target_panel_count"]

    # 4a. First, generate maximum capacity grid (for slider range)
    # Handle case where no usable area exists (e.g., shadow tolerance = 0)
    try:
        grid_result = generate_panel_grid(
            sun_mask, gsd, current_azimuth, user_tilt,
            panel_w=1.76,
            panel_h=1.13,
            edge_margin=ROOF_EDGE_MARGIN,
            panel_spacing=PANEL_SPACING,
            orientation=current_orientation
        )
        if grid_result and len(grid_result) == 2:
            all_panels_flat, all_rows_structure = grid_result
        else:
            all_panels_flat, all_rows_structure = [], []
    except (ValueError, TypeError):
        all_panels_flat, all_rows_structure = [], []

    # Cap target panel count to roof capacity if recommended exceeds available space
    max_roof_capacity = len(all_panels_flat)
    if max_roof_capacity > 0 and st.session_state.data["target_panel_count"] > max_roof_capacity:
        st.session_state.data["target_panel_count"] = max_roof_capacity
        limit = max_roof_capacity

    # 4b. Generate Optimized Panel Grid for actual placement
    # For small counts (≤10), tries multiple positions for best contiguity
    # For large counts (>10), uses standard maximum capacity grid
    try:
        optimal_result = generate_optimal_grid(
            sun_mask, gsd, current_azimuth, user_tilt,
            target_count=limit,
            panel_w=1.76,
            panel_h=1.13,
            edge_margin=ROOF_EDGE_MARGIN,
            panel_spacing=PANEL_SPACING,
            orientation=current_orientation
        )
        if optimal_result and len(optimal_result) == 4:
            panels, selected_rows, contiguity_score, grid_warning = optimal_result
        else:
            panels, selected_rows, contiguity_score, grid_warning = [], [], 0, None
    except (ValueError, TypeError):
        panels, selected_rows, contiguity_score, grid_warning = [], [], 0, None

    selected_count = len(panels)

    print(f"   Selected for installation: {selected_count} panels")

    # Pre-compute metrics used by both columns
    system_kwp = (selected_count * 440) / 1000
    base_specific_yield = 950  # kWh/kWp/year, German average (PVGIS baseline)
    orientation_factor = min(current_irradiance / 1000, 1.0)  # tilt/azimuth quality vs ideal
    annual_production = system_kwp * base_specific_yield * orientation_factor
    coverage_pct = (annual_production / annual_kwh * 100) if annual_kwh > 0 else 0

    string_info = f"{selected_count}"
    if selected_count < MIN_PANELS_PER_STRING:
        string_info += " ⚠️"
    elif selected_count > MAX_PANELS_PER_STRING:
        string_info += " ⚠️"

    # 5. UI Layout
    col_main, col_R = st.columns([4, 1.5])

    # Track active view for conditional rendering of the right panel
    active_view = st.session_state.data.get("solar_active_view", "roof_view")

    with col_main:
        # Show view selector only after shadow tolerance is confirmed
        if shadow_confirmed:
            radio_key = f"solar_view_radio_{st.session_state.data.get('view_reset_counter', 0)}"
            selected_view = st.radio(
                "View",
                ["📍 Roof View", "🌤️ Shadow Tolerance"],
                index=0 if active_view == "roof_view" else 1,
                horizontal=True,
                label_visibility="collapsed",
                key=radio_key
            )

            # Update active view in session state
            new_view = "roof_view" if selected_view == "📍 Roof View" else "shadow_tolerance"
            if new_view != active_view:
                st.session_state.data["solar_active_view"] = new_view
                active_view = new_view

            if active_view == "roof_view":
                # Roof View with panels
                display_img = roof_only.copy()
                display_img = draw_azimuth_arrow(display_img, current_azimuth)

                # Create solar panel sprite
                if current_orientation == "Landscape":
                    panel_w_px = 1.13 / gsd
                    panel_h_px = (1.76 * math.cos(math.radians(user_tilt))) / gsd
                else:  # Portrait
                    panel_w_px = 1.76 / gsd
                    panel_h_px = (1.13 * math.cos(math.radians(user_tilt))) / gsd

                panel_sprite = create_solar_panel_sprite(panel_w_px, panel_h_px, current_azimuth)

                # Overlay each panel as a sprite
                for p in panels:
                    display_img = overlay_panel_sprite(display_img, p, panel_sprite)

                # System metrics as native Streamlit widgets (sharp at any resolution)
                # Weight columns by value width so spacing is even
                _vals = [
                    str(selected_count),
                    f"{system_kwp:.2f} kWp",
                    f"{current_irradiance:.0f} W/m\u00b2",
                    f"{annual_production:,.0f} kWh/yr",
                    f"{coverage_pct:.0f}%",
                ]
                _labels = ["Panels", "System", "Irradiance", "Production", "Coverage"]
                _weights = [max(len(l), len(v)) for l, v in zip(_labels, _vals)]
                with st.container(border=True):
                    m1, m2, m3, m4, m5 = st.columns(_weights)
                    m1.metric(_labels[0], _vals[0])
                    m2.metric(_labels[1], _vals[1])
                    m3.metric(_labels[2], _vals[2])
                    m4.metric(_labels[3], _vals[3])
                    m5.metric(_labels[4], _vals[4])

                st.image(display_img, width="stretch", caption="Panel placement on roof")

                # Show placement optimization status
                if grid_warning:
                    st.warning(grid_warning)
                elif contiguity_score > 0 and selected_count <= 10:
                    st.success(f"✓ Optimized panel placement (contiguity score: {contiguity_score})")

                if st.button("Run Simulation And Generate Report ☀️", type="primary", use_container_width=True):
                    # Validate minimum panel count before generating report
                    if selected_count < MIN_PANELS_PER_STRING:
                        st.error(f"Cannot generate report: Minimum {MIN_PANELS_PER_STRING} panels required for series connection.")
                    else:
                        # Generate and store images for PDF export
                        pdf_panel_img = roof_only.copy()
                        pdf_panel_img = draw_azimuth_arrow(pdf_panel_img, current_azimuth)
                        if current_orientation == "Landscape":
                            pdf_panel_w_px = 1.13 / gsd
                            pdf_panel_h_px = (1.76 * math.cos(math.radians(user_tilt))) / gsd
                        else:
                            pdf_panel_w_px = 1.76 / gsd
                            pdf_panel_h_px = (1.13 * math.cos(math.radians(user_tilt))) / gsd
                        pdf_panel_sprite = create_solar_panel_sprite(pdf_panel_w_px, pdf_panel_h_px, current_azimuth)
                        for p in panels:
                            pdf_panel_img = overlay_panel_sprite(pdf_panel_img, p, pdf_panel_sprite)
                        st.session_state.data["pdf_panel_image"] = pdf_panel_img

                        # Store the exact same values displayed in the live preview
                        st.session_state.data["solar_results"] = {
                            "total_roof_area_m2": total_area_m2,
                            "usable_roof_area_m2": usable_area_m2,
                            "panel_count": selected_count,
                            "system_kwp": system_kwp,
                            "azimuth": current_azimuth,
                            "tilt_angle": user_tilt,
                            "panel_orientation": current_orientation,
                            "roof_form": st.session_state.data["auto_roof_type"],
                            "irradiance_potential": current_irradiance,
                            "annual_production_kwh": annual_production,
                            "coverage_percentage": coverage_pct,
                        }

                        # Clear stale report data so it regenerates with new settings
                        for key in ["final_analysis", "detailed_solar_data"]:
                            st.session_state.data.pop(key, None)
                        st.session_state.pop("rag_bot", None)

                        # Auto-save inquiry (includes images stored earlier)
                        if st.session_state.get("inquiry_id"):
                            save_inquiry(
                                st.session_state.inquiry_id,
                                st.session_state.data,
                                step=5,
                                sub_step="verify"
                            )

                        st.session_state.step = 5
                        st.rerun()

            else:
                # Shadow Tolerance adjustment with overlay
                shadow_img = roof_only.copy()
                if np.any(sun_mask > 0):
                    pink_tint = np.zeros_like(shadow_img)
                    pink_tint[sun_mask > 0] = (255, 20, 147)  # Hot pink in BGR
                    shadow_img = cv2.addWeighted(shadow_img, 1.0, pink_tint, 0.3, 0)

                    contours, _ = cv2.findContours(sun_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(shadow_img, contours, -1, (0, 255, 0), 1)

                st.image(shadow_img, width="stretch", caption="Usable area overlay")

                st.slider(
                    "Shadow Tolerance",
                    -100, 50, int(current_threshold),
                    key="sun_slider_widget",
                    on_change=update_threshold,
                    help="Negative values = very strict (only brightest areas). Positive values = more tolerant (include darker areas)."
                )
                st.caption("Use negative values to select only the brightest roof sections. The pink overlay shows usable area.")

                if st.button("Confirm Shadow Tolerance", type="primary", use_container_width=True):
                    st.session_state.data["solar_active_view"] = "roof_view"
                    st.session_state.data["view_reset_counter"] = st.session_state.data.get("view_reset_counter", 0) + 1
                    st.session_state.data.pop("target_panel_count", None)
                    st.rerun()

        else:
            # Before confirmation: show shadow tolerance setup view
            display_img = roof_only.copy()

            # Show pink overlay and green outline during setup
            if np.any(sun_mask > 0):
                pink_tint = np.zeros_like(display_img)
                pink_tint[sun_mask > 0] = (255, 20, 147)  # Hot pink in BGR
                display_img = cv2.addWeighted(display_img, 1.0, pink_tint, 0.3, 0)

                contours, _ = cv2.findContours(sun_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(display_img, contours, -1, (0, 255, 0), 1)

            st.image(display_img, width="stretch", caption="Adjust shadow tolerance to define usable area")

        if not shadow_confirmed:
            # Shadow Tolerance setup - shown prominently when not yet confirmed
            st.markdown("### 🌤️ Shadow Tolerance")
            st.caption("Adjust this slider to define which areas of your roof are usable for solar panels. "
                       "Use negative values to select only the brightest sections.")

            st.slider(
                "Shadow Tolerance",
                -100, 50, int(current_threshold),
                key="sun_slider_widget",
                on_change=update_threshold,
                help="Negative values = very strict (only brightest areas). Positive values = more tolerant (include darker areas)."
            )

            st.info("Once you're happy with the usable area (pink overlay), click the button below to continue.")
            if st.button("Confirm Shadow Tolerance", type="primary", use_container_width=True):
                st.session_state.data["shadow_tolerance_confirmed"] = True
                st.rerun()

    with col_R:
        if not shadow_confirmed:
            st.markdown("### 📊 Roof Metrics")
            st.metric("Total Roof Area", f"{total_area_m2:.1f} m²")
            st.metric("Usable Space", f"{usable_area_m2:.1f} m²")
        elif active_view == "shadow_tolerance":
            st.markdown("### 📊 Roof Metrics")
            st.metric("Total Roof Area", f"{total_area_m2:.1f} m²")
            st.metric("Usable Space", f"{usable_area_m2:.1f} m²")
        else:
            st.markdown("### Panel Configuration")
            max_capacity = len(all_panels_flat)
            min_installable = max(MIN_PANELS_PER_STRING, 1)

            # Panel count warnings
            if max_capacity < MIN_PANELS_PER_STRING:
                st.error(f"⚠️ Roof capacity ({max_capacity} panels) is below the minimum requirement "
                        f"({MIN_PANELS_PER_STRING} panels in series). Installation not viable with current settings.")
            elif max_capacity < TYPICAL_MIN_STRING:
                st.warning(f"⚠️ Roof capacity ({max_capacity} panels) is below typical minimum "
                          f"({TYPICAL_MIN_STRING} panels). Consider adjusting settings or using micro-inverters.")

            # Panel Count Slider
            slider_max = min(max_capacity, MAX_PANELS_PER_STRING)
            default_target = min(
                st.session_state.data.get("target_panel_count", recommended_limit),
                slider_max
            )
            default_target = max(default_target, min_installable)

            if slider_max <= min_installable:
                current_count = max(slider_max, 1)
                st.metric("Number of Panels", f"{current_count}")
                st.caption(f"Limited capacity: only {current_count} panel(s) fit on usable area.")
                st.session_state.data["target_panel_count"] = current_count
            else:
                current_count = st.slider(
                    "Number of Panels to Install",
                    min_value=min_installable,
                    max_value=slider_max,
                    value=int(default_target),
                    key="panel_slider_widget",
                    help=(f"String limits: Min {MIN_PANELS_PER_STRING}, Max {MAX_PANELS_PER_STRING}. "
                          f"Typical: {TYPICAL_MIN_STRING}-{TYPICAL_MAX_STRING} panels per string. "
                          f"Questionnaire recommended: {recommended_limit}. "
                          f"Only contiguous panel groups shown."),
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

            # Azimuth slider
            st.slider("Solar Orientation (Azimuth °)", 0, 359, int(current_azimuth),
                      key="az_slider_widget", on_change=update_azimuth,
                      help="Watch the Irradiance Potential change as you rotate!")

            # Panel Orientation
            selected_orientation = st.selectbox(
                "Panel Orientation",
                ["Portrait", "Landscape"],
                index=0 if current_orientation == "Portrait" else 1,
                help="Portrait: Vertical strings (1.76m wide). Landscape: Horizontal strings (1.13m wide)."
            )

            # Update orientation if changed
            if selected_orientation != st.session_state.data.get("panel_orientation"):
                st.session_state.data["panel_orientation"] = selected_orientation
                st.rerun()

            # Detection Details
            if SHOW_DEBUG_INFO and "detection_debug" in st.session_state.data:
                debug = st.session_state.data["detection_debug"]
                confidence = st.session_state.data.get("detection_confidence", 0)
                detected_type = st.session_state.data.get("detected_roof_type", "Unknown")
                manually_set = st.session_state.data.get("roof_type_manually_set", False)

                with st.expander("🔍 Detection Details", expanded=False):
                    col_det1, col_det2 = st.columns(2)
                    with col_det1:
                        st.metric("Detected Type", detected_type)
                    with col_det2:
                        st.metric("Confidence", f"{confidence:.0%}")

                    current_roof_type = st.session_state.data.get("auto_roof_type", "Pitched")
                    if manually_set and current_roof_type != detected_type:
                        st.info(f"You selected **{current_roof_type}** (detection suggests {detected_type})")
                        if st.button("Reset to Auto-Detection", key="reset_roof_type"):
                            st.session_state.data["auto_roof_type"] = detected_type
                            st.session_state.data["roof_type_manually_set"] = False
                            st.session_state.data["user_tilt"] = DEFAULT_PITCHED_TILT if detected_type == "Pitched" else DEFAULT_FLAT_TILT
                            recalculate_irradiance()
                            st.rerun()

                    st.write(f"**Reasoning:** {debug.get('reason', 'N/A')}")
                    st.write(f"**Coverage Ratio:** {debug.get('coverage_ratio', 0):.1%} "
                            f"(Flat if ≥ {MIN_FLAT_ROOF_COVERAGE:.0%})")
                    st.write(f"**Brightness Range:** {debug.get('brightness_range', 0):.0f}/255 "
                            f"(Pitched if ≥ 30)")
                    st.write(f"**Texture Variance:** {debug.get('std_dev', 0):.1f} "
                            f"(Pitched if > 15)")

            # Roof Form
            selected_type = st.selectbox("Roof Form", ["Pitched", "Flat"],
                                        index=0 if st.session_state.data["auto_roof_type"] == "Pitched" else 1)

            # Re-run if roof type changes to update tilt
            if selected_type != st.session_state.data["auto_roof_type"]:
                st.session_state.data["auto_roof_type"] = selected_type
                st.session_state.data["roof_type_manually_set"] = True
                st.session_state.data["user_tilt"] = DEFAULT_PITCHED_TILT if selected_type == "Pitched" else DEFAULT_FLAT_TILT
                recalculate_irradiance()
                st.rerun()

            # Tilt Angle Slider - visible for PITCHED roofs
            if selected_type == "Pitched":
                st.slider(
                    "Tilt Angle (°)",
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
                st.info(f"ℹ️ Flat roof: {DEFAULT_FLAT_TILT}° mounting angle for optimal drainage and performance.")

            # Roof Metrics
            st.markdown("---")
            st.markdown("### 📊 Roof Metrics")
            st.metric("Total Roof Area", f"{total_area_m2:.1f} m²")
            st.metric("Usable Space", f"{usable_area_m2:.1f} m²")