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

# Panel optimization - Simple row-by-row selection
from src.panel_optimization import (
    select_panels_from_grid
)

# Electrical configuration - Serpentine wiring
from src.electrical_config import (
    create_serpentine_wiring,
    calculate_electrical_specs,
    create_wiring_schematic,
    PANEL_VOLTAGE,
    PANEL_CURRENT,
    PANEL_POWER
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
ROOF_EDGE_MARGIN = 0.30         # Minimum distance from roof edge (meters) - safety margin
PANEL_SPACING = 0.05            # Gap between adjacent panels (meters) - maintenance access

# Panel Orientation Options
# Portrait: 1.76m wide × 1.13m tall (vertical strings, taller than wide)
# Landscape: 1.13m wide × 1.76m tall (horizontal strings, wider than tall)
DEFAULT_PANEL_ORIENTATION = "Portrait"  # "Portrait" or "Landscape"

# Wiring Visualization Parameters
WIRING_LINE_THICKNESS = 1       # Thickness of connection lines (pixels) - try 1-3
WIRING_ARROW_SIZE = 6           # Size of direction arrows (pixels) - try 4-10
WIRING_START_MARKER_SIZE = 6    # Size of start marker circles (pixels) - try 4-10
WIRING_SHOW_ARROWS = True       # Show direction arrows on connections
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
            "panel_orientation": DEFAULT_PANEL_ORIENTATION,  # Initialize orientation
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
    
    # Get current panel orientation
    current_orientation = st.session_state.data.get("panel_orientation", DEFAULT_PANEL_ORIENTATION)

    current_azimuth = st.session_state.data["user_azimuth"]
    user_tilt = st.session_state.data["user_tilt"]

    # Calculate real-time irradiance if not already calculated
    if "current_irradiance" not in st.session_state.data:
        recalculate_irradiance()

    current_irradiance = st.session_state.data.get("current_irradiance", 0)

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
    all_panels_flat, all_rows_structure = generate_panel_grid(
        sun_mask, gsd, current_azimuth, user_tilt,
        panel_w=1.76,
        panel_h=1.13,
        edge_margin=ROOF_EDGE_MARGIN,
        panel_spacing=PANEL_SPACING,
        orientation=current_orientation
    )

    # 4b. Generate Optimized Panel Grid for actual placement
    # For small counts (≤10), tries multiple positions for best contiguity
    # For large counts (>10), uses standard maximum capacity grid
    panels, selected_rows, contiguity_score, grid_warning = generate_optimal_grid(
        sun_mask, gsd, current_azimuth, user_tilt,
        target_count=limit,
        panel_w=1.76,
        panel_h=1.13,
        edge_margin=ROOF_EDGE_MARGIN,
        panel_spacing=PANEL_SPACING,
        orientation=current_orientation
    )

    selected_count = len(panels)

    # Create serpentine wiring path
    wiring_path = create_serpentine_wiring(panels, selected_rows)

    # Calculate electrical specifications
    electrical_specs = calculate_electrical_specs(selected_count)

    print(f"   Selected for installation: {selected_count} panels")
    print(f"   System voltage: {electrical_specs['voltage']:.1f}V")
    print(f"   System current: {electrical_specs['current']:.1f}A")
    print(f"   System power: {electrical_specs['power']:.0f}W")

    # 5. UI Layout
    col_main, col_R = st.columns([4, 1.5])
    
    with col_main:
        # Create two visualizations: roof context and wiring schematic
        viz_tabs = st.tabs(["📍 Roof View", "🔌 Wiring Schematic"])
        
        with viz_tabs[0]:
            # Roof view with panels
            display_img = roof_only.copy()
            
            # Yellow Mask Overlay
            mask_overlay = np.zeros_like(display_img)
            mask_overlay[sun_mask > 0] = (0, 255, 255)
            display_img = cv2.addWeighted(display_img, 1.0, mask_overlay, 0.3, 0)
            
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

            # DON'T draw wiring on roof view - it obscures panels
            # Wiring is shown clearly in the separate schematic tab

            st.image(display_img, use_container_width=True, caption="Panel placement on roof")

            # Show placement optimization status
            if grid_warning:
                st.warning(grid_warning)
            elif contiguity_score > 0 and selected_count <= 10:
                st.success(f"✓ Optimized panel placement (contiguity score: {contiguity_score})")

        with viz_tabs[1]:
            # Electrical schematic showing serpentine wiring
            if selected_count > 0 and wiring_path:
                schematic = create_wiring_schematic(
                    panels,
                    selected_rows,
                    wiring_path
                )
                if schematic is not None:
                    st.image(schematic, use_container_width=True, caption="Serpentine Wiring Schematic")

                    # Add detailed electrical information
                    st.markdown("### ⚡ Electrical Specifications")

                    # System overview
                    with st.expander("📊 System Overview", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("System Voltage", f"{electrical_specs['voltage']:.0f}V")
                        with col2:
                            st.metric("System Current", f"{electrical_specs['current']:.0f}A")
                        with col3:
                            st.metric("System Power", f"{electrical_specs['power']:.0f}W")

                        st.write(f"**Configuration:** Series (All panels in one string)")
                        st.write(f"**Wiring Pattern:** Serpentine (S-shape)")

                    # Wiring details
                    with st.expander("🔌 Wiring Instructions"):
                        st.write(f"**Total Panels:** {selected_count}")
                        st.write(f"**Wiring Order:** Panel 1 → Panel {selected_count}")
                        st.write(f"   {' → '.join([f'P{i+1}' for i in range(selected_count)])}")
                        st.write("")
                        st.write("**Physical Rule:** Connect + terminal of Panel N to - terminal of Panel N+1")
                        st.write("**Pattern:** Serpentine (S-pattern) minimizes wire length")
                        st.write(f"**Voltage:** {electrical_specs['voltage']:.0f}V (31V × {selected_count})")
                        st.write(f"**Current:** {electrical_specs['current']:.0f}A (constant in series)")

                    # Safety notes
                    with st.expander("⚠️ Safety & Installation Notes"):
                        st.write("**Electrical Safety:**")
                        st.write("- Maximum DC voltage (NEC): 1000V")
                        st.write("- Minimum inverter start voltage: 200V")
                        st.write(f"- Your system: {electrical_specs['voltage']:.0f}V ✓")

                        st.write("\n**Installation Guidelines:**")
                        st.write("- Series panels must be physically adjacent")
                        st.write("- Use proper DC-rated connectors (MC4)")
                        st.write("- Follow serpentine pattern to minimize wire runs")
                        st.write("- Keep wire gauge appropriate for current rating")
                else:
                    st.info("Configure panels to see electrical schematic")
            else:
                st.info("Configure panels to see electrical schematic")
        
        with st.expander("🛠️ Analysis And Adjustments", expanded=True):
            # Panel Count Slider
            max_capacity = len(all_panels_flat)

            # Enforce minimum panel count
            min_installable = max(MIN_PANELS_PER_STRING, 1)

            # Show warning if roof capacity is below minimum requirement
            if max_capacity < MIN_PANELS_PER_STRING:
                st.error(f"⚠️ Roof capacity ({max_capacity} panels) is below the minimum requirement "
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

            c1, c2 = st.columns(2)
            selected_type = c1.selectbox("Roof Form", ["Pitched", "Flat"], 
                                       index=0 if st.session_state.data["auto_roof_type"] == "Pitched" else 1)
            
            # Panel Orientation Toggle
            selected_orientation = c2.selectbox(
                "Panel Orientation",
                ["Portrait", "Landscape"],
                index=0 if current_orientation == "Portrait" else 1,
                help="Portrait: Vertical strings (1.76m wide). Landscape: Horizontal strings (1.13m wide)."
            )
            
            # Update orientation if changed
            if selected_orientation != st.session_state.data.get("panel_orientation"):
                st.session_state.data["panel_orientation"] = selected_orientation
                st.rerun()
            
            # Show detection confidence and reasoning if enabled
            if SHOW_DEBUG_INFO and "detection_debug" in st.session_state.data:
                debug = st.session_state.data["detection_debug"]
                confidence = st.session_state.data.get("detection_confidence", 0)
                
                # Show detection info in expander instead of column
                with st.expander("🔍 Detection Details", expanded=False):
                    st.metric("Detection Confidence", f"{confidence:.0%}", 
                             help=debug.get("reason", "Auto-detected roof type"))
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
        
        # Electrical configuration display
        if selected_count > 0:
            st.caption(f"⚡ Configuration: Series (Single String)")
            st.caption(f"🔌 Wiring: Serpentine pattern")
            st.caption(f"⚙️ System: {electrical_specs['voltage']:.0f}V @ {electrical_specs['current']:.0f}A")
        
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
                # Store electrical configuration
                string_config_data = {
                    'num_strings': 1,
                    'config_type': f"{selected_count} panels in series (Serpentine)",
                    'wiring_type': 'series',
                    'total_voltage': electrical_specs['voltage'],
                    'total_current': electrical_specs['current'],
                    'panels_per_string': [selected_count],
                    'wiring_path': wiring_path
                }

                st.session_state.data["solar_results"] = {
                    "total_roof_area_m2": total_area_m2,
                    "usable_roof_area_m2": usable_area_m2,
                    "panel_count": selected_count,
                    "system_kwp": system_kwp,
                    "azimuth": current_azimuth,
                    "tilt_angle": user_tilt,
                    "panel_orientation": current_orientation,  # NEW: Save orientation
                    "roof_form": st.session_state.data["auto_roof_type"],
                    "irradiance_potential": current_irradiance,
                    "annual_production_kwh": annual_production,
                    "coverage_percentage": coverage_pct,
                    "string_configuration": string_config_data
                }
                st.session_state.step = 5
                st.rerun()