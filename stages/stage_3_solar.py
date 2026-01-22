import streamlit as st
import numpy as np
import cv2
import math
import ui_components as ui
from src.solar_engine import (
    get_masked_roof_array, 
    analyze_roof_texture, 
    calculate_azimuth,
    draw_azimuth_arrow,
    get_sunny_polygon_mask,
    calculate_solar_potential,
    calculate_global_gsd # Ensure this is in your src/solar_engine.py
)

def update_azimuth():
    """Callback to sync the azimuth slider state."""
    st.session_state.data["user_azimuth"] = float(st.session_state.az_slider_widget)

def update_threshold():
    """Callback to sync the shadow threshold slider state."""
    st.session_state.data["sun_threshold"] = int(st.session_state.sun_slider_widget)

def show():
    st.header("Step 3: Solar & Irradiance Analysis")
    
    # 1. DATA VALIDATION
    if "final_poly" not in st.session_state.data:
        st.warning("Please complete the roof refinement in Step 2 first.")
        if st.button("⬅️ Back to Step 2"):
            st.session_state.step = 2
            st.rerun()
        return

    # Extract stored data
    res = st.session_state.data["res"]
    zoom_img = res["zoom_img"]
    poly_pts = st.session_state.data["final_poly"]
    
    # Get location for Global GSD calculation
    # Fallback to Hamburg if search data is missing
    lat = st.session_state.data.get("location", {}).get("lat", 53.5511)
    zoom_level = 19 

    # 2. GENERATE MASKED ARRAY
    mask, roof_only = get_masked_roof_array(zoom_img, poly_pts)

    # 3. SMART INITIAL AUTOMATED DETECTION
    if "auto_roof_type" not in st.session_state.data:
        roof_type, variance = analyze_roof_texture(roof_only, mask)
        auto_azimuth = calculate_azimuth(roof_only, poly_pts)
        
        # Adaptive Shadow Thresholding
        gray_roof = cv2.cvtColor(roof_only, cv2.COLOR_BGR2GRAY)
        roof_pixels = gray_roof[mask > 0]
        mean_brightness = np.mean(roof_pixels) if len(roof_pixels) > 0 else 128
        initial_threshold = min(255, int(mean_brightness + ui.SHADOW_BIAS))
        
        # Geometry defaults
        if roof_type == "Flat":
            auto_azimuth = 180.0
            init_tilt = 0.0
        else:
            init_tilt = 38.0

        st.session_state.data.update({
            "auto_roof_type": roof_type,
            "roof_variance": variance,
            "user_azimuth": float(auto_azimuth),
            "user_tilt": init_tilt,
            "sun_threshold": initial_threshold 
        })

    # 4. UI LAYOUT
    col_L, col_center, col_R = st.columns([1, 4, 1.5])

    with col_center:
        st.markdown("### 🛠️ Geometry & Shadow Mapping")
        
        # SYNCED SHADOW SLIDER
        st.slider(
            "Shadow Sensitivity", 
            min_value=0, 
            max_value=255, 
            value=int(st.session_state.data["sun_threshold"]),
            key="sun_slider_widget",
            on_change=update_threshold
        )
        
        current_threshold = st.session_state.data["sun_threshold"]
        sun_mask = get_sunny_polygon_mask(roof_only, current_threshold)
        
        # Preview with Yellow Shadow Mask and Orientation Arrow
        sunny_preview = roof_only.copy()
        sunny_preview[sun_mask > 0] = [0, 255, 255] 
        preview_overlay = cv2.addWeighted(roof_only, 0.7, sunny_preview, 0.3, 0)
        final_preview = draw_azimuth_arrow(preview_overlay, st.session_state.data["user_azimuth"])
        
        st.image(final_preview, caption="Yellow = Sunny Area | Red Arrow = Slope Direction", width=ui.DISPLAY_WIDTH)
        
        # USER CONTROLS
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            
            # Form Toggle
            type_options = ["Flat", "Pitched"]
            curr_type_idx = 1 if st.session_state.data["auto_roof_type"] == "Pitched" else 0
            selected_type = c1.selectbox("Roof Form", type_options, index=curr_type_idx)
            
            if selected_type != st.session_state.data["auto_roof_type"]:
                st.session_state.data["auto_roof_type"] = selected_type
                if selected_type == "Flat":
                    st.session_state.data["user_azimuth"] = 180.0
                    st.session_state.data["user_tilt"] = 0.0
                else:
                    st.session_state.data["user_tilt"] = 38.0
                st.rerun()
            
            # Tilt
            if selected_type == "Pitched":
                tilt = c2.number_input("Tilt Angle (°)", 0.0, 90.0, st.session_state.data["user_tilt"])
                st.session_state.data["user_tilt"] = tilt
            else:
                tilt = 0.0
                c2.number_input("Tilt Angle (°)", value=0.0, disabled=True)
            
            # Azimuth
            c3.slider("Orientation (°)", 0, 359, int(st.session_state.data["user_azimuth"]),
                      key="az_slider_widget", on_change=update_azimuth)

        st.divider()
        
        # NAVIGATION
        btn_col_1, btn_col_2 = st.columns(2)
        if btn_col_1.button("⬅️ Back to Step 2", use_container_width=True):
            st.session_state.step = 2
            st.rerun()
            
        if btn_col_2.button("Run Simulation ☀️", type="primary", use_container_width=True):
            # Calculate with corrected pvlib logic
            irrad_val = calculate_solar_potential(lat, st.session_state.data.get("location", {}).get("lon", 9.99), 
                                                tilt, st.session_state.data["user_azimuth"])
            
            # Finalize metrics for Stage 4
            current_gsd = calculate_global_gsd(lat, zoom_level)
            sunny_pixel_count = np.sum(sun_mask > 0)
            footprint_area = sunny_pixel_count * (current_gsd ** 2)
            
            # Slope Correction: Surface Area = Footprint / cos(tilt)
            true_surface_area = footprint_area / math.cos(math.radians(tilt)) if tilt > 0 else footprint_area
            panel_count = int(true_surface_area / 1.75) 
            
            st.session_state.data["solar_results"] = {
                "irradiance_w_m2": irrad_val,
                "true_area_m2": true_surface_area,
                "panel_count": panel_count,
                "total_kwp": (panel_count * 400) / 1000 
            }
            st.session_state.step = 4
            st.rerun()

    with col_R:
        st.markdown("### 📊 Global Metrics")
        
        # Precise GSD and Area Scaling
        current_gsd = calculate_global_gsd(lat, zoom_level)
        roof_pixel_count = np.sum(mask > 0)
        footprint_m2 = roof_pixel_count * (current_gsd ** 2)
        
        # Tilt Correction
        true_area_m2 = footprint_m2 / math.cos(math.radians(tilt)) if tilt > 0 else footprint_m2
        
        st.metric("Local GSD", f"{current_gsd:.3f} m/px")
        st.metric("Actual Surface", f"{true_area_m2:.1f} m²")
        
        # Sunny Area Metric
        sunny_pixel_count = np.sum(sun_mask > 0)
        sunny_footprint = sunny_pixel_count * (current_gsd ** 2)
        sunny_true_area = sunny_footprint / math.cos(math.radians(tilt)) if tilt > 0 else sunny_footprint
        
        st.metric("Usable Area", f"{sunny_true_area:.1f} m²", 
                  delta=f"{sunny_true_area - true_area_m2:.1f} m²", delta_color="inverse")
        
        st.info(f"Analysis at Lat: {lat:.2f}°. Scale corrected for Mercator distortion.")