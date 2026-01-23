import streamlit as st
import numpy as np
import cv2
import math
from shapely.geometry import Polygon
from shapely import affinity
import ui_components as ui

# Consolidated Imports
from src.solar_engine import (
    get_masked_roof_array, 
    analyze_roof_texture, 
    draw_azimuth_arrow,
    get_sunny_polygon_mask,
    calculate_solar_potential,
    calculate_global_gsd
)
from src.geometry_utils import calculate_azimuth, mask_to_polygon

def update_azimuth():
    st.session_state.data["user_azimuth"] = float(st.session_state.az_slider_widget)

def update_threshold():
    st.session_state.data["sun_threshold"] = int(st.session_state.sun_slider_widget)

def generate_panel_grid(sunny_mask, gsd, azimuth, panel_w=1.75, panel_h=1.05):
    """
    Creates a vector grid of panels rotated to match the roof orientation.
    """
    sunny_pts = mask_to_polygon(sunny_mask)
    if not sunny_pts:
        return []
    
    sunny_poly = Polygon(sunny_pts).buffer(-1) # 1px safety margin
    center = sunny_poly.centroid
    
    # Rotate to North-Up for grid generation
    aligned_poly = affinity.rotate(sunny_poly, -azimuth, origin=center)
    
    pw_px = panel_w / gsd
    ph_px = panel_h / gsd
    
    minx, miny, maxx, maxy = aligned_poly.bounds
    aligned_panels = []
    
    for x in np.arange(minx, maxx, pw_px):
        for y in np.arange(miny, maxy, ph_px):
            p = Polygon([(x, y), (x+pw_px, y), (x+pw_px, y+ph_px), (x, y+ph_px)])
            if aligned_poly.contains(p):
                aligned_panels.append(p)
    
    # Rotate back to original roof orientation
    return [affinity.rotate(p, azimuth, origin=center) for p in aligned_panels]

def show():
    st.header("Step 3b: Solar And Irradiance Analysis")
    
    # 1. DATA VALIDATION
    if "final_poly" not in st.session_state.data:
        st.warning("Please complete the roof refinement in Step 2 first.")
        return

    res = st.session_state.data["res"]
    zoom_img = res["zoom_img"]
    poly_pts = st.session_state.data["final_poly"]
    lat = st.session_state.data.get("confirmed_lat", 53.5511)
    lon = st.session_state.data.get("confirmed_lon", 9.99)
    zoom_level = 19 

    # 2. IMAGE PREPARATION
    mask, roof_only = get_masked_roof_array(zoom_img, poly_pts)

    # 3. INITIAL AUTOMATED DETECTION
    if "auto_roof_type" not in st.session_state.data:
        detected_type, _ = analyze_roof_texture(roof_only, mask)
        auto_azimuth = calculate_azimuth(poly_pts, img=roof_only)
        
        # Smart initial shadow threshold (looking for histogram peak)
        gray_roof = cv2.cvtColor(roof_only, cv2.COLOR_BGR2GRAY)
        roof_pixels = gray_roof[mask > 0]
        peak_val = np.argmax(np.histogram(roof_pixels, bins=256, range=(0, 256))[0]) if len(roof_pixels) > 0 else 128
        
        st.session_state.data.update({
            "auto_roof_type": detected_type,
            "user_azimuth": float(auto_azimuth),
            "user_tilt": 38.0 if detected_type == "Pitched" else 0.0,
            "sun_threshold": 25 # Initial offset from peak
        })

    # 4. MAIN LAYOUT
    col_L, col_center, col_R = st.columns([1, 4, 1.5])

    with col_center:
        st.markdown("### 🛠️ Geometry And Shadow Mapping")
        
        # Shadow Control
        st.slider("Shadow Sensitivity", 0, 100, int(st.session_state.data["sun_threshold"]),
                  key="sun_slider_widget", on_change=update_threshold)
        
        current_threshold = st.session_state.data["sun_threshold"]
        sun_mask = get_sunny_polygon_mask(roof_only, mask, threshold_offset=current_threshold)
        current_gsd = calculate_global_gsd(lat, zoom_level)
        current_azimuth = st.session_state.data["user_azimuth"]
        
        # --- CALCULATE MAXIMUM POSSIBLE GRID ---
        full_grid = generate_panel_grid(sun_mask, current_gsd, current_azimuth)
        max_possible = len(full_grid)

        # --- PANEL SELECTION SLIDER ---
        st.divider()
        st.markdown("### 🔢 System Sizing And Selection")
        
        recommended = st.session_state.data.get("recommended_count", 1)
        # Cap recommendation by what actually fits
        default_val = min(recommended, max_possible) if max_possible > 0 else 0
        
        selected_count = st.slider(
            "Number of Panels", 
            min_value=1 if max_possible > 0 else 0, 
            max_value=max_possible, 
            value=default_val,
            help=f"Questionnaire recommendation: {recommended}. Max physical limit: {max_possible}."
        )

        # Selection of panels for rendering
        display_panels = full_grid[:selected_count]

        # RENDER VIZ
        viz_img = roof_only.copy()
        sunny_overlay = viz_img.copy()
        sunny_overlay[sun_mask > 0] = [0, 255, 255]
        viz_img = cv2.addWeighted(viz_img, 0.7, sunny_overlay, 0.3, 0)
        
        for p in display_panels:
            pts = np.array(p.exterior.coords, np.int32)
            cv2.polylines(viz_img, [pts], True, (255, 0, 0), 1)
            overlay = viz_img.copy()
            cv2.fillPoly(overlay, [pts], (255, 200, 0))
            viz_img = cv2.addWeighted(overlay, 0.4, viz_img, 0.6, 0)

        final_preview = draw_azimuth_arrow(viz_img, current_azimuth)
        st.image(final_preview, caption="Panel Alignment And Shadow Analysis", width=ui.DISPLAY_WIDTH)
        
        # ORIENTATION CONTROLS
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            selected_type = c1.selectbox("Roof Form", ["Flat", "Pitched"], 
                                         index=0 if st.session_state.data["auto_roof_type"] == "Flat" else 1)
            
            if selected_type != st.session_state.data["auto_roof_type"]:
                st.session_state.data["auto_roof_type"] = selected_type
                st.session_state.data["user_tilt"] = 0.0 if selected_type == "Flat" else 38.0
                st.rerun()

            is_flat = st.session_state.data["auto_roof_type"] == "Flat"
            tilt = c2.number_input("Tilt Angle (°)", 0.0, 90.0, float(st.session_state.data["user_tilt"]), disabled=is_flat)
            st.session_state.data["user_tilt"] = tilt
            
            c3.slider("Orientation (°)", 0, 359, int(st.session_state.data["user_azimuth"]),
                      key="az_slider_widget", on_change=update_azimuth)

        st.divider()
        
        if st.button("Run Simulation And Generate Report ☀️", type="primary", use_container_width=True):
            irrad_val = calculate_solar_potential(lat, lon, tilt, current_azimuth)
            st.session_state.data["solar_results"] = {
                "irradiance_w_m2": irrad_val,
                "panel_count": selected_count,
                "total_kwp": (selected_count * 400) / 1000 
            }
            st.session_state.step = 5
            st.rerun()

    with col_R:
        st.markdown("### 📊 Global Metrics")
        st.metric("Selected Panels", f"{selected_count}")
        st.metric("System Size", f"{(selected_count * 400) / 1000:.2f} kWp")
        if recommended > max_possible:
            st.warning(f"Note: Roof only fits {max_possible} panels. Your requirement was {recommended}.")
        
        st.info(f"System optimized for {current_azimuth:.1f}° azimuth And {tilt}° tilt.")