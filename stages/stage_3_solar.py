import streamlit as st
import numpy as np
import cv2
import math
from shapely.geometry import Polygon
from shapely import affinity
import ui_components as ui
from src.solar_engine import (
    get_masked_roof_array, 
    analyze_roof_texture, 
    draw_azimuth_arrow,
    get_sunny_polygon_mask,
    calculate_solar_potential,
    calculate_global_gsd
)
from src.geometry_utils import mask_to_polygon, calculate_azimuth

def update_azimuth():
    """Callback to sync the azimuth slider state."""
    st.session_state.data["user_azimuth"] = float(st.session_state.az_slider_widget)

def update_threshold():
    """Callback to sync the shadow threshold slider state."""
    st.session_state.data["sun_threshold"] = int(st.session_state.sun_slider_widget)

def generate_panel_grid(sunny_mask, gsd, azimuth, panel_w=1.75, panel_h=1.05):
    """
    Creates a geometrized vector grid of panels rotated to match the roof azimuth.
    """
    sunny_pts = mask_to_polygon(sunny_mask)
    if not sunny_pts:
        return []
    
    # 1. Create the base polygon and buffer inward for safety
    sunny_poly = Polygon(sunny_pts).buffer(-1)
    
    # 2. Rotate the roof polygon so its edges are axis-aligned (North-Up)
    # We rotate by -azimuth to "un-rotate" it for grid generation
    center = sunny_poly.centroid
    aligned_poly = affinity.rotate(sunny_poly, -azimuth, origin=center)
    
    # 3. Dimensions in pixels
    pw_px = panel_w / gsd
    ph_px = panel_h / gsd
    
    # 4. Generate grid on the aligned polygon
    minx, miny, maxx, maxy = aligned_poly.bounds
    aligned_panels = []
    
    for x in np.arange(minx, maxx, pw_px):
        for y in np.arange(miny, maxy, ph_px):
            p = Polygon([(x, y), (x+pw_px, y), (x+pw_px, y+ph_px), (x, y+ph_px)])
            if aligned_poly.contains(p):
                aligned_panels.append(p)
    
    # 5. Rotate the panels back to the original roof orientation
    final_panels = [affinity.rotate(p, azimuth, origin=center) for p in aligned_panels]
    
    return final_panels

def show():
    st.header("Step 3: Solar And Irradiance Analysis")
    
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
    lat = st.session_state.data.get("confirmed_lat", 53.5511)
    lon = st.session_state.data.get("confirmed_lon", 9.99)
    zoom_level = 19 

    # 2. GENERATE MASKED ARRAY
    mask, roof_only = get_masked_roof_array(zoom_img, poly_pts)

    # 3. INITIAL AUTOMATED DETECTION
    if "auto_roof_type" not in st.session_state.data:
        roof_type, variance = analyze_roof_texture(roof_only, mask)
        
        # --- PLACE THE LINE HERE ---
        # Passing 'roof_only' allows the brightness heuristic to work
        auto_azimuth = calculate_azimuth(poly_pts, img=roof_only)
        
        # Initialize the shadow threshold
        gray_roof = cv2.cvtColor(roof_only, cv2.COLOR_BGR2GRAY)
        roof_pixels = gray_roof[mask > 0]
        mean_brightness = np.mean(roof_pixels) if len(roof_pixels) > 0 else 128
        initial_threshold = min(255, int(mean_brightness + ui.SHADOW_BIAS))
        
        # Save to session state
        st.session_state.data.update({
            "auto_roof_type": roof_type,
            "user_azimuth": float(auto_azimuth),
            "user_tilt": 38.0 if roof_type == "Pitched" else 0.0,
            "sun_threshold": initial_threshold 
        })

    # 4. UI LAYOUT
    col_L, col_center, col_R = st.columns([1, 4, 1.5])

    with col_center:
        st.markdown("### 🛠️ Geometry And Shadow Mapping")
        
        st.slider("Shadow Sensitivity", 0, 255, int(st.session_state.data["sun_threshold"]),
                  key="sun_slider_widget", on_change=update_threshold)
        
        current_threshold = st.session_state.data["sun_threshold"]
        sun_mask = get_sunny_polygon_mask(roof_only, current_threshold)
        current_gsd = calculate_global_gsd(lat, zoom_level)
        current_azimuth = st.session_state.data["user_azimuth"]
        
        # --- GENERATE ROTATED PANEL GRID ---
        fitted_panels = generate_panel_grid(sun_mask, current_gsd, current_azimuth)
        
        # Rendering
        viz_img = roof_only.copy()
        sunny_overlay = viz_img.copy()
        sunny_overlay[sun_mask > 0] = [0, 255, 255]
        viz_img = cv2.addWeighted(viz_img, 0.7, sunny_overlay, 0.3, 0)
        
        for p in fitted_panels:
            pts = np.array(p.exterior.coords, np.int32)
            cv2.polylines(viz_img, [pts], True, (255, 0, 0), 1)
            overlay = viz_img.copy()
            cv2.fillPoly(overlay, [pts], (255, 200, 0))
            viz_img = cv2.addWeighted(overlay, 0.4, viz_img, 0.6, 0)

        final_preview = draw_azimuth_arrow(viz_img, current_azimuth)
        st.image(final_preview, caption="Panels Aligned to Roof Orientation", width=ui.DISPLAY_WIDTH)
        
        # USER CONTROLS
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            selected_type = c1.selectbox("Roof Form", ["Flat", "Pitched"], 
                                         index=0 if st.session_state.data["auto_roof_type"] == "Flat" else 1)
            
            if selected_type != st.session_state.data["auto_roof_type"]:
                st.session_state.data["auto_roof_type"] = selected_type
                st.rerun()

            tilt = c2.number_input("Tilt Angle (°)", 0.0, 90.0, st.session_state.data["user_tilt"])
            st.session_state.data["user_tilt"] = tilt
            
            c3.slider("Orientation (°)", 0, 359, int(st.session_state.data["user_azimuth"]),
                      key="az_slider_widget", on_change=update_azimuth)

        st.divider()
        
        # NAVIGATION And SIMULATION
        btn_col_1, btn_col_2 = st.columns(2)
        if btn_col_1.button("⬅️ Back to Step 2", use_container_width=True):
            st.session_state.step = 2
            st.rerun()
            
        if btn_col_2.button("Run Simulation ☀️", type="primary", use_container_width=True):
            irrad_val = calculate_solar_potential(lat, lon, tilt, st.session_state.data["user_azimuth"])
            st.session_state.data["solar_results"] = {
                "irradiance_w_m2": irrad_val,
                "panel_count": len(fitted_panels),
                "total_kwp": (len(fitted_panels) * 400) / 1000 
            }
            st.session_state.step = 4
            st.rerun()

    with col_R:
        st.markdown("### 📊 Global Metrics")
        st.metric("Panels Fitted", f"{len(fitted_panels)}")
        st.metric("Potential Capacity", f"{(len(fitted_panels) * 400) / 1000:.2f} kWp")
        st.info(f"Panels are now dynamically aligned to the {current_azimuth:.1f}° azimuth.")