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

def generate_panel_grid(sunny_mask, gsd, azimuth, tilt, panel_w=1.76, panel_h=1.13):
    """
    Creates a grid of panels (Modern Trina/Jinko Glass-Glass dimensions).
    Adjusts the VISUAL height based on tilt angle (Cosine Projection).
    """
    sunny_pts = mask_to_polygon(sunny_mask)
    if not sunny_pts:
        return []
    
    # PROJECTED HEIGHT: Bird's eye view shrinks the panel height as tilt increases
    # 0 deg = full height, 90 deg = 0 height
    projected_h = panel_h * math.cos(math.radians(tilt))
    
    # Geometry setup
    sunny_poly = Polygon(sunny_pts).buffer(-0.5) 
    center = sunny_poly.centroid
    
    # Rotate to North-Up for grid placement
    aligned_poly = affinity.rotate(sunny_poly, -azimuth, origin=center)
    
    # Convert meters to pixels
    pw_px = panel_w / gsd
    ph_px = projected_h / gsd
    
    minx, miny, maxx, maxy = aligned_poly.bounds
    aligned_panels = []
    
    for x in np.arange(minx, maxx, pw_px):
        for y in np.arange(miny, maxy, ph_px):
            p = Polygon([(x, y), (x+pw_px, y), (x+pw_px, y+ph_px), (x, y+ph_px)])
            if aligned_poly.contains(p):
                aligned_panels.append(p)
    
    # Rotate back to real-world orientation
    return [affinity.rotate(p, azimuth, origin=center) for p in aligned_panels]

def show():
    st.header("Step 3b: Solar And Irradiance Analysis")
    
    if "final_poly" not in st.session_state.data:
        st.warning("Please complete the roof refinement in Step 2 first.")
        return

    res = st.session_state.data["res"]
    lat = st.session_state.data["confirmed_lat"]
    lon = st.session_state.data["confirmed_lon"]
    
    # 1. INITIALIZATION & DATA RETRIEVAL
    # Get the recommendation from the questionnaire (Stage 3a)
    recommended_limit = st.session_state.data.get("recommended_count", 20)
    
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

    # 3. Analysis
    if "auto_roof_type" not in st.session_state.data:
        detected_type, _ = analyze_roof_texture(roof_only, mask, sun_mask=sun_mask)
        auto_azimuth = calculate_azimuth(st.session_state.data["final_poly"], img=roof_only)
        st.session_state.data.update({
            "auto_roof_type": detected_type,
            "user_azimuth": float(auto_azimuth),
            "user_tilt": 38.0 if detected_type == "Pitched" else 0.0
        })

    current_azimuth = st.session_state.data["user_azimuth"]
    user_tilt = st.session_state.data["user_tilt"]

    # 4. Generate Panel Grid (Limited by Target Count)
    all_possible_panels = generate_panel_grid(sun_mask, gsd, current_azimuth, user_tilt)
    
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
        
        for p in panels:
            p_int = np.array(p.exterior.coords, dtype=np.int32)
            cv2.polylines(display_img, [p_int], True, (0, 255, 0), 1)

        st.image(display_img, use_container_width=True)
        
        with st.expander("🛠️ Analysis And Adjustments", expanded=True):
            # Restored Panel Limit Slider
            # Max value is the physical capacity of the roof
            max_capacity = len(all_possible_panels)
            st.slider("Number of Panels to Install", 1, max(1, max_capacity), 
                      int(st.session_state.data["target_panel_count"]), 
                      key="panel_slider_widget", 
                      help=f"Questionnaire recommended: {recommended_limit}",
                      on_change=lambda: st.session_state.data.update({"target_panel_count": st.session_state.panel_slider_widget}))

            c1, c2 = st.columns(2)
            selected_type = c1.selectbox("Roof Form", ["Pitched", "Flat"], 
                                       index=0 if st.session_state.data["auto_roof_type"] == "Pitched" else 1)
            
            # Re-run if type changes to update tilt
            if selected_type != st.session_state.data["auto_roof_type"]:
                st.session_state.data["auto_roof_type"] = selected_type
                st.session_state.data["user_tilt"] = 0.0 if selected_type == "Flat" else 38.0
                st.rerun()

            st.slider("Solar Orientation (Azimuth °)", 0, 359, int(current_azimuth), 
                      key="az_slider_widget", on_change=update_azimuth)
            
            st.slider("Shadow Tolerance (Threshold)", 0, 100, int(current_threshold), 
                      key="sun_slider_widget", on_change=update_threshold)

    

    with col_R:
        st.markdown("### 📊 Global Metrics")
        st.metric("Total Roof Area", f"{total_area_m2:.1f} m²")
        st.metric("Usable Space", f"{usable_area_m2:.1f} m²")
        st.metric("Selected Panels", f"{selected_count}", help=f"Targeting {st.session_state.data['target_panel_count']}")
        st.metric("System Size", f"{((selected_count * 440) / 1000):.2f} kWp")
        
        if st.button("Run Simulation And Generate Report ☀️", type="primary", use_container_width=True):
            irrad_val = calculate_solar_potential(lat, lon, user_tilt, current_azimuth)
            st.session_state.data["solar_results"] = {
                "total_roof_area_m2": total_area_m2,
                "usable_roof_area_m2": usable_area_m2,
                "panel_count": selected_count,
                "system_kwp": (selected_count * 440) / 1000,
                "azimuth": current_azimuth,
                "tilt_angle": user_tilt,
                "roof_form": st.session_state.data["auto_roof_type"],
                "irradiance_potential": irrad_val
            }
            st.session_state.step = 5
            st.rerun()