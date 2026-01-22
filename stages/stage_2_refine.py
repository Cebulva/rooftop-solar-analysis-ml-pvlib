import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import ui_components as ui

def show():
    # 1. Initialize data if coming from Step 1
    if "res" not in st.session_state.data:
        from src.image_processing import get_zoom_crop
        from src.geometry_utils import mask_to_polygon
        
        # USE THE CLEAN IMAGE FOR CROPPING
        # We use 'full_img' which is the original aerial shot without the cyan tint
        z_img, z_mask, offsets = get_zoom_crop(
            st.session_state.data["full_img"], 
            st.session_state.data["full_mask"]
        )
        
        poly_points = mask_to_polygon(z_mask)
        st.session_state.data["res"] = {
            'zoom_img': z_img, 
            'initial_poly': poly_points, 
            'offsets': offsets
        }
        st.session_state.sub_step = "verify"

    res = st.session_state.data["res"]
    
    # 2. Prepare standardized background image for the editor
    # This 'res['zoom_img']' is now derived from the clean 'full_img'
    base_img_uint8 = (res['zoom_img']).astype(np.uint8) 
    
    h, w = base_img_uint8.shape[:2]
    scaling_factor = ui.DISPLAY_WIDTH / w
    display_h = int(h * scaling_factor)
    
    # Pre-resize the CLEAN base image for the canvas background
    base_resized = cv2.resize(
        base_img_uint8, 
        (ui.DISPLAY_WIDTH, display_h), 
        interpolation=cv2.INTER_LANCZOS4
    )

    # 3. SUB-STEP ROUTER (This fixes your NameError)
    if st.session_state.sub_step == "verify":
        render_verify(base_resized, scaling_factor, res)
    elif st.session_state.sub_step == "adjust":
        render_adjust(base_resized, scaling_factor, display_h, res)
    elif st.session_state.sub_step == "draw_new":
        render_draw_new(base_resized, scaling_factor, display_h)
    elif st.session_state.sub_step == "verify_custom":
        render_verify_custom(base_resized, scaling_factor)

# --- HELPER RENDER FUNCTIONS ---

def render_verify(base_resized, scaling_factor, res):
    st.subheader("Step 2a: Verify AI Detection")
    col_L, col_center, col_R = st.columns([1, 4, 1.5])
    
    with col_center:
        preview_draw = base_resized.copy()
        if res['initial_poly'] is not None and len(res['initial_poly']) > 0:
            ai_pts = (np.array(res['initial_poly']) * scaling_factor).astype(np.int32)
            mask_layer = preview_draw.copy()
            cv2.fillPoly(mask_layer, [ai_pts], (0, 255, 255))
            preview_draw = cv2.addWeighted(preview_draw, 0.6, mask_layer, 0.4, 0)
        
        st.image(preview_draw, width=ui.DISPLAY_WIDTH)
        gap = (ui.DISPLAY_WIDTH - ui.BUTTON_SIZE_VERIFY) / 2
        _, sub_col, _ = st.columns([gap, ui.BUTTON_SIZE_VERIFY, gap])
        
        with sub_col:
            btn_left, btn_right = st.columns(2)
            if btn_left.button("✅ Correct", use_container_width=True, type="primary"):
                st.session_state.data["final_poly"] = res['initial_poly']
                st.session_state.step = 3
                st.rerun()
            if btn_right.button("✏️ Adjust", use_container_width=True):
                st.session_state.sub_step = "adjust"
                st.rerun()

def render_adjust(base_resized, scaling_factor, display_h, res):
    st.subheader("Step 2b: Refine Roof Vertices")
    
    if "adjust_canvas_init" not in st.session_state.data:
        scaled_pts = [{"x": p[0]*scaling_factor, "y": p[1]*scaling_factor} for p in res['initial_poly']]
        path_str = "M " + " L ".join([f"{p['x']} {p['y']}" for p in scaled_pts]) + " Z"
        objects = [{"type": "path", "path": path_str, "fill": "rgba(0, 255, 255, 0.3)", "stroke": "#00FFFF", "strokeWidth": 2, "selectable": False, "evented": False}]
        for i, pt in enumerate(scaled_pts):
            objects.append({"type": "circle", "left": pt['x']-8, "top": pt['y']-8, "radius": 8, "fill": "#00FFFF", "stroke": "#000000", "strokeWidth": 2, "selectable": True, "hasControls": False, "name": f"v_{i:03d}"})
        st.session_state.data["adjust_canvas_init"] = {"version": "4.4.0", "objects": objects}

    col_L, col_center, col_dash = st.columns([1, 4, 1.5])
    with col_center:
        canvas_result = st_canvas(
            background_image=Image.fromarray(base_resized),
            initial_drawing=st.session_state.data.get("adjust_canvas_init"),
            drawing_mode="transform", display_toolbar=False, update_streamlit=True,
            height=display_h, width=ui.DISPLAY_WIDTH, key="integrated_editor"
        )

    extracted_pts_raw = []
    if canvas_result.json_data and "objects" in canvas_result.json_data:
        circles = sorted([o for o in canvas_result.json_data["objects"] if o["type"] == "circle"], key=lambda c: c.get("name", ""))
        if circles:
            has_moved = False
            new_pts_scaled = []
            for i, c in enumerate(circles):
                cx, cy = c["left"] + 8, c["top"] + 8
                prev_x = st.session_state.data["adjust_canvas_init"]["objects"][i+1]["left"] + 8
                prev_y = st.session_state.data["adjust_canvas_init"]["objects"][i+1]["top"] + 8
                if abs(cx - prev_x) > 0.5 or abs(cy - prev_y) > 0.5: has_moved = True
                new_pts_scaled.append({"x": cx, "y": cy})
                extracted_pts_raw.append([cx/scaling_factor, cy/scaling_factor])
            if has_moved:
                st.session_state.data["adjust_canvas_init"]["objects"][0]["path"] = "M " + " L ".join([f"{p['x']} {p['y']}" for p in new_pts_scaled]) + " Z"
                for i, p in enumerate(new_pts_scaled):
                    st.session_state.data["adjust_canvas_init"]["objects"][i+1]["left"], st.session_state.data["adjust_canvas_init"]["objects"][i+1]["top"] = p["x"]-8, p["y"]-8
                st.rerun()

    with col_dash:
        st.markdown("### 📊 Metrics")
        if extracted_pts_raw:
            area = cv2.contourArea(np.array(extracted_pts_raw).astype(np.float32)) * (ui.GSD_19**2)
            st.metric("Refined Area", f"{area:.1f} m²")
            st.divider()
            if st.button("✨ Draw from Scratch", use_container_width=True):
                if "adjust_canvas_init" in st.session_state.data: del st.session_state.data["adjust_canvas_init"]
                st.session_state.sub_step = "draw_new"; st.rerun()
            c1, c2 = st.columns(2)
            if c1.button("💾 Save", type="primary", use_container_width=True):
                st.session_state.data["final_poly"] = extracted_pts_raw
                st.session_state.sub_step = "verify_custom"; st.rerun()
            if c2.button("🔄 Reset", use_container_width=True):
                if "adjust_canvas_init" in st.session_state.data: del st.session_state.data["adjust_canvas_init"]
                st.rerun()
        if st.button("⬅️ Back", use_container_width=True):
            st.session_state.sub_step = "verify"; st.rerun()

def render_draw_new(base_resized, scaling_factor, display_h):
    st.subheader("Step 2b: Draw New Roof Boundary")
    col_L, col_center, col_dash = st.columns([1, 4, 1.5])
    with col_center:
        canvas_result = st_canvas(fill_color="rgba(0, 255, 255, 0.25)", stroke_width=3, stroke_color="#00FFFF", background_image=Image.fromarray(base_resized), drawing_mode="polygon", update_streamlit=True, height=display_h, width=ui.DISPLAY_WIDTH, key="draw_new_canvas")

    with col_dash:
        st.markdown("### 📊 Metrics")
        if canvas_result.json_data and canvas_result.json_data["objects"]:
            obj = canvas_result.json_data["objects"][-1]
            path = obj.get("path", [])
            extracted_pts = [[p[1]/scaling_factor, p[2]/scaling_factor] for p in path if len(p) > 2]
            if extracted_pts and len(extracted_pts) >= 3:
                area = cv2.contourArea(np.array(extracted_pts).astype(np.float32)) * (ui.GSD_19**2)
                st.metric("Roof Area", f"{area:.1f} m²")
                if st.button("💾 Save Shape", type="primary", use_container_width=True):
                    st.session_state.data["final_poly"] = extracted_pts
                    st.session_state.sub_step = "verify_custom"; st.rerun()
        st.divider()
        if st.button("⬅️ Back to Adjust", use_container_width=True):
            st.session_state.sub_step = "adjust"; st.rerun()

def render_verify_custom(base_resized, scaling_factor):
    st.subheader("Step 2c: Confirm Your Manual Mask")
    col_L, col_center, col_R = st.columns([1, 4, 1.5])
    with col_center:
        final_p = base_resized.copy()
        user_pts = (np.array(st.session_state.data["final_poly"]) * scaling_factor).astype(np.int32)
        m_layer = final_p.copy()
        cv2.fillPoly(m_layer, [user_pts], (0, 255, 255))
        final_p = cv2.addWeighted(final_p, 0.6, m_layer, 0.4, 0)
        st.image(final_p, width=ui.DISPLAY_WIDTH)
        gap = (ui.DISPLAY_WIDTH - ui.BUTTON_SIZE_VERIFY) / 2
        _, sub_col, _ = st.columns([gap, ui.BUTTON_SIZE_VERIFY, gap])
        with sub_col:
            c1, c2 = st.columns(2)
            if c1.button("✅ Perfect", use_container_width=True, type="primary"):
                st.session_state.step = 3; st.rerun()
            if c2.button("🔄 Redraw", use_container_width=True):
                # Convert final_poly back to draggable state
                pts = st.session_state.data["final_poly"]
                s_pts = [{"x": p[0]*scaling_factor, "y": p[1]*scaling_factor} for p in pts]
                path_str = "M " + " L ".join([f"{p['x']} {p['y']}" for p in s_pts]) + " Z"
                objs = [{"type": "path", "path": path_str, "fill": "rgba(0, 255, 255, 0.3)", "stroke": "#00FFFF", "strokeWidth": 2, "selectable": False, "evented": False}]
                for i, pt in enumerate(s_pts):
                    objs.append({"type": "circle", "left": pt['x']-8, "top": pt['y']-8, "radius": 8, "fill": "#00FFFF", "stroke": "#000000", "strokeWidth": 2, "selectable": True, "hasControls": False, "name": f"v_{i:03d}"})
                st.session_state.data["adjust_canvas_init"] = {"version": "4.4.0", "objects": objs}
                st.session_state.sub_step = "adjust"; st.rerun()