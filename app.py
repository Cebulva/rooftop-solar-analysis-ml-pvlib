import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import cv2
import uuid
import io
import streamlit.elements.image as st_image

# --- 1. COMPATIBILITY & CACHING ---
if not hasattr(st_image, 'image_to_url'):
    def image_to_url(data, width, height, clamp, channels, output_format, image_id=None):
        if image_id is None: image_id = str(uuid.uuid4())
        if isinstance(data, Image.Image):
            buffered = io.BytesIO(); data.save(buffered, format="PNG")
            data = buffered.getvalue()
        from streamlit.runtime import get_instance
        return get_instance().media_file_mgr.add(data, output_format, image_id)
    st_image.image_to_url = image_to_url

@st.cache_resource
def get_model(path):
    from src.model_engine import load_roof_model
    return load_roof_model(path)

@st.cache_data
def get_image_data(lat, lon, zoom):
    from src.map_utils import get_aerial_image_tensor
    return get_aerial_image_tensor(lat, lon, zoom=zoom)

# --- 2. CONFIGURATION ---
st.set_page_config(page_title="SolarSight AI", layout="wide")
MODEL_PATH = "models/model_zoom19.pth"
AI_ZOOM = 19
GSD_19 = 0.298 

DISPLAY_WIDTH = 700  # Adjust this to change the size of all images/canvases
BUTTON_SIZE_VERIFY = 350 # Adjust this to change the size of the button during the verification

# Initialize session state keys
if 'step' not in st.session_state: st.session_state.step = 1
if 'data' not in st.session_state: st.session_state.data = {}

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("SolarSight AI")
    lat = st.number_input("Latitude", value=53.631249, format="%.6f")
    lon = st.number_input("Longitude", value=10.089139, format="%.6f")
    
    st.divider()
    if st.button("New Search", type="primary"):
        st.session_state.step = 1
        st.session_state.data = {}
        st.cache_data.clear()
        st.rerun()

# --- 4. STAGE 1: STABLE PREVIEW ---
if st.session_state.step == 1:
    st.subheader("Step 1: Confirm Target Building")
    coord_key = f"{lat}_{lon}"
    
    # Data fetching and AI Inference (cached for speed)
    if st.session_state.data.get("key") != coord_key:
        with st.spinner("Fetching stable imagery..."):
            # --- HYBRID OSM + ML MASK ---
            from src.model_engine import run_roof_pipeline
            model = get_model(MODEL_PATH)
            
            # Always use same zoom inside run_roof_pipeline
            gray_mask, source, full_img = run_roof_pipeline(model, lat, lon)
            
            from src.image_processing import filter_non_roof_objects
            clean_mask = filter_non_roof_objects(gray_mask)
            # -------------------------------
            
            # Create the initial cyan preview overlay
            overlay = full_img.copy()
            # RGB Cyan: (0, 255, 255) ensures we don't get yellow
            mask_layer = overlay.copy()
            mask_layer[clean_mask > 0] = [0, 1, 1] 
            overlay = cv2.addWeighted(overlay, 0.6, mask_layer, 0.4, 0)
            
            st.session_state.data = {
                "key": coord_key,
                "full_img": full_img,
                "full_mask": clean_mask,
                "preview_overlay": overlay
            }

    # Standardized Layout for UI Stability [Spacer, Content, Dashboard/Spacer]
    col_L, col_center, col_R = st.columns([1, 4, 1.5])
    
    with col_center:
        # Standardized Image Width
        st.image(st.session_state.data["preview_overlay"], width=DISPLAY_WIDTH)
        
        # Centering the button beneath the image to match BUTTON_SIZE_VERIFY
        gap_ratio = (DISPLAY_WIDTH - BUTTON_SIZE_VERIFY) / 2
        _, sub_col, _ = st.columns([gap_ratio, BUTTON_SIZE_VERIFY, gap_ratio])
        
        with sub_col:
            if st.button("✅ Confirm Building", use_container_width=True, type="primary"):
                st.session_state.step = 2
                st.rerun()

# --- 5. STAGE 2: CONFIRMATION And REFINEMENT LOOP ---
elif st.session_state.step == 2:
    # Initialization (Runs once when entering Step 2)
    if "res" not in st.session_state.data:
        from src.image_processing import get_zoom_crop
        from src.geometry_utils import mask_to_polygon
        z_img, z_mask, offsets = get_zoom_crop(st.session_state.data["full_img"], st.session_state.data["full_mask"])
        poly_points = mask_to_polygon(z_mask)
        st.session_state.data["res"] = {'zoom_img': z_img, 'initial_poly': poly_points, 'offsets': offsets}
        st.session_state.sub_step = "verify"

    res = st.session_state.data["res"]
    
    # Standardize image for scaling
    base_img_uint8 = (res['zoom_img'] * 255).astype(np.uint8)
    h, w = base_img_uint8.shape[:2]
    scaling_factor = DISPLAY_WIDTH / w
    display_h = int(h * scaling_factor)
    
    # Pre-resize base image
    base_resized = cv2.resize(base_img_uint8, (DISPLAY_WIDTH, display_h), interpolation=cv2.INTER_LANCZOS4)

    # --- SUB-STEP A: VERIFY AI MASK ---
    if st.session_state.sub_step == "verify":
        st.subheader("Step 2a: Verify AI Detection")
        
        col_L, col_center, col_R = st.columns([1, 4, 1.5])
        
        with col_center:
            preview_draw = base_resized.copy()
            if res['initial_poly'] is not None and len(res['initial_poly']) > 0:
                ai_pts = (np.array(res['initial_poly']) * scaling_factor).astype(np.int32)
                
                # 1. Create a transparent layer
                mask_layer = preview_draw.copy()
                
                # 2. DEFINITIVE CYAN FOR RGB: (Red=0, Green=255, Blue=255)
                # This will give you the same bright Cyan as your manual drawing.
                RGB_CYAN = (0, 255, 255) 
                
                cv2.fillPoly(mask_layer, [ai_pts], RGB_CYAN)
                
                # 3. Blend at 0.6 / 0.4 for professional transparency
                preview_draw = cv2.addWeighted(preview_draw, 0.6, mask_layer, 0.4, 0)
            
            st.image(preview_draw, width=DISPLAY_WIDTH, caption="AI Suggested Roof Boundary")
            
            # --- Centered Button Logic ---
            gap_ratio = (DISPLAY_WIDTH - BUTTON_SIZE_VERIFY) / 2
            _, sub_col, _ = st.columns([gap_ratio, BUTTON_SIZE_VERIFY, gap_ratio])
            
            with sub_col:
                btn_left, btn_right = st.columns(2)
                with btn_left:
                    if st.button("✅ Correct", use_container_width=True, type="primary"):
                        st.session_state.data["final_poly"] = res['initial_poly']
                        st.session_state.step = 3
                        st.rerun()
                with btn_right:
                    if st.button("✏️ Adjust", use_container_width=True):
                        st.session_state.sub_step = "adjust"
                        st.rerun()

    # --- SUB-STEP B: INTEGRATED DRAGGABLE EDITOR (Stable Version) ---
    elif st.session_state.sub_step == "adjust":
        st.subheader("Step 2b: Refine Roof Vertices")

        bg_for_editor = Image.fromarray(base_resized)

        # 1. Initialization (Only runs once)
        if "adjust_canvas_init" not in st.session_state.data:
            scaled_pts = [{"x": p[0] * scaling_factor, "y": p[1] * scaling_factor} 
                          for p in res['initial_poly']]
            
            objects = []
            path_str = "M " + " L ".join([f"{p['x']} {p['y']}" for p in scaled_pts]) + " Z"
            objects.append({
                "type": "path", "path": path_str,
                "fill": "rgba(0, 255, 255, 0.3)", "stroke": "#00FFFF", "strokeWidth": 2,
                "selectable": False, "evented": False, "name": "mask_fill"
            })
            for i, pt in enumerate(scaled_pts):
                objects.append({
                    "type": "circle", "left": pt['x'] - 8, "top": pt['y'] - 8,
                    "radius": 8, "fill": "#00FFFF", "stroke": "#000000",
                    "strokeWidth": 2, "selectable": True, "hasControls": False,
                    "hasBorders": False, "name": f"v_{i:03d}"
                })
            st.session_state.data["adjust_canvas_init"] = {"version": "4.4.0", "objects": objects}

        col_L, col_center, col_dash = st.columns([1, 4, 1.5])

        with col_center:
            canvas_result = st_canvas(
                background_image=bg_for_editor,
                initial_drawing=st.session_state.data.get("adjust_canvas_init"),
                drawing_mode="transform",
                display_toolbar=False,
                update_streamlit=True,
                height=display_h,
                width=DISPLAY_WIDTH,
                key="integrated_editor_vFinal",
            )

        # 2. SELECTIVE UPDATE LOGIC (Stops the Loop)
        extracted_pts_raw = []
        if canvas_result.json_data and "objects" in canvas_result.json_data:
            circles = [obj for obj in canvas_result.json_data["objects"] if obj["type"] == "circle"]
            circles = sorted(circles, key=lambda c: c.get("name", ""))
            
            if circles:
                has_moved = False
                new_pts_scaled = []
                
                for i, c in enumerate(circles):
                    # Get current pos from canvas
                    curr_x = c["left"] + 8
                    curr_y = c["top"] + 8
                    
                    # Get previous pos from session state
                    prev_x = st.session_state.data["adjust_canvas_init"]["objects"][i+1]["left"] + 8
                    prev_y = st.session_state.data["adjust_canvas_init"]["objects"][i+1]["top"] + 8
                    
                    # Check for movement (using a 0.5px threshold to ignore micro-jitters)
                    if abs(curr_x - prev_x) > 0.5 or abs(curr_y - prev_y) > 0.5:
                        has_moved = True
                    
                    new_pts_scaled.append({"x": curr_x, "y": curr_y})
                    extracted_pts_raw.append([curr_x / scaling_factor, curr_y / scaling_factor])

                # ONLY update state if something actually moved
                if has_moved:
                    new_path_str = "M " + " L ".join([f"{p['x']} {p['y']}" for p in new_pts_scaled]) + " Z"
                    st.session_state.data["adjust_canvas_init"]["objects"][0]["path"] = new_path_str
                    for i, p in enumerate(new_pts_scaled):
                        st.session_state.data["adjust_canvas_init"]["objects"][i+1]["left"] = p["x"] - 8
                        st.session_state.data["adjust_canvas_init"]["objects"][i+1]["top"] = p["y"] - 8
                    
                    # Trigger a single rerun to lock in the new state
                    st.rerun()

        with col_dash:
            st.markdown("### 📊 Metrics")
            if extracted_pts_raw:
                area = cv2.contourArea(np.array(extracted_pts_raw).astype(np.float32)) * (GSD_19**2)
                st.metric("Refined Area", f"{area:.1f} m²")
                
                st.divider()
                st.write("### Actions")
                
                # --- NEW BUTTON: TRIGGER DRAW NEW ---
                if st.button("✨ Draw from Scratch", use_container_width=True):
                    # Clear the draggable state so it doesn't conflict
                    if "adjust_canvas_init" in st.session_state.data:
                        del st.session_state.data["adjust_canvas_init"]
                    st.session_state.sub_step = "draw_new"
                    st.rerun()

                c1, c2 = st.columns(2)
                with c1:
                    if st.button("💾 Save", type="primary", use_container_width=True):
                        st.session_state.data["final_poly"] = extracted_pts_raw
                        st.session_state.sub_step = "verify_custom"
                        st.rerun()
                with c2:
                    if st.button("🔄 Reset", use_container_width=True):
                        if "adjust_canvas_init" in st.session_state.data:
                            del st.session_state.data["adjust_canvas_init"]
                        st.rerun()

            if st.button("⬅️ Back", use_container_width=True):
                st.session_state.sub_step = "verify"
                st.rerun()

    # --- SUB-STEP B2: DRAW NEW SHAPE FROM SCRATCH ---
    elif st.session_state.sub_step == "draw_new":
        st.subheader("Step 2b: Draw New Roof Boundary")

        bg_for_editor = Image.fromarray(base_resized)

        col_L, col_center, col_dash = st.columns([1, 4, 1.5])

        with col_center:
            canvas_result = st_canvas(
                fill_color="rgba(0, 255, 255, 0.25)",
                stroke_width=3,
                stroke_color="#00FFFF",
                background_image=bg_for_editor,
                drawing_mode="polygon",
                point_display_radius=5,
                update_streamlit=True,
                height=display_h,
                width=DISPLAY_WIDTH,
                key="draw_new_canvas",
            )
            st.info("🎯 **Draw New**: Click each corner of the roof. **Right-click** or **double-click** the last point to complete.")

        with col_dash:
            st.markdown("### 📊 Metrics")
            extracted_pts = None

            if canvas_result.json_data and canvas_result.json_data["objects"]:
                obj = canvas_result.json_data["objects"][-1]
                path = obj.get("path", [])
                extracted_pts = [[p[1]/scaling_factor, p[2]/scaling_factor]
                                 for p in path if len(p) > 2]

                if extracted_pts and len(extracted_pts) >= 3:
                    area = cv2.contourArea(np.array(extracted_pts).astype(np.float32)) * (GSD_19**2)
                    st.metric("Roof Area", f"{area:.1f} m²")
                    st.write(f"**Vertices:** {len(extracted_pts)}")

                    st.divider()
                    if st.button("💾 Save Shape", type="primary", use_container_width=True):
                        st.session_state.data["final_poly"] = extracted_pts
                        st.session_state.sub_step = "verify_custom"
                        st.rerun()

            st.divider()
            if st.button("⬅️ Back to Adjust", use_container_width=True):
                st.session_state.sub_step = "adjust"
                st.rerun()

    # --- SUB-STEP C: VERIFY CUSTOM MASK ---
    elif st.session_state.sub_step == "verify_custom":
        st.subheader("Step 2c: Confirm Your Manual Mask")
        
        col_L, col_center, col_R = st.columns([1, 4, 1.5])
        
        with col_center:
            final_preview = base_resized.copy()
            user_pts = (np.array(st.session_state.data["final_poly"]) * scaling_factor).astype(np.int32)
            
            mask_layer = final_preview.copy()
            cv2.fillPoly(mask_layer, [user_pts], (0, 255, 255))
            final_preview = cv2.addWeighted(final_preview, 0.6, mask_layer, 0.4, 0)
            
            st.image(final_preview, width=DISPLAY_WIDTH, caption="Your Refined Mask")
            
            gap_ratio = (DISPLAY_WIDTH - BUTTON_SIZE_VERIFY) / 2
            _, sub_col, _ = st.columns([gap_ratio, BUTTON_SIZE_VERIFY, gap_ratio])
            
            with sub_col:
                btn_left, btn_right = st.columns(2)
                with btn_left:
                    if st.button("✅ Perfect", use_container_width=True, type="primary"):
                        st.session_state.step = 3
                        st.rerun()
                with btn_right:
                    if st.button("🔄 Redraw", use_container_width=True):
                        # --- THE FIX: Convert current final_poly into the new initial state ---
                        current_pts = st.session_state.data["final_poly"]
                        scaled_pts = [{"x": p[0] * scaling_factor, "y": p[1] * scaling_factor} for p in current_pts]
                        
                        path_str = "M " + " L ".join([f"{p['x']} {p['y']}" for p in scaled_pts]) + " Z"
                        objects = [{"type": "path", "path": path_str, "fill": "rgba(0, 255, 255, 0.3)", 
                                    "stroke": "#00FFFF", "strokeWidth": 2, "selectable": False, "evented": False}]
                        
                        for i, pt in enumerate(scaled_pts):
                            objects.append({"type": "circle", "left": pt['x'] - 8, "top": pt['y'] - 8,
                                            "radius": 8, "fill": "#00FFFF", "stroke": "#000000",
                                            "strokeWidth": 2, "selectable": True, "hasControls": False, 
                                            "name": f"v_{i:03d}"})
                        
                        # Set this as the new starting point for the 'adjust' step
                        st.session_state.data["adjust_canvas_init"] = {"version": "4.4.0", "objects": objects}
                        st.session_state.sub_step = "adjust"
                        st.rerun()