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
MODEL_PATH = "models/segm_Unet_model_aerial.pth"
AI_ZOOM = 19
GSD_19 = 0.298 

# Initialize session state keys
if 'step' not in st.session_state: st.session_state.step = 1
if 'data' not in st.session_state: st.session_state.data = {}

# Helper: Inject Midpoints for better editing
def inject_midpoints(points_list):
    new_pts = []
    for i in range(len(points_list)):
        p1 = points_list[i]
        p2 = points_list[(i + 1) % len(points_list)]
        mid = {"x": (p1["x"] + p2["x"]) / 2, "y": (p1["y"] + p2["y"]) / 2}
        new_pts.extend([p1, mid])
    return new_pts

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("SolarSight AI")
    lat = st.number_input("Latitude", value=53.631249774510124, format="%.6f")
    lon = st.number_input("Longitude", value=10.08913954324913, format="%.6f")
    
    if st.button("New Search", type="primary"):
        st.session_state.step = 1
        st.session_state.data = {}
        st.cache_data.clear()
        st.rerun()

# --- 4. STAGE 1: STABLE PREVIEW ---
if st.session_state.step == 1:
    st.subheader("Step 1: Confirm Target Building")
    
    # Only process if we don't have this specific coordinate's data
    coord_key = f"{lat}_{lon}"
    if st.session_state.data.get("key") != coord_key:
        with st.spinner("Fetching stable imagery..."):
            input_tensor = get_image_data(lat, lon, AI_ZOOM)
            full_img = input_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
            
            from src.model_engine import run_inference
            mask_np = run_inference(get_model(MODEL_PATH), input_tensor)
            
            from src.image_processing import filter_non_roof_objects
            clean_mask = filter_non_roof_objects((mask_np * 255).astype('uint8'))
            
            overlay = full_img.copy()
            overlay[clean_mask > 0] = [0, 1, 1]
            
            st.session_state.data = {
                "key": coord_key,
                "full_img": full_img,
                "full_mask": clean_mask,
                "preview_overlay": overlay
            }

    _, col_m, _ = st.columns([1, 2, 1])
    with col_m:
        st.image(st.session_state.data["preview_overlay"], use_container_width=True)
        if st.button("✅ Confirm Building", use_container_width=True):
            st.session_state.step = 2
            st.rerun()

# --- 5. STAGE 2: STABLE EDITOR ---
elif st.session_state.step == 2:
    st.subheader("Step 2: Professional Geometry Editor")
    
    # Prepare Zoomed Data Once
    if "res" not in st.session_state.data:
        from src.image_processing import get_zoom_crop
        from src.geometry_utils import mask_to_polygon
        
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

    res = st.session_state.data["res"]
    target_width = 850
    scaling_factor = target_width / res['zoom_img'].shape[1]
    display_h = int(res['zoom_img'].shape[0] * scaling_factor)
    
    # Cache the background image object
    bg_display = Image.fromarray((res['zoom_img']*255).astype(np.uint8)).resize((target_width, display_h), Image.LANCZOS)
    
    # Setup Polyline
    initial_draw = None
    if res['initial_poly'] and "canvas_init" not in st.session_state.data:
        # 1. Scale points to the UI display size
        raw_scaled = [{"x": p[0]*scaling_factor, "y": p[1]*scaling_factor} for p in res['initial_poly']]
        
        # 2. Inject midpoints
        scaled_pts = inject_midpoints(raw_scaled)
        
        # 3. CALCULATE BOUNDING BOX (Crucial for centering)
        xs = [p['x'] for p in scaled_pts]
        ys = [p['y'] for p in scaled_pts]
        min_x, min_y = min(xs), min(ys)
        
        # 4. NORMALIZE POINTS (Make them relative to the top-left of the shape)
        # This prevents the canvas from 'jumping' the points away from the image
        normalized_pts = [{"x": p['x'] - min_x, "y": p['y'] - min_y} for p in scaled_pts]
        
        initial_draw = {
            "version": "4.4.0",
            "objects": [{
                "type": "polyline",
                "points": normalized_pts,
                "left": min_x,  # Set the actual position on the image
                "top": min_y,
                "fill": "rgba(0, 255, 255, 0.2)",
                "stroke": "#00FFFF",
                "strokeWidth": 3,
                "closed": True,
                "cornerSize": 12,
                "cornerColor": "#00FFFF",
                "transparentCorners": False
            }]
        }
        st.session_state.data["canvas_init"] = initial_draw

    col_edit, col_stats = st.columns([3, 1])
    
    with col_edit:
        canvas_result = st_canvas(
            fill_color="rgba(0, 255, 255, 0.2)",
            stroke_width=3,
            stroke_color="#00FFFF",
            background_image=bg_display,
            initial_drawing=st.session_state.data.get("canvas_init"),
            drawing_mode="transform",
            point_display_radius=8,
            update_streamlit=True,
            height=display_h,
            width=target_width,
            key="stable_editor",
        )

    with col_stats:
        if canvas_result.json_data:
            objs = canvas_result.json_data["objects"]
            if objs:
                obj = objs[-1]
                l, t = obj.get('left', 0), obj.get('top', 0)
                pts = [[(p['x']+l)/scaling_factor, (p['y']+t)/scaling_factor] for p in obj.get('points', [])]
                if pts:
                    area = cv2.contourArea(np.array(pts).astype(np.float32)) * (GSD_19**2)
                    st.metric("Refined Area", f"{area:.1f} m²")
                    st.write(f"**Vertices:** {len(pts)}")

        if st.button("⬅️ Restart Search"):
            st.session_state.step = 1
            st.session_state.data = {}
            st.rerun()