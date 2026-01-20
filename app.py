import streamlit as st
import streamlit.elements.image as st_image
import uuid
import io
import os
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import cv2

# --- 1. COMPATIBILITY BRIDGE (Monkey Patch for Streamlit 1.52.2) ---
if not hasattr(st_image, 'image_to_url'):
    def image_to_url(data, width, height, clamp, channels, output_format, image_id=None):
        if image_id is None:
            image_id = str(uuid.uuid4())
        if isinstance(data, Image.Image):
            buffered = io.BytesIO()
            data.save(buffered, format="PNG")
            data = buffered.getvalue()
        from streamlit.runtime import get_instance
        runtime = get_instance()
        if runtime:
            return runtime.media_file_mgr.add(data, output_format, image_id)
        return ""
    st_image.image_to_url = image_to_url

# --- 2. MODULAR IMPORTS ---
from src.model_engine import load_roof_model, run_inference
from src.map_utils import get_aerial_image_tensor
from src.image_processing import filter_non_roof_objects, get_zoom_crop, format_poly_for_canvas
from src.geometry_utils import mask_to_polygon, calculate_azimuth

# --- 3. CONFIGURATION & CONSTANTS ---
st.set_page_config(page_title="SolarSight AI", layout="wide")
st.title("☀️ SolarSight: AI Roof Analysis And Planning")

MODEL_PATH = "models/segm_Unet_model_aerial.pth"
AI_ZOOM = 19  # Hard-coded to match model training
GSD_19 = 0.298 # Meters per pixel at Zoom 19 (approx)

if 'step' not in st.session_state:
    st.session_state.step = 1
if 'results' not in st.session_state:
    st.session_state.results = None

# --- 4. SIDEBAR CONTROL ---
with st.sidebar:
    st.header("Site Selection")
    lat = st.number_input("Latitude", value=53.631249774510124, format="%.6f")
    lon = st.number_input("Longitude", value=10.08913954324913, format="%.6f")
    
    st.warning(f"🎯 Model optimized for **Zoom {AI_ZOOM}**.")
    
    if st.button("Search Location", type="primary"):
        st.session_state.step = 1
        st.session_state.results = None
        st.session_state.full_img = None
        st.session_state.full_mask = None

# --- 5. STAGE 1: LOCATION PREVIEW ---
if st.session_state.step == 1:
    st.subheader("Step 1: Confirm Building Location")
    
    with st.spinner("Fetching satellite preview..."):
        input_tensor = get_aerial_image_tensor(lat, lon, zoom=AI_ZOOM)
        full_img = input_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        
        # 1. Run AI
        model = load_roof_model(MODEL_PATH)
        raw_mask_np = run_inference(model, input_tensor)
        
        # 2. Convert to 8-bit and FILTER IMMEDIATELY
        mask_8u = (raw_mask_np * 255).astype('uint8')
        clean_mask_8u = filter_non_roof_objects(mask_8u) # This picks the central/largest roof
        
        # 3. Create a clean overlay
        overlay = full_img.copy()
        # Highlight ONLY the filtered roof
        overlay[clean_mask_8u > 0] = [0, 1, 1] 
        
        # Save the CLEAN mask to session state for Step 2
        st.session_state.full_img = full_img
        st.session_state.full_mask = clean_mask_8u / 255.0 # Back to 0-1 float for consistency

    st.image(overlay, caption="Primary Building Identified (Cyan)", width=700)
    
    if st.button("✅ Yes, Analyze This Building", width=700):
        st.session_state.step = 2
        st.session_state.results = None # Force Step 2 to recalculate based on this clean mask
        st.rerun()

# --- 6. STAGE 2: GEOMETRY REFINEMENT ---
elif st.session_state.step == 2:
    st.subheader("Step 2: Refine Roof Geometry")
    
    if st.session_state.results is None:
        mask_8u = (st.session_state.full_mask * 255).astype('uint8')
        clean_mask = filter_non_roof_objects(mask_8u)
        
        z_img, z_mask, offsets = get_zoom_crop(st.session_state.full_img, clean_mask)
        poly_points = mask_to_polygon(z_mask)
        azimuth = calculate_azimuth(poly_points) if poly_points else 0.0

        st.session_state.results = {
            'zoom_img': z_img,
            'initial_poly': poly_points,
            'azimuth': azimuth,
            'offsets': offsets
        }

    res = st.session_state.results
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Scaling logic for clear visibility
        target_width = 800  
        original_h, original_w = res['zoom_img'].shape[:2]
        scaling_factor = target_width / original_w
        display_h = int(original_h * scaling_factor)

        img_data = (res['zoom_img'] * 255).astype(np.uint8)
        bg_pil = Image.fromarray(img_data)
        bg_display = bg_pil.resize((target_width, display_h), resample=Image.LANCZOS)
        
        # Prepare points
        scaled_poly = []
        if res.get('initial_poly') is not None:
            scaled_poly = [[p[0] * scaling_factor, p[1] * scaling_factor] for p in res['initial_poly']]
        else:
            cx, cy = target_width // 2, display_h // 2
            scaled_poly = [[cx-50, cy-50], [cx+50, cy-50], [cx+50, cy+50], [cx-50, cy+50]]

        canvas_result = st_canvas(
            fill_color="rgba(0, 255, 255, 0.3)",
            stroke_width=3,
            stroke_color="#00FFFF",
            background_image=bg_display,
            initial_drawing=format_poly_for_canvas(scaled_poly),
            drawing_mode="transform",
            update_streamlit=True,
            height=display_h,
            width=target_width,
            key="roof_canvas",
        )

    with col2:
        st.subheader("Engineering Metrics")
        
        # Live Area Calculation
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            if len(objects) > 0:
                obj = objects[0]
                left, top = obj.get("left", 0), obj.get("top", 0)
                raw_path = obj.get("path", [])
                
                # Reverse scaling and offsets to get real pixel coordinates
                real_pts = []
                for pt in raw_path:
                    if len(pt) > 1:
                        real_pts.append([(pt[1] + left) / scaling_factor, (pt[2] + top) / scaling_factor])
                
                # Calculate Area
                pts_np = np.array(real_pts).astype(np.float32)
                pixel_area = cv2.contourArea(pts_np)
                m2_area = pixel_area * (GSD_19 ** 2)
                
                st.metric("Detected Area", f"{m2_area:.1f} m²")
                st.metric("Roof Azimuth", f"{res['azimuth']:.1f}°")
                st.write(f"**Est. Panels:** {int(m2_area / 1.7)}")
        
        if st.button("⬅️ Back to Preview", use_container_width=True):
            st.session_state.step = 1
            st.session_state.results = None
            st.rerun()

    st.info("💡 Adjust the polygon to fit the roof edges exactly for the best area estimate.")