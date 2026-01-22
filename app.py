import streamlit as st
import numpy as np
import uuid
import io
import streamlit.elements.image as st_image

# 1. SHARED UI & CONSTANTS
import ui_components as ui
from stages import stage_1_preview, stage_2_refine, stage_3_solar, stage_4_report

# --- COMPATIBILITY PATCH ---
# Ensures internal Streamlit image handling works across all custom components
if not hasattr(st_image, 'image_to_url'):
    def image_to_url(data, width, height, clamp, channels, output_format, image_id=None):
        if image_id is None: image_id = str(uuid.uuid4())
        from PIL import Image
        if isinstance(data, Image.Image):
            buffered = io.BytesIO()
            data.save(buffered, format="PNG")
            data = buffered.getvalue()
        from streamlit.runtime import get_instance
        return get_instance().media_file_mgr.add(data, output_format, image_id)
    st_image.image_to_url = image_to_url

# --- CORE CACHING (The "Engine" interface) ---
@st.cache_resource
def get_model(path):
    from src.model_engine import load_roof_model
    return load_roof_model(path)

@st.cache_data
def get_image_data(lat, lon, zoom):
    from src.map_utils import get_aerial_image_tensor
    return get_aerial_image_tensor(lat, lon, zoom=zoom)

# --- 2. CONFIGURATION ---
st.set_page_config(
    page_title="SolarSight AI", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Initialize global session state keys
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'data' not in st.session_state:
    st.session_state.data = {}
if 'sub_step' not in st.session_state:
    st.session_state.sub_step = "verify"

# --- 3. SIDEBAR & NAVIGATION ---
# This returns the lat/lon values for Stage 1 to use
lat, lon = ui.render_sidebar()

# --- 4. STAGE ROUTER ---
# This section acts as the traffic controller for your team's different files
if st.session_state.step == 1:
    stage_1_preview.show(lat, lon, get_model)

elif st.session_state.step == 2:
    stage_2_refine.show()

elif st.session_state.step == 3:
    # New Stage: Heatmap and Panels
    stage_3_solar.show()

elif st.session_state.step == 4:
    # New Stage: Final Report
    stage_4_report.show()

# --- 5. FOOTER (Optional) ---
st.caption("SolarSight AI v0.2 | Modular Architecture (Stages And Engine)")