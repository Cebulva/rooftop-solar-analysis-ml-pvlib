import streamlit as st
import numpy as np
import uuid
import io
import streamlit.elements.image as st_image

# --- CUSTOM IMPORTS ---
import ui_components as ui
from stages import stage_1_preview, stage_2_refine, stage_3_solar, stage_4_report
from src import rag_bot

# --- COMPATIBILITY PATCH ---
if not hasattr(st_image, 'image_to_url'):
    def image_to_url(data, width, height, clamp, channels, output_format, image_id=None):
        if image_id is None: image_id = str(uuid.uuid4())
        from PIL import Image
        if isinstance(data, Image.Image):
            buffered = io.BytesIO(); data.save(buffered, format="PNG"); data = buffered.getvalue()
        from streamlit.runtime import get_instance
        return get_instance().media_file_mgr.add(data, output_format, image_id)
    st_image.image_to_url = image_to_url

# --- ENGINE CACHING ---
@st.cache_resource
def get_model(path):
    from src.model_engine import load_roof_model
    return load_roof_model(path)

# --- CONFIG ---
st.set_page_config(page_title="SolarSight AI", layout="wide")

if 'step' not in st.session_state: st.session_state.step = 1
if 'data' not in st.session_state: st.session_state.data = {}
if 'sub_step' not in st.session_state: st.session_state.sub_step = "verify"

# --- SIDEBAR ---
lat, lon = ui.render_sidebar()

# --- ROUTER ---
if st.session_state.step == 1:
    stage_1_preview.show(lat, lon, get_model)

elif st.session_state.step == 2:
    stage_2_refine.show(get_model)

elif st.session_state.step == 3:
    stage_3_solar.show()

elif st.session_state.step == 4:
    stage_4_report.show()

# --- PERSISTENT RAG BOT ---
# This stays at the bottom so it is globally available
rag_bot.initialize_chat()
rag_bot.render_chat_interface()