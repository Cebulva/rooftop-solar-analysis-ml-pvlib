import streamlit as st

# Global Constants
MODEL_PATH = "models/model_zoom19.pth"
AI_ZOOM = 19
GSD_19 = 0.298 
DISPLAY_WIDTH = 700
BUTTON_SIZE_VERIFY = 350

def render_sidebar():
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
        return lat, lon