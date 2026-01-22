import streamlit as st
import ui_components as ui

def show(lat_input, lon_input):
    st.subheader("Step 1: Locate Your Property")
    
    col_L, col_center, col_R = st.columns([1, 4, 1.5])
    
    with col_center:
        st.markdown("### 📍 Confirm Project Location")
        with st.container(border=True):
            st.write("")
            st.markdown(f"<h2 style='text-align: center;'>{lat_input}, {lon_input}</h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: gray;'>[Interactive Map Placeholder]</p>", unsafe_allow_html=True)
            st.write("")

        st.divider()
        gap = (ui.DISPLAY_WIDTH - ui.BUTTON_SIZE_VERIFY) / 2
        _, sub_col, _ = st.columns([gap, ui.BUTTON_SIZE_VERIFY, gap])
        
        with sub_col:
            if st.button("🚀 Analyze Building", use_container_width=True, type="primary"):
                st.session_state.data["confirmed_lat"] = lat_input
                st.session_state.data["confirmed_lon"] = lon_input
                st.session_state.step = 2
                st.rerun()