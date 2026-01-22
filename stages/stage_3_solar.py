import streamlit as st
import ui_components as ui

def show():
    st.subheader("Step 3: Solar Potential And Panel Layout")
    
    # Standard 3-column layout
    col_L, col_center, col_R = st.columns([1, 4, 1.5])
    
    with col_center:
        # Replaced st.rect with a styled container for a better placeholder
        st.info("💡 **Developer Note:** This area is reserved for the `folium` map or `plotly` heatmap.")
        
        # Placeholder visual box
        with st.container(border=True):
            st.write("")
            st.write("")
            st.markdown("<h3 style='text-align: center; color: gray;'>Heatmap Visualization Placeholder</h3>", unsafe_allow_html=True)
            st.write("")
            st.write("")
        
        st.divider()
        gap = (ui.DISPLAY_WIDTH - ui.BUTTON_SIZE_VERIFY) / 2
        _, sub_col, _ = st.columns([gap, ui.BUTTON_SIZE_VERIFY, gap])
        
        with sub_col:
            c1, c2 = st.columns(2)
            if c1.button("⬅️ Back", use_container_width=True):
                st.session_state.step = 2
                st.rerun()
            if c2.button("Generate Report ✅", use_container_width=True, type="primary"):
                st.session_state.step = 4
                st.rerun()

    with col_R:
        st.markdown("### ☀️ Solar Metrics")
        # These metrics can be updated by the person working on the solar logic
        st.metric("Est. Annual Yield", "0.0 kWh")
        st.metric("Panel Capacity", "0 panels")
        st.metric("Avg. Irradiance", "0 W/m²")