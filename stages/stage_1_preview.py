import streamlit as st
import folium
from streamlit_folium import st_folium
import ui_components as ui
from src.inquiry_manager import save_inquiry

def show():
    """
    Stage 1: Pure Location Selection.
    No AI mask is drawn here to keep the UI fast and clean.
    """
    st.subheader("Step 1: Locate Your Property")

    # Get current map center
    if st.session_state["selected_pos"]:
        center_lat, center_lon = st.session_state["selected_pos"]
    else:
        center_lat, center_lon = st.session_state["map_center"]

    col_L, col_center, col_R = st.columns([1, 4, 1.5])

    with col_center:
        st.markdown("Click on the **center of your roof** to select the building.")

        # Create map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=19)

        # Satellite Tiles
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Esri Satellite',
            overlay=False
        ).add_to(m)

        # Add marker if position selected
        if st.session_state["selected_pos"]:
            pin_lat, pin_lon = st.session_state["selected_pos"]
            folium.Marker(
                [pin_lat, pin_lon],
                icon=folium.Icon(color="red", icon="home")
            ).add_to(m)

        # Render map
        output = st_folium(m, width=1000, height=600, key="main_map")

        # Map navigation tip
        st.caption("💡 **Tip:** Zoom out to see wider satellite view, zoom in for detailed street map.")

        # Handle map clicks
        if output and output.get("last_clicked"):
            new_lat = output["last_clicked"]["lat"]
            new_lng = output["last_clicked"]["lng"]
            st.session_state["selected_pos"] = (new_lat, new_lng)
            st.rerun()

        # Confirmation Button
        if st.session_state["selected_pos"]:
            sel_lat, sel_lon = st.session_state["selected_pos"]
            
            st.divider()
            gap = (ui.DISPLAY_WIDTH - ui.BUTTON_SIZE_VERIFY) / 2
            _, sub_col, _ = st.columns([gap, ui.BUTTON_SIZE_VERIFY, gap])

            with sub_col:
                if st.button("🚀 Analyze Building", use_container_width=True, type="primary"):
                    # Save coordinates and move to Step 2 where inference actually happens
                    st.session_state.data["confirmed_lat"] = sel_lat
                    st.session_state.data["confirmed_lon"] = sel_lon

                    # Auto-save inquiry
                    if st.session_state.get("inquiry_id"):
                        save_inquiry(
                            st.session_state.inquiry_id,
                            st.session_state.data,
                            step=2,
                            sub_step="verify"
                        )

                    st.session_state.step = 2
                    st.rerun()
        else:
            st.info("👆 Click on the map to select your building location.")