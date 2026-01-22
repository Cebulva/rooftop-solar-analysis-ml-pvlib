import streamlit as st
import cv2
import numpy as np
import folium
from streamlit_folium import st_folium
import ui_components as ui

def show(lat, lon, get_model):
    """
    Stage 1: Location selection and building confirmation.
    - First, user selects location on map
    - Then AI analyzes the building
    """

    # Check if location has been confirmed for analysis
    if "location_confirmed" not in st.session_state.data:
        # --- LOCATION SELECTION MODE ---
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
                overlay=False,
                control=True
            ).add_to(m)

            # Add marker if position selected
            if st.session_state["selected_pos"]:
                pin_lat, pin_lon = st.session_state["selected_pos"]
                folium.Marker(
                    [pin_lat, pin_lon],
                    popup="Selected Building",
                    icon=folium.Icon(color="red", icon="home"),
                    draggable=False
                ).add_to(m)

            # Render map
            output = st_folium(m, width=1000, height=600)

            # Handle map clicks
            if output and output.get("last_clicked"):
                new_lat = output["last_clicked"]["lat"]
                new_lng = output["last_clicked"]["lng"]

                current_pos = st.session_state["selected_pos"]

                # Only update if click is new/different
                if current_pos is None or (new_lat != current_pos[0] or new_lng != current_pos[1]):
                    st.session_state["selected_pos"] = (new_lat, new_lng)
                    st.rerun()

            # Show confirmation button only if a position is selected
            if st.session_state["selected_pos"]:
                st.divider()
                sel_lat, sel_lon = st.session_state["selected_pos"]

                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Latitude", f"{sel_lat:.6f}")
                with c2:
                    st.metric("Longitude", f"{sel_lon:.6f}")

                gap = (ui.DISPLAY_WIDTH - ui.BUTTON_SIZE_VERIFY) / 2
                _, sub_col, _ = st.columns([gap, ui.BUTTON_SIZE_VERIFY, gap])

                with sub_col:
                    if st.button("🚀 Analyze Building", use_container_width=True, type="primary"):
                        st.session_state.data["location_confirmed"] = True
                        st.session_state.data["confirmed_lat"] = sel_lat
                        st.session_state.data["confirmed_lon"] = sel_lon
                        st.rerun()
            else:
                st.info("👆 Click on the map to select your building location.")

    else:
        # --- AI ANALYSIS MODE (Original code) ---
        st.subheader("Step 1: Confirm Target Building")

        # Use confirmed coordinates
        lat = st.session_state.data["confirmed_lat"]
        lon = st.session_state.data["confirmed_lon"]
        coord_key = f"{lat}_{lon}"

        # Data fetching and AI Inference (cached for speed)
        if st.session_state.data.get("key") != coord_key:
            with st.spinner("Fetching stable imagery and running AI..."):
                # --- HYBRID OSM + ML MASK PIPELINE ---
                from src.model_engine import run_roof_pipeline

                # Use global MODEL_PATH from ui_components
                model = get_model(ui.MODEL_PATH)

                # gray_mask: (H, W), source: str, full_img: (H, W, 3)
                gray_mask, source, full_img = run_roof_pipeline(model, lat, lon)

                from src.image_processing import filter_non_roof_objects
                clean_mask = filter_non_roof_objects(gray_mask)

                # --- DATA TYPE STANDARDIZATION ---
                # Ensure base image is 0-255 uint8 to prevent Streamlit float crashes
                if full_img.dtype != np.uint8:
                    if full_img.max() <= 1.0:
                        full_img = (full_img * 255).astype(np.uint8)
                    else:
                        full_img = full_img.astype(np.uint8)

                # Create the initial cyan preview overlay
                overlay = full_img.copy()
                mask_layer = overlay.copy()

                # RGB Cyan: (0, 255, 255)
                mask_layer[clean_mask > 0] = [0, 255, 255]

                # Blend and force uint8 result
                overlay = cv2.addWeighted(overlay, 0.6, mask_layer, 0.4, 0).astype(np.uint8)

                st.session_state.data.update({
                    "key": coord_key,
                    "full_img": full_img,
                    "full_mask": clean_mask,
                    "preview_overlay": overlay,
                    "mask_source": source
                })

        # Standardized Layout for UI Stability
        col_L, col_center, col_R = st.columns([1, 4, 1.5])

        with col_center:
            # We display the overlay created above
            st.image(
                st.session_state.data["preview_overlay"],
                width=ui.DISPLAY_WIDTH,
                caption=f"Source: {st.session_state.data.get('mask_source', 'Unknown')}"
            )

            # Centering the button beneath the image to match ui.BUTTON_SIZE_VERIFY
            gap_ratio = (ui.DISPLAY_WIDTH - ui.BUTTON_SIZE_VERIFY) / 2
            _, sub_col, _ = st.columns([gap_ratio, ui.BUTTON_SIZE_VERIFY, gap_ratio])

            with sub_col:
                if st.button("✅ Confirm Building", use_container_width=True, type="primary"):
                    st.session_state.step = 2
                    st.rerun()
