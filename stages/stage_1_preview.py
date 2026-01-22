import streamlit as st
import cv2
import numpy as np
import ui_components as ui

def show(lat, lon, get_model):
    st.subheader("Step 1: Confirm Target Building")
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
            
            st.session_state.data = {
                "key": coord_key,
                "full_img": full_img,
                "full_mask": clean_mask,
                "preview_overlay": overlay,
                "mask_source": source
            }

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