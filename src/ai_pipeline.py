import numpy as np
from src.model_engine import run_roof_pipeline
from src.image_processing import filter_non_roof_objects, get_zoom_crop
from src.geometry_utils import mask_to_polygon

def get_initial_rooftop_data(model, lat, lon):
    """The heavy lifting: fetches data and returns raw geometry."""
    gray_mask, source, full_img = run_roof_pipeline(model, lat, lon)
    
    # --- SAFETY CHECK ---
    if full_img is None:
        raise ValueError(f"Failed to fetch satellite imagery for coordinates: {lat}, {lon}. Check your internet connection or API limits.")

    if full_img.dtype != np.uint8:
        full_img = (full_img * 255 if full_img.max() <= 1.0 else full_img).astype(np.uint8)

    clean_mask = filter_non_roof_objects(gray_mask)
    z_img, z_mask, offsets = get_zoom_crop(full_img, clean_mask)
    poly_points = mask_to_polygon(z_mask)
    
    return {
        "full_img": full_img,
        "zoom_img": z_img,
        "initial_poly": poly_points,
        "offsets": offsets,
        "source": source
    }