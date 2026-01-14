import os
import math
from PIL import Image
from io import BytesIO

import numpy as np
import requests

import torch

from src.get_satellite_image import (get_aerial_image_tensor)

def save_aerial_data(lat, lon, zoom, size, filename="tile_1.png"):
    # 1. Ensure the directory exists
    os.makedirs("data/images", exist_ok=True)
    
    # 2. Get the tensor AND the raw image (Modify your function to return both)
    # Note: Using your existing get_aerial_image_tensor function
    img_tensor = get_aerial_image_tensor(lat=lat, lon=lon, zoom=zoom, size=size, show=False)
    
    # 3. Convert Tensor back to a Saveable Image
    # We remove the batch dim, move channels to the end, and scale back to 0-255
    input_img = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    
    # If you applied ImageNet normalization, we must 'un-normalize' to save
    # If you haven't added the mean/std shift yet, just use: img_to_save = (input_img * 255)
    img_to_save = (input_img * 255).astype(np.uint8)
    
    # 4. Save using PIL
    save_path = os.path.join("data/images", filename)
    Image.fromarray(img_to_save).save(save_path)
    print(f"Successfully saved image to: {save_path}")

# --- EXECUTION ---
save_aerial_data(
    lat=53.634869813414404, 
    lon=10.090849074245684, 
    zoom=19, 
    size=400,
    filename="tile_1.png"
)