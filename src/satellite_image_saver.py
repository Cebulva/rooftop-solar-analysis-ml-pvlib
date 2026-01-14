import os
import math
from PIL import Image
from io import BytesIO

import numpy as np
import requests

import torch
import pandas as pd


import sys
from pathlib import Path

csv_path = "data/coordinates.csv"

# Adds the project root to the path so 'src' can be found
root = str(Path(__file__).resolve().parent.parent)
if root not in sys.path:
    sys.path.append(root)

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

# # --- EXECUTION Single IMG ---
# save_aerial_data(
#     lat=53.634869813414404, 
#     lon=10.090849074245684, 
#     zoom=19, 
#     size=400,
#     filename="tile_1.png"
# )

def download_batch_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    
    for index, row in df.iterrows():
        lat = row['lat']
        lon = row['lon']
        
        # Automatically generate a unique filename using the row index
        file_name = f"satellite_tile_{index}.png"
        
        save_aerial_data(lat=lat, lon=lon, zoom=19, size=400, filename=file_name)

# --- EXECUTION list ---
download_batch_from_csv(csv_path)