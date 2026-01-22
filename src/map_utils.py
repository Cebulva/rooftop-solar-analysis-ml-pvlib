import math
import time
import torch
import requests
import numpy as np
from io import BytesIO
from PIL import Image

def get_aerial_image_tensor(lat, lon, zoom=19, size=400, device=None):
    """
    Downloads a 3x3 tile grid at Zoom 19 for perfect centering.
    Applies Lanczos supersampling to improve visual quality and reduce pixelation.
    """
    
    # 1. Auto-select device
    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    # 2. Mercator Projection Math
    def latlon_to_tile_fraction(lat, lon, zoom):
        n = 2 ** zoom
        x_frac = (lon + 180.0) / 360.0 * n
        lat_rad = math.radians(lat)
        # Standard Web Mercator formula
        y_frac = (1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n
        return x_frac, y_frac

    xf, yf = latlon_to_tile_fraction(lat, lon, zoom)
    
    # Calculate top-left tile for a 3x3 grid (Target is in the middle tile)
    x_base = int(math.floor(xf - 1.0))
    y_base = int(math.floor(yf - 1.0))

    # 3. Download the 3x3 Grid (9 tiles total)
    # This creates a 768x768 pixel canvas
    rows = []
    for ty in range(y_base, y_base + 3):
        cols = []
        for tx in range(x_base, x_base + 3):
            url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{ty}/{tx}"
            success = False
            for attempt in range(3):
                try:
                    resp = requests.get(url, timeout=5)
                    resp.raise_for_status()
                    img_tile = Image.open(BytesIO(resp.content)).convert('RGB')
                    cols.append(np.array(img_tile))
                    success = True
                    break
                except Exception as e:
                    print(f"Retry {attempt+1} for tile {tx},{ty}: {e}")
                    time.sleep(0.5)
            
            if not success:
                print(f"Failed to download tile {tx},{ty}.")
                return None
        rows.append(np.concatenate(cols, axis=1))
    
    full_img_np = np.concatenate(rows, axis=0)
    full_img_pil = Image.fromarray(full_img_np)

    # 4. Precise Sub-pixel Centering
    # Find the target pixel on the 768x768 canvas
    pixel_x = (xf - x_base) * 256
    pixel_y = (yf - y_base) * 256

    # 5. Supersampling Crop
    # We crop a slightly larger area (size + 4 pixels) then downsample.
    # This acts as an anti-aliasing filter to remove jagged "pixelated" edges.
    buffer = 2 
    left = pixel_x - (size // 2) - buffer
    top = pixel_y - (size // 2) - buffer
    right = pixel_x + (size // 2) + buffer
    bottom = pixel_y + (size // 2) + buffer
    
    # Perform the crop
    img_cropped = full_img_pil.crop((left, top, right, bottom))
    
    # Resample back to original 'size' using LANCZOS for maximum sharpness
    img_final = img_cropped.resize((size, size), resample=Image.Resampling.LANCZOS)

    # 6. Tensor Conversion And Normalization
    img_norm = np.array(img_final) / 255.0
    # Reshape from (H, W, C) to (C, H, W) for PyTorch
    tensor = torch.tensor(img_norm.transpose(2,0,1), dtype=torch.float32).unsqueeze(0).to(device)

    return tensor