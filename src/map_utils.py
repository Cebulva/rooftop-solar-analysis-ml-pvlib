import math
import time
import torch
import requests
import numpy as np
from io import BytesIO
from PIL import Image

def get_aerial_image_tensor(lat, lon, zoom=19, size=400, device=None):
    """
    Downloads and precisely centers an aerial image on a lat/lon coordinate.
    Uses a 2x2 tile grid to allow for pixel-perfect cropping across tile boundaries.
    """
    
    # 1. Auto-select device
    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    # 2. Mercator Projection Math (Sub-tile precision)
    def latlon_to_tile_fraction(lat, lon, zoom):
        n = 2 ** zoom
        x_frac = (lon + 180.0) / 360.0 * n
        lat_rad = math.radians(lat)
        y_frac = (1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n
        return x_frac, y_frac

    # Get fractional coordinates
    xf, yf = latlon_to_tile_fraction(lat, lon, zoom)
    
    # Calculate top-left tile of the 2x2 grid
    # We use floor(coord - 0.5) so the target is roughly in the center of 4 tiles
    x_base = int(math.floor(xf - 0.5))
    y_base = int(math.floor(yf - 0.5))

    # 3. Download the 2x2 Grid
    images = []
    # Grid order: (TL, TR, BL, BR)
    for ty in [y_base, y_base + 1]:
        for tx in [x_base, x_base + 1]:
            url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{ty}/{tx}"
            success = False
            for attempt in range(3):
                try:
                    resp = requests.get(url, timeout=5)
                    resp.raise_for_status()
                    img_tile = Image.open(BytesIO(resp.content)).convert('RGB')
                    images.append(np.array(img_tile))
                    success = True
                    break
                except Exception as e:
                    print(f"Retry {attempt+1} for tile {tx},{ty}: {e}")
                    time.sleep(0.5)
            
            if not success:
                print(f"Failed to download tile {tx},{ty}. Process stopped.")
                return None

    # 4. Stitch 2x2 grid into a 512x512 canvas
    top = np.concatenate([images[0], images[1]], axis=1)
    bottom = np.concatenate([images[2], images[3]], axis=1)
    full_img = np.concatenate([top, bottom], axis=0)

    # 5. Precise Pixel Centering
    # Tile size is 256. We find the pixel of our target relative to the grid start.
    pixel_x = int((xf - x_base) * 256)
    pixel_y = int((yf - y_base) * 256)

    # Calculate crop boundaries
    half_size = size // 2
    start_x = pixel_x - half_size
    start_y = pixel_y - half_size
    
    # Boundary Safety (Ensure we don't crop outside the 512px stitched image)
    start_x = max(0, min(start_x, 512 - size))
    start_y = max(0, min(start_y, 512 - size))

    img_cropped = full_img[start_y:start_y+size, start_x:start_x+size]

    # 6. Tensor Conversion And Normalization
    img_norm = img_cropped / 255.0
    # Change shape from (H, W, C) to (C, H, W)
    tensor = torch.tensor(img_norm.transpose(2,0,1), dtype=torch.float32).unsqueeze(0).to(device)

    return tensor