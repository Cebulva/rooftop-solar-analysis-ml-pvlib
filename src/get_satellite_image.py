import os
import math
import time
from io import BytesIO
from PIL import Image

import numpy as np
import requests

import torch

def get_aerial_image_tensor(lat=53.5625, lon=9.9630, zoom=18, size=400, device=None, show=True):
    """
    Download a 400x400 aerial image of a residential area and return as a PyTorch tensor.
    Robust version: retries, zoom fallback, Matplotlib display, auto device selection.

    Args:
        lat, lon : float, center coordinates
        zoom     : int, tile zoom level (17-19 recommended)
        size     : int, final image size (size x size)
        device   : torch device (None auto-selects mps/cuda/cpu)
        show     : bool, whether to display the image

    Returns:
        torch.Tensor of shape (1, 3, size, size), normalized 0-1
        Returns None if tiles cannot be downloaded
    """

    # Select device only when function is called
    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    # Convert lat/lon to tile numbers
    def latlon_to_tile(lat, lon, zoom):
        n = 2 ** zoom
        xtile = (lon + 180.0) / 360.0 * n
        ytile = (1.0 - math.log(math.tan(math.radians(lat)) +
                                1 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n
        return int(math.floor(xtile)), int(math.floor(ytile))

    x, y = latlon_to_tile(lat, lon, zoom)

    # 2x2 tile coordinates
    tiles = [(x, y), (x+1, y), (x, y+1), (x+1, y+1)]
    images = []

    # Download tiles with retries
    for tx, ty in tiles:
        url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{ty}/{tx}"
        success = False
        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=3)
                resp.raise_for_status()
                img_tile = Image.open(BytesIO(resp.content)).convert('RGB')
                images.append(np.array(img_tile))
                success = True
                break
            except requests.RequestException as e:
                print(f"Attempt {attempt+1} failed for tile {tx},{ty}: {e}")
                time.sleep(0.5)
        if not success:
            if zoom > 17:
                print(f"Falling back to zoom {zoom-1} for tile {tx},{ty}")
                return get_aerial_image_tensor(lat, lon, zoom=zoom-1, size=size, device=device, show=show)
            else:
                print(f"Cannot download tile {tx},{ty}. Returning None.")
                return None

    # Combine tiles into 2x2 grid
    top = np.concatenate([images[0], images[1]], axis=1)
    bottom = np.concatenate([images[2], images[3]], axis=1)
    full_img = np.concatenate([top, bottom], axis=0)

    # Center crop
    start_h = (full_img.shape[0] - size) // 2
    start_w = (full_img.shape[1] - size) // 2
    img_cropped = full_img[start_h:start_h+size, start_w:start_w+size]

   
    # Normalize and convert to tensor
    img_norm = img_cropped / 255.0
    tensor = torch.tensor(img_norm.transpose(2,0,1), dtype=torch.float32).unsqueeze(0).to(device)

    return tensor