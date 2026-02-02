#!/usr/bin/env python3
import os
import math
import time
import requests
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image
import torch
import segmentation_models_pytorch as smp

# -----------------------------
# Device
# -----------------------------
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# -----------------------------
# Model architecture
# -----------------------------
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None
).to(device)

# -----------------------------
# Load model weights
# -----------------------------
url = "https://github.com/Cebulva/rooftop-solar-analysis-ml-pvlib/raw/main/models/custom_ds_roof_model.pth"

response = requests.get(url)
response.raise_for_status()
checkpoint = torch.load(BytesIO(response.content), map_location=device)
model.load_state_dict(checkpoint)
model.eval()


# -----------------------------
# Aerial image download function
# -----------------------------
def get_aerial_image_tensor(lat, lon, zoom=18, size=400, device=None, imagenet_norm=False):
    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    def latlon_to_tile(lat, lon, zoom):
        n = 2 ** zoom
        xtile = (lon + 180.0) / 360.0 * n
        ytile = (1.0 - math.log(math.tan(math.radians(lat)) +
                                1 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n
        return int(math.floor(xtile)), int(math.floor(ytile))

    x, y = latlon_to_tile(lat, lon, zoom)
    tiles = [(x, y), (x+1, y), (x, y+1), (x+1, y+1)]
    images = []

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
            except requests.RequestException:
                time.sleep(0.5)
        if not success:
            if zoom > 17:
                return get_aerial_image_tensor(lat, lon, zoom=zoom-1, size=size,
                                               device=device, imagenet_norm=imagenet_norm)
            else:
                return None, None

    top = np.concatenate([images[0], images[1]], axis=1)
    bottom = np.concatenate([images[2], images[3]], axis=1)
    full_img = np.concatenate([top, bottom], axis=0)

    start_h = (full_img.shape[0] - size) // 2
    start_w = (full_img.shape[1] - size) // 2
    img_cropped = full_img[start_h:start_h+size, start_w:start_w+size]

    img = img_cropped.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    if imagenet_norm:
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std

    return tensor, img_cropped  # Return tensor and raw image

# -----------------------------
# Process coordinates
# -----------------------------
def process_coordinates(coords, output_dir="output"):
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/masks", exist_ok=True)

    for i, (lat, lon) in enumerate(coords, start=1):
        print(f"\n=== Point {i}: lat={lat}, lon={lon} ===")
        img_tensor, img_raw = get_aerial_image_tensor(lat, lon, zoom=20, size=400, device=device, imagenet_norm=True)
        if img_tensor is None:
            print("Download failed, skipping this point.")
            continue

        with torch.no_grad():
            logits = model(img_tensor)
            probs  = torch.sigmoid(logits)
            mask   = (probs.squeeze().cpu().numpy() * 255).astype(np.uint8)

        # Save raw image
        img_pil = Image.fromarray((img_raw * 255).astype(np.uint8))
        img_pil.save(f"{output_dir}/images/point_{i}.png")

        # Save predicted mask
        mask_pil = Image.fromarray(mask)
        mask_pil.save(f"{output_dir}/masks/point_{i}_mask.png")

        print(f"Saved {output_dir}/images/point_{i}.png and {output_dir}/masks/point_{i}_mask.png")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Example: load coordinates from CSV
    # CSV format: lat,lon
    csv_file = "data/coordinates_for_inference.csv"
    if not os.path.exists(csv_file):
        print(f"CSV file '{csv_file}' not found!")
        exit(1)

    df = pd.read_csv(csv_file)
    coords = list(zip(df['lat'], df['lon']))

    process_coordinates(coords, output_dir="output")
