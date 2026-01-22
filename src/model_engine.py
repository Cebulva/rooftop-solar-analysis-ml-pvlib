# ========================================
# Hybrid OSM + ML Roof Pipeline (ENGINE)
# Grayscale mask output
# ========================================

import torch
import requests
import numpy as np
import segmentation_models_pytorch as smp
import pyproj
import streamlit as st

from io import BytesIO
from PIL import Image
from shapely.geometry import Polygon, Point
from rasterio.features import rasterize
from rasterio.transform import from_origin


# -------------------------------
# Device
# -------------------------------
def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# -------------------------------
# Load ML model (cached)
# -------------------------------
@st.cache_resource
def load_roof_model(model_path: str):
    device = get_device()

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


# -------------------------------
# Aerial image loader (zoom 19)
# -------------------------------
def get_aerial_image_tensor(
    lat: float,
    lon: float,
    size: int = 400,
    size_m: float = 120,
    imagenet_norm: bool = True,
):
    device = get_device()
    headers = {"User-Agent": "Mozilla/5.0"}

    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", "EPSG:3857", always_xy=True
    )

    x, y = transformer.transform(lon, lat)
    half = size_m / 2
    minx, miny, maxx, maxy = x - half, y - half, x + half, y + half

    url = (
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/export?"
        f"bbox={minx},{miny},{maxx},{maxy}"
        f"&bboxSR=3857&size={size},{size}&format=png&f=image"
    )

    try:
        resp = requests.get(url, timeout=10, headers=headers)
        resp.raise_for_status()

        image_raw = (
            np.array(Image.open(BytesIO(resp.content)).convert("RGB"))
            .astype(np.float32) / 255.0
        )

        tensor = (
            torch.tensor(image_raw.transpose(2, 0, 1))
            .unsqueeze(0)
            .to(device)
        )

        if imagenet_norm:
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            tensor = (tensor - mean) / std

        return tensor, image_raw

    except Exception:
        return None, None


# -------------------------------
# OSM footprint mask
# -------------------------------
def get_osm_mask(
    lat: float,
    lon: float,
    img_shape=(400, 400),
    size_m: float = 120,
):
    h, w = img_shape

    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", "EPSG:3857", always_xy=True
    )

    x_center, y_center = transformer.transform(lon, lat)
    half = size_m / 2
    minx, miny, maxx, maxy = (
        x_center - half,
        y_center - half,
        x_center + half,
        y_center + half,
    )

    query = f"""
    [out:json][timeout:15];
    way["building"]({lat-0.001},{lon-0.001},{lat+0.001},{lon+0.001});
    out geom;
    """

    try:
        r = requests.get(
            "https://overpass.kumi.systems/api/interpreter",
            params={"data": query},
            timeout=5,
        )
        r.raise_for_status()

        data = r.json()
        buildings = data.get("elements", [])

        polys = []
        center_pt = Point(x_center, y_center)

        for elem in buildings:
            if "geometry" not in elem or len(elem["geometry"]) < 3:
                continue

            coords = [
                transformer.transform(n["lon"], n["lat"])
                for n in elem["geometry"]
            ]

            poly = Polygon(coords)
            if poly.distance(center_pt) < 60:
                polys.append(poly)

        if not polys:
            return None

        x_res = (maxx - minx) / w
        y_res = (maxy - miny) / h
        transform = from_origin(minx, maxy, x_res, y_res)

        mask = rasterize(
            [(p, 1) for p in polys],
            out_shape=(h, w),
            transform=transform,
            fill=0,
            all_touched=True,
        )

        return (mask.astype(np.uint8) * 255)

    except Exception:
        return None


# -------------------------------
# Full pipeline (OSM → ML)
# -------------------------------
def run_roof_pipeline(
    model,
    lat: float,
    lon: float,
    ml_threshold: float = 0.3,
):
    """
    Returns:
        gray_mask: uint8 (H, W), values {0,255}
        source: "osm" | "ml" | "failed"
        image_raw: float32 RGB image
    """
    img_tensor, img_raw = get_aerial_image_tensor(lat, lon)
    if img_tensor is None:
        return None, "failed", None

    osm_mask = get_osm_mask(lat, lon, img_shape=img_raw.shape[:2])
    if osm_mask is not None:
        return osm_mask, "osm", img_raw

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits)
        ml_mask = probs.squeeze().cpu().numpy() > ml_threshold

    gray_mask = (ml_mask.astype(np.uint8) * 255)
    return gray_mask, "ml", img_raw
