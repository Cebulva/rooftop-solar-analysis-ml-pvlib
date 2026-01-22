import cv2
import numpy as np
import math

def mask_to_polygon(mask, epsilon_factor=0.02):
    """
    Converts a binary mask into a simplified polygon.
    epsilon_factor: Controls how much to simplify the shape (lower = more detail).
    """
    # Find the external boundary of the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # Pick the largest object (the primary roof)
    main_contour = max(contours, key=cv2.contourArea)
    
    # Simplify the contour into a clean polygon
    perimeter = cv2.arcLength(main_contour, True)
    approx = cv2.approxPolyDP(main_contour, epsilon_factor * perimeter, True)
    
    # Reshape to a simple list of [x, y] points
    return approx.reshape(-1, 2).tolist()

def calculate_azimuth(polygon, img=None):
    """
    Finds the dominant orientation of the roof. 
    If img is provided, uses brightness to guess the slope direction.
    0° = North, 90° = East, 180° = South, 270° = West.
    """
    if not polygon or len(polygon) < 2:
        return 0.0
    
    # 1. Get the Minimum Rotated Rectangle
    pts = np.array(polygon).astype(np.int32)
    rect = cv2.minAreaRect(pts)
    (x, y), (w, h), angle = rect
    
    # 2. Base Geometric Orientation (align to the long axis)
    azimuth = angle if w > h else angle + 90
    azimuth = azimuth % 360

    # 3. Brightness Heuristic (only if image is provided)
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h_i, w_i = gray.shape
        # Compare top half vs bottom half brightness
        if np.mean(gray[h_i//2:, :]) > np.mean(gray[:h_i//2, :]):
            # Bottom is brighter, likely facing South
            if azimuth < 90 or azimuth > 270: 
                azimuth = (azimuth + 180) % 360
        else:
            # Top is brighter, likely facing North
            if 90 < azimuth < 270: 
                azimuth = (azimuth + 180) % 360
                
    return float(azimuth)

def generate_panel_grid(roof_poly, panel_w=1.75, panel_h=1.05, gsd=0.15, azimuth=0):
    """
    Generates a list of shapely Polygons representing solar panels.
    panel_w/h: in meters
    gsd: meters per pixel
    """
    # 1. Convert panel dimensions to pixels
    pw_px = panel_w / gsd
    ph_px = panel_h / gsd
    
    # 2. Get bounding box of the roof
    minx, miny, maxx, maxy = roof_poly.bounds
    
    panels = []
    # 3. Iterate through the bounding box
    for x in np.arange(minx, maxx, pw_px):
        for y in np.arange(miny, maxy, ph_px):
            # Create a rectangle (panel)
            p = Polygon([(x, y), (x+pw_px, y), (x+pw_px, y+ph_px), (x, y+ph_px)])
            
            # 4. VALIDATION: Check if panel is completely inside the SUNNY area
            if roof_poly.contains(p):
                panels.append(p)
                
    return panels