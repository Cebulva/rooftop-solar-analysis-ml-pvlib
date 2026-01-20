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

def calculate_azimuth(polygon):
    """
    Finds the dominant orientation of the roof. 
    0° = North, 90° = East, 180° = South, 270° = West.
    """
    if not polygon or len(polygon) < 2:
        return 0.0
    
    # Find the Minimum Rotated Rectangle (this handles slanted roofs perfectly)
    pts = np.array(polygon).astype(np.int32)
    rect = cv2.minAreaRect(pts)
    (x, y), (w, h), angle = rect
    
    # Logic to ensure we always get the 'long side' orientation
    if w < h:
        azimuth = angle + 180
    else:
        azimuth = angle + 90
        
    return azimuth % 360