import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops

# ==========================================
# ⚙️ PROCESSING CONFIGURATION (Tweak these)
# ==========================================
CROP_PADDING = 40       # Extra space around the roof (px)
MIN_CROP_SIZE = 220     # Max Zoom Limit (Minimum pixels wide/tall)
MORPH_KERNEL_SIZE = 8   # Size of the "brush" for cleaning the mask
EPSILON_FACTOR = 0.02   # Smoothing factor for polygon simplification
# ==========================================

# ### Mask Refienment - Morphology ###

# raw_mask = cv2.imread('data/images/satellite_tile_0.png', cv2.IMREAD_GRAYSCALE)

# # load img in greyscale
# img = cv2.imread('data/images/satellite_tile_0.png', cv2.IMREAD_GRAYSCALE)
# # Create a mask of the img
# _, mask = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)

# # Define a square kernal (brush) - tweak size if dialation isn't clean
# kernal = np.ones((8,8), np.uint8)

# # Different methods:
# dialation = cv2.dilate(mask, kernal, iterations=2)

# erosion = cv2.erode(mask, kernal, iterations=1)

# opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)

# # performs Dilation followed by Erosion
# closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal, iterations=2)

# mg = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernal)

# ### Smoothing the Edges ###

# blurred = cv2.GaussianBlur(closed, (3,3), 0)
# _, final_mask = cv2.threshold(blurred, 127,255, cv2.THRESH_BINARY)

# ### Visualization Part ###

# # titles = ['image', 'mask', 'closed', 'final_mask']
# # images = [img, mask, closed, final_mask]

# # for i in range(4):
# #     plt.subplot(2, 2, i+1), plt.imshow(images [i], 'gray')
# #     plt.title(titles[i])
# #     plt.xticks([]),plt.yticks([])

# # plt.show()

# def clean_roof_mask(raw_mask):
#     """
#     Binarize the input image of the satellite.
#     Remove noize and fill gaps using morphology.
#     Smooth edges using gaussian refinement.
#     Returns: (final_mask)
#     """
#     # Threshold
#     _, binary =cv2.threshold(raw_mask, 220, 255, cv2.THRESH_BINARY_INV)

#     # Define a square kernal (brush) - tweak size if dialation isn't clean
#     kernal = np.ones((8,8), np.uint8)

#     # performs Dilation followed by Erosion
#     closed = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernal, iterations=2)

#     #Smoothing the Edges #
#     blurred = cv2.GaussianBlur(closed, (3,3), 0)
#     _, final_mask = cv2.threshold(blurred, 127,255, cv2.THRESH_BINARY)

#     return final_mask

# ### Geometric Vectorization ###

# ## Identify external contours ##

# # cv2.RETR_EXTERNAL ignores holes inside of the mask and only grabs the outermost boundary of the roof
# # cv2.CHAIN_APPROX_SIMPLE stores endpoints instead of every pixel along a straight line 
# contours, _ = cv2.findContours(final_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# ## Select the Primary Roof ##

# # Sort by area descending and pick the largest contour (primary roof)
# main_roof_countour = max(contours, key=cv2.contourArea)

# ## Simplify the Geometry ##

# perimeter = cv2.arcLength(main_roof_countour, True)

# epsilon = 0.01 * perimeter

# simplified_roof = cv2.approxPolyDP(main_roof_countour, epsilon, True)

# ## Review the Corner Count
# num_corners = len(simplified_roof)
# print(f"Number of corners detected: {num_corners}")

# if num_corners == 4:
#     print("Shape: Simple Rectangle")
# elif num_corners == 6:
#     print("Shape: L-Shaped or T-Shaped roof")
# elif num_corners > 8:
#     print("Shape: Complex Multi-level roof")

# def vectorize_roof(final_mask, epsilon_factor=0.1):
#     """
#     Converts a binary mask into a simplified geometric polygon.
#     Returns: (simplified_roof, corner_count)
#     """
#     ## Identify external contours ##

#     # cv2.RETR_EXTERNAL ignores holes inside of the mask and only grabs the outermost boundary of the roof
#     # cv2.CHAIN_APPROX_SIMPLE stores endpoints instead of every pixel along a straight line 
#     contours, _ = cv2.findContours(final_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     if not contours:
#         return None, 0
    
#     ## Sort by area descending and pick the largest contour (primary roof)
#     main_roof_countour = max(contours, key=cv2.contourArea)

#     ## Simplify the Geometry ##
#     perimeter = cv2.arcLength(main_roof_countour, True)
#     epsilon = epsilon_factor * perimeter
#     simplified_roof = cv2.approxPolyDP(main_roof_countour, epsilon, True)

#     ## Extract Corner Count
#     corner_count = len(simplified_roof)

#     return simplified_roof, corner_count


# """
# Calculates the compass orientation of the roof.
# Returns: azimuth_degrees (float)
# """

# ## Label the components
# label_img = label(final_mask)
# # regionprops() calculates everything from the center of mass (centroid) to the orientation.
# props = regionprops(label_img)

# # Sort through all labeled objects and pick the one with the largest area.
# main_roof = max(props, key=lambda x: x.area)

# # Finds the "Major Axis" (the longest line you could draw through the roof). 
# # The orientation is the angle between that Major Axis and the vertical axis of the image.
# angle_rad = main_roof.orientation

# # Convert Radians to Degrees
# angle_deg = math.degrees(angle_rad)

# """
# 1. Converts radians (0.78) to degrees (45°)
# 2. + 90: Shifts the coordinate system so that "Up" (North) 
# aligns with the vertical axis of your satellite image.
# 3. % 180: Because a roof is a line, the major axis points in two directions 
# (e.g., North AND South). For now, we find the line of the roof.
# """
# azimuth = (math.degrees(angle_rad) + 90) %180

# def calculate_roof_azimuth(final_mask):
#     """
#     Calculates the compass orientation of the roof.
#     Returns: azimuth_degrees (float)
#     """

#     ## Label the components
#     label_img = label(final_mask)
#     # regionprops() calculates everything from the center of mass (centroid) to the orientation.
#     props = regionprops(label_img)

#     if not props:
#             return 0.0
    
#     # Sort through all labeled objects and pick the one with the largest area.
#     main_roof = max(props, key=lambda x: x.area)

#     # Finds the "Major Axis" (the longest line you could draw through the roof). 
#     # The orientation is the angle between that Major Axis and the vertical axis of the image.
#     angle_rad = main_roof.orientation

#     # Convert Radians to Degrees
#     angle_deg = math.degrees(angle_rad)

#     """
#     1. Converts radians (0.78) to degrees (45°)
#     2. + 90: Shifts the coordinate system so that "Up" (North) 
#     aligns with the vertical axis of your satellite image.
#     3. % 180: Because a roof is a line, the major axis points in two directions 
#     (e.g., North AND South). For now, we find the line of the roof.
#     """
#     azimuth = (math.degrees(angle_rad) + 90) %180

#     # Note: Because a roof line is symmetrical, this gives the 'axis'.
#     # For solar, we usually assume the panels face 'out' toward the nearest gutter.
#     return azimuth

# print(f"Azimuth: {azimuth}")

def refine_and_analyze(raw_mask, kernel_size=MORPH_KERNEL_SIZE, epsilon=EPSILON_FACTOR):
    """
    Cleans the mask and vectorizes using exposed hyperparameters.
    """
    mask_8u = (raw_mask * 255).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # 1. Clean & Smooth
    closed = cv2.morphologyEx(mask_8u, cv2.MORPH_CLOSE, kernel, iterations=2)
    blurred = cv2.GaussianBlur(closed, (3,3), 0)
    _, final_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # 2. Vectorization
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None, 0, 0
    
    main_contour = max(contours, key=cv2.contourArea)
    simplified_poly = cv2.approxPolyDP(main_contour, epsilon * cv2.arcLength(main_contour, True), True)

    # 3. Azimuth
    label_img = label(final_mask)
    props = regionprops(label_img)
    azimuth = (math.degrees(max(props, key=lambda x: x.area).orientation) + 90) % 180 if props else 0.0

    return simplified_poly, azimuth, final_mask

def get_zoom_crop(image, mask, padding=CROP_PADDING, min_size=MIN_CROP_SIZE):
    """
    Crops the image to the roof with a safety floor for zoom levels.
    """
    coords = cv2.findNonZero(mask.astype(np.uint8))
    if coords is None:
        return image, mask, (0, 0)

    x, y, w, h = cv2.boundingRect(coords)

    # Calculate target boundaries
    start_x, end_x = x - padding, x + w + padding
    start_y, end_y = y - padding, y + h + padding

    # Enforce Max Zoom Limit (Safety Floor)
    # If the window is smaller than min_size, expand it symmetrically
    curr_w, curr_h = end_x - start_x, end_y - start_y
    
    if curr_w < min_size:
        pad_w = (min_size - curr_w) // 2
        start_x -= pad_w
        end_x += pad_w
        
    if curr_h < min_size:
        pad_h = (min_size - curr_h) // 2
        start_y -= pad_h
        end_y += pad_h

    # Prevent out-of-bounds
    start_x = max(0, start_x)
    start_y = max(0, start_y)
    end_x = min(image.shape[1], end_x)
    end_y = min(image.shape[0], end_y)

    return image[start_y:end_y, start_x:end_x], mask[start_y:end_y, start_x:end_x], (start_x, start_y)

def select_center_component(mask_8u):
    """Select the connected component closest to image center, preserving original pixels."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_8u, connectivity=8)

    if num_labels <= 1:
        return mask_8u

    img_center = np.array([mask_8u.shape[1] / 2, mask_8u.shape[0] / 2])

    best_label = None
    min_dist = float('inf')

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 300:
            continue
        dist = np.linalg.norm(centroids[i] - img_center)
        if dist < min_dist:
            min_dist = dist
            best_label = i

    if best_label is None:
        return np.zeros_like(mask_8u)

    return ((labels == best_label).astype(np.uint8) * 255)

def filter_non_roof_objects(mask_8u):
    contours, _ = cv2.findContours(mask_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask_8u)

    # 1. Identify the center of your 400x400 image
    img_center = np.array([200, 200])
    
    best_cnt = None
    min_dist = float('inf')

    print(f"\n--- 📍 Centroid Proximity Log ---")

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 300: # Ignore tiny noise
            continue

        # 2. Calculate the center of this specific object (Centroid)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroid = np.array([cX, cY])
            
            # 3. Calculate distance from image center to object center
            dist = np.linalg.norm(centroid - img_center)
            
            print(f"Object {i}: Area {int(area)} | Distance from center: {dist:.1f}px")

            # 4. We keep the one closest to the target coordinates
            if dist < min_dist:
                min_dist = dist
                best_cnt = cnt

    # 5. Create a mask containing ONLY the best object
    cleaned_mask = np.zeros_like(mask_8u)
    if best_cnt is not None:
        print(f"🏆 Selected Object closest to coordinates.")
        cv2.drawContours(cleaned_mask, [best_cnt], -1, 255, -1)
    
    return cleaned_mask

def format_poly_for_canvas(poly_points):
    """Converts [[x,y], [x,y]] points into a Streamlit Canvas path string."""
    if poly_points is None or len(poly_points) == 0:
        return None
    
    path_data = []
    for i, pt in enumerate(poly_points):
        # 'M' starts the path, 'L' draws lines to subsequent points
        command = "M" if i == 0 else "L"
        path_data.append([command, float(pt[0]), float(pt[1])])
    
    path_data.append(["Z"]) # 'Z' closes the polygon
    
    # Return in the specific dictionary format the component expects
    return {"objects": [{"type": "path", "path": path_data, "stroke": "#00FFFF", "fill": "rgba(0, 255, 255, 0.3)"}]}