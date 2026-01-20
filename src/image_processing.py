import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops

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

def refine_and_analyze(raw_mask):
    """
    The Master Function: Cleans the mask, vectorizes the shape, 
    and calculates orientation in one go.
    """
    # 1. Clean & Smooth (Morphology)
    mask_8u = (raw_mask * 255).astype(np.uint8)
    kernel = np.ones((8,8), np.uint8)
    closed = cv2.morphologyEx(mask_8u, cv2.MORPH_CLOSE, kernel, iterations=2)
    blurred = cv2.GaussianBlur(closed, (3,3), 0)
    _, final_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # 2. Vectorization
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0, 0
    
    main_contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(main_contour, True)
    epsilon = 0.02 * perimeter
    simplified_poly = cv2.approxPolyDP(main_contour, epsilon, True)

    # 3. Azimuth Calculation
    label_img = label(final_mask)
    props = regionprops(label_img)
    if not props:
        return simplified_poly, len(simplified_poly), 0.0
    
    main_roof = max(props, key=lambda x: x.area)
    azimuth = (math.degrees(main_roof.orientation) + 90) % 180

    return simplified_poly, azimuth, final_mask

def get_zoom_crop(image, mask, padding=40):
    # Find coordinates of all mask pixels
    coords = cv2.findNonZero(mask.astype(np.uint8))
    x, y, w, h = cv2.boundingRect(coords)

    # Add padding so the roof isn't touching the edge of the screen
    start_x = max(0, x - padding)
    start_y = max(0, y - padding)
    end_x = min(image.shape[1], x + w + padding)
    end_y = min(image.shape[0], y + h + padding)

    # Crop both the satellite image and the mask
    cropped_img = image[start_y:end_y, start_x:end_x]
    cropped_mask = mask[start_y:end_y, start_x:end_x]
    
    return cropped_img, cropped_mask, (start_x, start_y)

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