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