import cv2
import numpy as np
from matplotlib import pyplot as plt

### Mask cleaning - Morphology ###

# load img in greyscale
img = cv2.imread('images/test_satellite.png', cv2.IMREAD_GRAYSCALE)
# Create a mask of the img
_, mask = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)

# Define a square kernal (brush) - tweak size if dialation isn't clean
kernal = np.ones((8,8), np.uint8)

# Different methods:
dialation = cv2.dilate(mask, kernal, iterations=2)

erosion = cv2.erode(mask, kernal, iterations=1)

opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)

# performs Dilation followed by Erosion
closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal, iterations=2)

mg = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernal)

### Smoothing the Edges ###

blurred = cv2.GaussianBlur(closed, (3,3), 0)
_, final_mask = cv2.threshold(blurred, 127,255, cv2.THRESH_BINARY)

### Visualization Part ###

titles = ['image', 'mask', 'closed', 'final_mask']
images = [img, mask, closed, final_mask]

for i in range(4):
    plt.subplot(2, 2, i+1), plt.imshow(images [i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()

