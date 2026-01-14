import numpy as np
from PIL import Image

# Load one of your "black" masks
check_mask = Image.open("data/masks/satellite_tile_0.png")
mask_array = np.array(check_mask)

# Print the unique values found in the image
unique_values = np.unique(mask_array)
print(f"Unique pixel values in the mask: {unique_values}")

if len(unique_values) > 1:
    print("✅ Success: The data is there! The AI will see the different classes.")
else:
    print("❌ Warning: The mask is truly empty (only 0s found).")