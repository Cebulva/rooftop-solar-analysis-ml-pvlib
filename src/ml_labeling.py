import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pathlib import Path

def load_and_visualize_coco(json_path, img_dir):
    # 1. Load the COCO manager
    # This parses the JSON and builds an internal index of images and polygons.
    coco = COCO(json_path)

    # 2. Pick the first available image ID
    img_ids = coco.getImgIds()
    if not img_ids:
        print("No images found in the JSON file.")
        return None, None
    
    img_id = img_ids[0]
    
    # 3. Retrieve image metadata (filename, width, height)
    img_metadata = coco.loadImgs(img_id)[0]
    file_name = img_metadata['file_name']
    
    # 4. Open the actual satellite image
    img_path = os.path.join(img_dir, file_name)
    if not os.path.exists(img_path):
        print(f"Error: Could not find image file at {img_path}")
        return None, None
        
    image = Image.open(img_path).convert("RGB")
    width, height = image.size

    # 5. Get all annotations (polygons) for this specific image
    ann_ids = coco.getAnnIds(imgIds=img_id)
    annotations = coco.loadAnns(ann_ids)

    # 6. Initialize the Target Mask
    # A 2D array of zeros where each pixel will eventually be 1 (Roof), 2 (Shadow), etc.
    mask = np.zeros((height, width), dtype=np.uint8)

    # 7. Draw Polygons onto the Mask with Safety Filters
    # This prevents the 'IndexError' caused by broken/empty CVAT polygons.
    for ann in annotations:
        # Check A: Does the segmentation data actually exist?
        if 'segmentation' not in ann or not ann['segmentation']:
            continue
            
        # Check B: Is it a valid polygon? (Needs at least 3 points = 6 numbers)
        if isinstance(ann['segmentation'], list):
            if len(ann['segmentation'][0]) < 6:
                print(f"Skipping invalid polygon ID: {ann['id']} (Too few points)")
                continue

        # Try-Except block to catch any hidden pycocotools glitches
        try:
            category_id = ann['category_id']
            # Convert polygon coordinates to a binary 0/1 pixel mask
            pixel_mask = coco.annToMask(ann)
            
            # Apply the ID to the mask. 
            # If a pixel is inside the polygon (pixel_mask == 1), 
            # we set its value to the category_id (e.g., 1 for Roof, 2 for Shadow).
            mask[pixel_mask == 1] = category_id
        except Exception as e:
            print(f"Skipping annotation {ann['id']} due to error: {e}")
            continue

    return image, mask

# --- EXECUTION ---

# Adjust these paths to match your project structure
ANNOTATIONS_FILE = "data/annotations.json"
IMAGES_DIRECTORY = "data/images"

# Run the function
satellite_img, ground_truth_mask = load_and_visualize_coco(ANNOTATIONS_FILE, IMAGES_DIRECTORY)

# Verification and Visualization
if satellite_img is not None:
    print(f"Successfully processed image: {satellite_img.size}")
    
    plt.figure(figsize=(12, 6))

    # Show original satellite image
    plt.subplot(1, 2, 1)
    plt.title("Satellite Input")
    plt.imshow(satellite_img)
    plt.axis('off')

    # Show the generated mask
    # 'cmap=jet' helps see the difference between Roof and Shadow values
    plt.subplot(1, 2, 2)
    plt.title("Generated U-Net Mask")
    plt.imshow(ground_truth_mask, cmap='jet') 
    plt.axis('off')

    plt.show()

def generate_all_masks(json_path, img_dir, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    coco = COCO(json_path)
    img_ids = coco.getImgIds()
    
    print(f"Found {len(img_ids)} images. Starting priority mask generation...")

    for img_id in img_ids:
        img_metadata = coco.loadImgs(img_id)[0]
        file_name = img_metadata['file_name']
        mask = np.zeros((img_metadata['height'], img_metadata['width']), dtype=np.uint8)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)

        # --- THE FIX: SORT BY PRIORITY ---
        # We sort so that Class 1 (Roof) is processed LAST.
        # This way, if a shadow (Class 4) and a roof (Class 1) share a pixel,
        # the roof will overwrite the shadow.
        annotations.sort(key=lambda x: 1 if x['category_id'] == 1 else 0)

        for ann in annotations:
            if 'segmentation' not in ann or not ann['segmentation']:
                continue
            
            try:
                category_id = ann['category_id']
                pixel_mask = coco.annToMask(ann)
                # Overwrite pixels with the current category_id
                mask[pixel_mask == 1] = category_id
            except Exception as e:
                print(f"Error on annotation {ann['id']}: {e}")

        # Save the fixed mask
        mask_filename = os.path.splitext(file_name)[0] + ".png"
        mask_path = os.path.join(output_dir, mask_filename)
        Image.fromarray(mask).save(mask_path)
        
    print(f"Done! All fixed masks saved to: {output_dir}")

# --- EXECUTION ---
generate_all_masks(
    json_path="data/annotations.json", 
    img_dir="data/images", 
    output_dir="data/masks"
)