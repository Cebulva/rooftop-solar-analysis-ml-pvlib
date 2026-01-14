import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torchvision import transforms

# ==========================================
# 1. Dataset Class: Multi-Class to Binary
# ==========================================
class BinaryRoofDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # Match images and masks by filename
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, os.path.splitext(img_name)[0] + ".png")
        
        # Load satellite image
        image = Image.open(img_path).convert("RGB")
        
        # Load multi-class mask (0, 1, 2, 3, 4)
        mask_multi = Image.open(mask_path).convert("L")
        mask_np = np.array(mask_multi)

        # --- THE BINARY FILTER ---
        # Class 1 is 'Roof'. We set everything else to 0.
        # This focuses the AI entirely on rooftops again.
        binary_mask = np.where(mask_np == 1, 1, 0).astype(np.float32)
        
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        # Binary models expect mask shape (1, H, W)
        mask_tensor = torch.as_tensor(binary_mask).unsqueeze(0)
        
        return image, mask_tensor

# ==========================================
# 2. Training Function
# ==========================================
def run_binary_training():
    # Setup Device
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize Model (classes=1 for Binary)
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    ).to(device)

    # Load your "Better" Pre-trained Weights
    weights_path = "models/segm_Unet_model_aerial.pth"
    if os.path.exists(weights_path):
        print("Loading pre-trained model for improvement...")
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print("Warning: Pre-trained weights not found. Training from scratch.")

    # Binary Loss and Optimizer
    # BCEWithLogitsLoss is much more stable for binary roof detection
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5) # Low LR to preserve quality

    # Data Prep
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = BinaryRoofDataset("data/images", "data/masks", transform=transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    print(f"Fine-tuning on {len(dataset)} hand-labeled images...")
    model.train()
    for epoch in range(20):
        epoch_loss = 0
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/20 | Binary Loss: {epoch_loss/len(loader):.4f}")

    # Save the improved version
    torch.save(model.state_dict(), "models/final_refined_roof_model.pth")
    print("Success: improved_binary_roof_model.pth saved.")

# ==========================================
# 3. Prediction & Visual Comparison
# ==========================================
def predict_and_compare(test_image_path):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Load the newly trained model
    model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1).to(device)
    model.load_state_dict(torch.load("final_refined_roof_model.pth", map_location=device))
    model.eval()

    # Prep Image
    raw_img = Image.open(test_image_path).convert("RGB")
    prep = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    input_t = prep(raw_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_t)
        # Sigmoid converts raw output to probability (0 to 1)
        probs = torch.sigmoid(logits)
        prediction = (probs > 0.5).float().squeeze().cpu().numpy()

    # Show results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.imshow(raw_img); plt.title("Satellite View"); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(prediction, cmap='gray'); plt.title("Improved Roof Mask"); plt.axis('off')
    plt.show()

if __name__ == "__main__":
    run_binary_training()