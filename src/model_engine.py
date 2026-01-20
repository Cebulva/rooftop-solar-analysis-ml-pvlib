import torch
import segmentation_models_pytorch as smp
import streamlit as st

def get_device():
    return 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_roof_model(model_path):
    device = get_device()
    model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def run_inference(model, image_tensor):
    device = get_device()
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        logits = model(image_tensor)
        # Returns raw numpy mask for image_processing.py
        return (torch.sigmoid(logits) > 0.5).float().squeeze().cpu().numpy()