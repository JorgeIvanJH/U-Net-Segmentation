import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torch
import os
import sys
import albumentations as A
from albumentations.pytorch import ToTensorV2
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from models.prompt_based_unet import PromptUNet

model = PromptUNet(device="cpu", n_classes=4)
model.load_state_dict(torch.load("UNet_PROMPT_experiment/saved_models/train_model_13_weights_only.pth", map_location="cpu"))

st.title("🖱️ Click on the Image to Get Pixel Coordinates")

def get_test_sample(pil_image, point, img_size=224, device="cpu"):

    # Convert PIL to OpenCV format
    image = np.array(pil_image)

    # Albumentations transform
    transform = A.Compose([
        A.Resize(img_size, img_size),
        ToTensorV2()
    ])

    # Apply transform
    transformed = transform(image=image)
    image_tensor = transformed["image"].unsqueeze(0).float().to(device)  # [1, 3, H, W]

    # Resize point accordingly
    orig_w, orig_h = pil_image.size
    scale_x = img_size / orig_w
    scale_y = img_size / orig_h
    x, y = int(point[0] * scale_x), int(point[1] * scale_y)
    print(f"Original point: {point}, Resized point: ({x}, {y})")
    # Generate heatmap
    heatmap = np.zeros((img_size, img_size), dtype=np.float32)
    if 0 <= x < img_size and 0 <= y < img_size:
        heatmap[y, x] = 1
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=5)
        heatmap = heatmap / np.max(heatmap)

    heatmap_tensor = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0).float().to(device)  # [1, 1, H, W]

    return image_tensor, heatmap_tensor


def visualize_segmentation(model, image, heatmap, device, num_classes=2, idx=0, shadow_color=(0, 128, 255), alpha=0.5):
    # Log shapes
    print(f"image shape: {image.shape}, heatmap shape: {heatmap.shape}")

    # Predict
    with torch.no_grad():
        pred = model(image, heatmap)
        pred_mask = torch.argmax(pred.squeeze(0), dim=0).cpu().numpy()  # [H, W]
    # Convert image tensor to numpy [H, W, C]
    image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image_np = np.array(image_np).astype(np.float32)

    # Create shadow layer
    shadow_layer = np.full_like(image_np, shadow_color, dtype=np.float32)
    mask_3ch = np.stack([pred_mask]*3, axis=-1).astype(np.float32)

    # Blend only the masked region
    highlighted = np.where(
        mask_3ch == 1,  # [H, W, 1] for broadcasting
        (alpha * shadow_layer + (1 - alpha) * image_np), image_np)
    highlighted = np.clip(highlighted, 0, 255).astype(np.uint8)
    return highlighted



uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")  # Ensure image is in RGB mode

    # Display image (optional)
    st.image(image, caption="Click on the image below")

    # Create canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",  # Fill color
        stroke_width=1,
        background_image=image,
        update_streamlit=True,
        height=image.height,
        width=image.width,
        drawing_mode="point",
        key="canvas",
    )

    # Get clicked point
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        if objects:
            last_click = objects[-1]
            x = int(last_click["left"])
            y = int(last_click["top"])

            image_tensor, heatmap_tensor = get_test_sample(image, (x, y), device="cpu")
            image_mask = visualize_segmentation(model, image_tensor, heatmap_tensor, device="cpu")

            st.image(image_mask, caption="Segmented Image", use_column_width=True)
            st.write(f"📍 Last clicked pixel: **({x}, {y})**")
