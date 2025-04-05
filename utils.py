# Standard library
import os
import numpy as np
import torch
from tqdm import tqdm
# Plotly for interactive visualizations
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Matplotlib for static visualizations
import matplotlib.pyplot as plt

map_class = {}
map_class[0] = "Background"
map_class[1] = "Cat"
map_class[2] = "Dog"
map_class[3] = "Border"



def analyze(image, mask=None):
    """with plotly for detailed analysis"""
    if mask is None:
        fig = px.imshow(image)
        fig.show()
    fig = make_subplots(
    rows=1, cols=2)
    fig.add_trace(go.Image(z=image), 1, 1)
    fig.add_trace(go.Image(z=mask), 1, 2)
    fig.show()

def visualize(image, mask=None, masktitle='Mask'):
    plt.figure(figsize=(10, 5))
    
    if mask is None:
        plt.imshow(image)
        plt.axis('off')
        plt.show()
    else:
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        im = plt.imshow(mask, cmap='jet', vmin=0, vmax=3)  # Set fixed colorbar range
        plt.title(masktitle)
        plt.axis('off')

        # Add colorbar (acts as legend)
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label('Mask Intensity')

        plt.show()



def intersecton_over_union(preds, labels, num_classes, ignore_border=False, border_idx=3):
    
    ious = []
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (labels == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        
        if ignore_border and cls == border_idx:
            pred_inds = pred_inds & (preds != border_idx)
            target_inds = target_inds & (labels != border_idx)
            intersection = (pred_inds & target_inds).sum().item()
            union = (pred_inds | target_inds).sum().item()

        if union == 0:
            ious.append(float('nan'))  # ignore class if not present
        else:
            ious.append(intersection / union)
    return ious

def evluate_model(nn_model, test_dataloader, device, ignore_border=False, limit_samples=False):
    """
    Evaluate the model on the test dataset and calculate the mean IoU for each class.
    """
    iou_list = []
    for i,(image, mask, _) in enumerate(tqdm(test_dataloader)):
        image = image.to(device)
        mask = mask.to(device)

        nn_model.eval()
        with torch.no_grad():
            logits = nn_model(image)  # Output shape: (B, n_classes, H, W)
            predicted_classes = torch.argmax(logits, dim=1)  # Shape: (B, H, W)

        iou = intersecton_over_union(predicted_classes, mask, num_classes=4, ignore_border=ignore_border)
        iou_list.append(iou)
        if limit_samples and i > 10:
            break

    mean_per_class, std_per_class = np.nanmean(iou_list,axis=0), np.nanstd(iou_list,axis=0)
    mean_total, std_total = np.nanmean(iou_list), np.nanstd(iou_list)

    for i, (mean, std) in enumerate(zip(mean_per_class, std_per_class)):
        print(f"{map_class[i]}: {mean:.4f} ± {std:.4f}")
    print(f"TOTAL Mean IoU: {mean_total:.4f} ± {std_total:.4f}")
