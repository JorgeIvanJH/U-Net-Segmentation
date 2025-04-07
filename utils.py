# Standard library
import os
import numpy as np
import torch
from tqdm import tqdm
# Plotly for interactive visualizations
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import cv2
from albumentations.pytorch import ToTensorV2
import albumentations as A
import cv2
import numpy as np
import pandas as pd
import os
from config import SEED, TRAIN_SIZE, IMG_SIZE
import csv


# Matplotlib for static visualizations
import matplotlib.pyplot as plt

map_class = {}
map_class[0] = "Background"
map_class[1] = "Cat"
map_class[2] = "Dog"
map_class[3] = "Border"



def analyze(image, mask, ismask=True):
    """Detailed analysis with plotly (mimicking matplotlib style)"""
    if not ismask:
        # If no mask, simply show the image
        fig = make_subplots(rows=1, cols=2)
        fig.add_trace(go.Image(z=image), row=1, col=1)
        fig.update_layout(
            title_text="Image",
            title_x=0.5
        )
        fig.add_trace(go.Image(z=mask), row=1, col=2)
        fig.update_layout(
            title_text="Mask",
            title_x=0.5
        )
    else:
        # Create subplot with two images (image and mask)
        fig = make_subplots(rows=1, cols=2)

        # Add the image on the first subplot
        fig.add_trace(go.Image(z=image), row=1, col=1)
        fig.update_layout(
            title_text="Image",
            title_x=0.5
        )

        # Add the mask on the second subplot with colorbar and title
        fig.add_trace(go.Heatmap(z=mask[::-1], colorscale='Viridis', coloraxis="coloraxis"), row=1, col=2)
        fig.update_layout(
            title_text="Mask",
            title_x=0.5,
            coloraxis=dict(colorbar=dict(title="Mask Intensity"))
        )
        fig.update_yaxes(scaleanchor="y", row=1, col=2)  # This ensures the axes are consistent

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
        im = plt.imshow(mask, cmap='inferno', vmin=0, vmax=3)
        plt.title(masktitle)
        plt.axis('off')

        # Add colorbar (acts as legend)
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04, ticks=[0, 1, 2, 3])
        cbar.set_ticklabels(['0: background', '1: cat', '2: dog', '3: border'])
        cbar.set_label('Mask Class')

        plt.show()

def training_plot(training_summary_route):
    df = pd.read_csv(training_summary_route)

    best_epoch = df[df.val_iou == df.val_iou.max()]
    print("Best epoch:\n", best_epoch)

    fig, ax1 = plt.subplots(figsize=(14, 6))  # Wider figure

    # IoU plots
    ax1.plot(df.index, df['train_iou'], label='Train IoU', marker='o', color='blue')
    ax1.plot(df.index, df['val_iou'], label='Validation IoU', marker='o', color='orange')
    ax1.axvline(x=best_epoch.index[0], color='red', linestyle='--', label='Best Epoch')
    ax1.axhline(y=best_epoch.val_iou.values[0], color='green', linestyle='--', label='Best Val IoU')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('IoU', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True)

    # Loss plots
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['train_loss'], label='Train Loss', marker='o', color='purple')
    ax2.plot(df.index, df['val_loss'], label='Validation Loss', marker='o', color='brown')
    ax2.axhline(y=best_epoch.val_loss.values[0], color='green', linestyle='--', label='Best Val Loss')
    ax2.set_ylabel('Loss', color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')

    # Combine legends and place outside right
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper left', bbox_to_anchor=(1.05, 0.5))  # Shift legend further right

    plt.title('Training and Validation IoU and Loss')
    plt.subplots_adjust(right=0.8)  # Give more room on the right
    plt.show()

def rgb_to_class(mask):
    """Convert an RGB mask to a single-channel class index mask."""
    h, w, _ = mask.shape
    class_mask = np.zeros((h, w), dtype=np.uint8)

    color_map = {
            (128, 0, 0): 1,  # Cat
            (0, 128, 0): 2,  # Dog
            (0, 0, 0): 0,  # Background
            (255, 255, 255): 3,  # Border
        }

    # Iterate over color map and assign class values
    for rgb, class_idx in color_map.items():
        mask_match = np.all(mask == np.array(rgb, dtype=np.uint8), axis=-1)
        class_mask[mask_match] = class_idx
    return class_mask

def get_test_sample(image_path, mask_path):
    image_path = os.path.join(os.getcwd(), image_path)
    mask_path = os.path.join(os.getcwd(), mask_path)

    transform = A.Compose(
        [
            A.Resize(IMG_SIZE, IMG_SIZE),  
            ToTensorV2(),
        ]
    )
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    mask = rgb_to_class(mask)
    transformed = transform(image=image, mask=mask)
    image, mask = transformed["image"], transformed["mask"]
    mask = mask.clone().detach().long()
    image = image / 255.0
    return image, mask

def evaluate_model_on_sample(nn_model, image, mask, show = False):
    nn_model.eval()
    with torch.no_grad():
        logits = nn_model(image.unsqueeze(0).to("cpu"))
        predicted_classes = torch.argmax(logits, dim=1)
        segmentation_mask = predicted_classes[0].cpu().numpy()
    if show:
        visualize(image.permute(1, 2, 0).cpu().numpy(), mask)
        visualize(image.permute(1, 2, 0).cpu().numpy(), segmentation_mask, "Predicted Mask")
    return segmentation_mask

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

def dice_coefficient(pred, target, num_classes, epsilon=1e-6):
    dice = []
    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()

        intersection = torch.sum(pred_c * target_c)
        union = torch.sum(pred_c) + torch.sum(target_c)
        d = (2. * intersection + epsilon) / (union + epsilon)
        dice.append(d.item())
    return dice

def pixel_accuracy(pred, target):
    correct = (pred == target).float()
    acc = correct.sum() / correct.numel()
    return acc.item()

def evaluate_model(nn_model, test_dataloader, device, ignore_border=False, limit_samples=False):
    """
    Evaluate the model on the test dataset and calculate mean IoU, Dice, and Pixel Accuracy.
    """
    iou_list = []
    dice_list = []
    pixel_acc_list = []

    for i, (image, mask, _) in enumerate(tqdm(test_dataloader)):
        image = image.to(device)
        mask = mask.to(device)

        nn_model.eval()
        with torch.no_grad():
            logits = nn_model(image)  # Shape: (B, C, H, W)
            predicted_classes = torch.argmax(logits, dim=1)  # Shape: (B, H, W)

        iou = intersecton_over_union(predicted_classes, mask, num_classes=4, ignore_border=ignore_border)
        dice = dice_coefficient(predicted_classes, mask, num_classes=4)
        acc = pixel_accuracy(predicted_classes, mask)

        iou_list.append(iou)
        dice_list.append(dice)
        pixel_acc_list.append(acc)

        if limit_samples and i > 10:
            break

    # Convert to numpy for stats
    iou_array = np.array(iou_list)
    dice_array = np.array(dice_list)

    mean_iou_per_class = np.nanmean(iou_array, axis=0)
    std_iou_per_class = np.nanstd(iou_array, axis=0)
    mean_iou_total = np.nanmean(iou_array)
    std_iou_total = np.nanstd(iou_array)

    mean_dice_per_class = np.nanmean(dice_array, axis=0)
    mean_dice_total = np.nanmean(dice_array)
    mean_pixel_acc = np.mean(pixel_acc_list)

    for i, (iou_m, iou_s, dice_m) in enumerate(zip(mean_iou_per_class, std_iou_per_class, mean_dice_per_class)):
        print(f"{map_class[i]} - IoU: {iou_m:.4f} ± {iou_s:.4f}, Dice: {dice_m:.4f}")
    
    print(f"TOTAL Mean IoU: {mean_iou_total:.4f} ± {std_iou_total:.4f}")
    print(f"TOTAL Mean Dice: {mean_dice_total:.4f}")
    print(f"TOTAL Pixel Accuracy: {mean_pixel_acc:.4f}")
    return mean_iou_total, mean_dice_total, mean_pixel_acc


def save_statistics(experiment_log_dir, filename, stats_dict, current_epoch, continue_from_mode=False, save_full_dict=False):
    """
    Saves the statistics in stats dict into a csv file. Using the keys as the header entries and the values as the
    columns of a particular header entry
    :param experiment_log_dir: the log folder dir filepath
    :param filename: the name of the csv file
    :param stats_dict: the stats dict containing the data to be saved
    :param current_epoch: the number of epochs since commencement of the current training session (i.e. if the experiment continued from 100 and this is epoch 105, then pass relative distance of 5.)
    :param save_full_dict: whether to save the full dict as is overriding any previous entries (might be useful if we want to overwrite a file)
    :return: The filepath to the summary file
    """
    summary_filename = os.path.join(experiment_log_dir, filename)
    mode = 'a' if continue_from_mode else 'w'
    with open(summary_filename, mode) as f:
        writer = csv.writer(f)
        if not continue_from_mode:
            writer.writerow(list(stats_dict.keys()))

        if save_full_dict:
            total_rows = len(list(stats_dict.values())[0])
            for idx in range(total_rows):
                row_to_add = [value[idx] for value in list(stats_dict.values())]
                writer.writerow(row_to_add)
        else:
            row_to_add = [value[current_epoch] for value in list(stats_dict.values())]
            writer.writerow(row_to_add)

    return summary_filename

def save_statistics(log_dir, log_csv_name, stats_dict):
    
    os.makedirs(log_dir, exist_ok=True)
    csv_file_path = os.path.join(log_dir, log_csv_name)
    file_exists = os.path.isfile(csv_file_path)
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=stats_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(stats_dict)

def evaluate_model_dice(nn_model, test_dataloader, device, limit_samples=False, noise_function = None):
    """
    Evaluate the model on the test dataset and calculate mean IoU, Dice, and Pixel Accuracy.
    """
    def dice_coefficient(pred, target, num_classes, epsilon=1e-6):
        dice = []
        for c in range(num_classes):
            pred_c = (pred == c).float()
            target_c = (target == c).float()

            intersection = torch.sum(pred_c * target_c)
            union = torch.sum(pred_c) + torch.sum(target_c)
            d = (2. * intersection + epsilon) / (union + epsilon)
            dice.append(d.item())
        return dice
    
    dice_list = []

    for i, (image, mask, _) in enumerate(tqdm(test_dataloader)):
        image = image.to(device)
        mask = mask.to(device)

        nn_model.eval()
        with torch.no_grad():
            image = image if noise_function is None else noise_function(image)
            logits = nn_model(image)  # Shape: (B, C, H, W)
            predicted_classes = torch.argmax(logits, dim=1)  # Shape: (B, H, W)

        dice = dice_coefficient(predicted_classes, mask, num_classes=4)

        dice_list.append(dice)

        if limit_samples and i > 10:
            break

    # Convert to numpy for stats
    dice_array = np.array(dice_list)

    mean_dice_per_class = np.nanmean(dice_array, axis=0)
    mean_dice_total = np.nanmean(dice_array)

    results = {}
    for i, dice_m in enumerate(mean_dice_per_class):
        results[map_class[i]] = dice_m
    results["TOTAL"] = mean_dice_total
    return results

def visualize_noise_effect(model,image_cat_easy_back, mask_cat_easy_back,image_dog_easy_back, mask_dog_easy_back,noise_function,conds_of_noise):
    pred_masks_dogs = []
    pred_masks_cats = []
    dice_acc_cats = []
    dice_acc_dogs = []
    for cond in conds_of_noise:
        for image, mask, clss in zip([image_cat_easy_back,image_dog_easy_back], [mask_cat_easy_back, mask_dog_easy_back], ["c","d"]):
            image_w_noise = noise_function(image, cond)
            pred_mask = evaluate_model_on_sample(model, image_w_noise, mask)
            pred_masks_cats.append(pred_mask) if clss == "c" else pred_masks_dogs.append(pred_mask)
            dice = np.sum((pred_mask == mask.numpy()) & (mask.numpy() > 0)) / np.sum(mask.numpy() > 0)
            dice_acc_cats.append(dice) if clss == "c" else dice_acc_dogs.append(dice)



    # Plot the images with noise and their corresponding predicted masks for cats and dogs separately
    fig, axes = plt.subplots(2, len(conds_of_noise), figsize=(5 * len(conds_of_noise), 10))

    # Plot for cats
    for i, (cond, noisy_image, pred_mask, dice) in enumerate(zip(conds_of_noise, 
                                                                    [noise_function(image_cat_easy_back, cond) for cond in conds_of_noise], 
                                                                    pred_masks_cats, 
                                                                    dice_acc_cats)):
        # Plot noisy image
        axes[0, i].imshow(noisy_image.permute(1, 2, 0).numpy())
        axes[0, i].set_title(f"Cat - Noisy Image\n(cond={cond})", fontsize=25)
        axes[0, i].axis("off")

        # Plot predicted mask
        axes[1, i].imshow(pred_mask, cmap='inferno')
        axes[1, i].set_title(f"Cat - Predicted Mask\n(cond={cond}, Dice={dice:.2f})", fontsize=25)
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(2, len(conds_of_noise), figsize=(5 * len(conds_of_noise), 10))

    # Plot for dogs
    for i, (cond, noisy_image, pred_mask, dice) in enumerate(zip(conds_of_noise, 
                                                                    [noise_function(image_dog_easy_back, cond) for cond in conds_of_noise], 
                                                                    pred_masks_dogs, 
                                                                    dice_acc_dogs)):
        # Plot noisy image
        axes[0, i].imshow(noisy_image.permute(1, 2, 0).numpy())
        axes[0, i].set_title(f"Dog - Noisy Image\n(cond={cond})", fontsize=25)
        axes[0, i].axis("off")

        # Plot predicted mask
        axes[1, i].imshow(pred_mask, cmap='inferno')
        axes[1, i].set_title(f"Dog - Predicted Mask\n(cond={cond}, Dice={dice:.2f})", fontsize=25)
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()

    plt.tight_layout()
    plt.show()