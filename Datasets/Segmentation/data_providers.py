import torch
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
import albumentations as A
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from config import SEED, TRAIN_SIZE, IMG_SIZE


PATH_TRAINVAL = 'Datasets/Segmentation/TrainVal'
PATH_TEST = 'Datasets/Segmentation/Test'

test_image_path = os.path.join(PATH_TEST, 'color/')
test_mask_path = os.path.join(PATH_TEST, 'label/')
test_image_list_orig = sorted(os.listdir(test_image_path))
test_mask_list_orig = sorted(os.listdir(test_mask_path))

trainval_image_path = os.path.join(PATH_TRAINVAL, 'color/')
trainval_mask_path = os.path.join(PATH_TRAINVAL, 'label/')
trainval_image_list_orig = sorted(os.listdir(trainval_image_path))
trainval_mask_list_orig = sorted(os.listdir(trainval_mask_path))

test_paths_df = pd.DataFrame({"images":test_image_list_orig,"masks": test_mask_list_orig})
test_paths_df["path_images"] = test_image_path
test_paths_df["path_masks"] = test_mask_path
test_paths_df["path_images"] = test_paths_df["path_images"] + test_paths_df["images"]
test_paths_df["path_masks"] = test_paths_df["path_masks"] + test_paths_df["masks"]
test_paths_df.drop(columns=["images","masks"],inplace=True)

trainval_paths_df = pd.DataFrame({"images":trainval_image_list_orig,"masks": trainval_mask_list_orig})
trainval_paths_df["path_images"] = trainval_image_path
trainval_paths_df["path_masks"] = trainval_mask_path
trainval_paths_df["path_images"] = trainval_paths_df["path_images"] + trainval_paths_df["images"]
trainval_paths_df["path_masks"] = trainval_paths_df["path_masks"] + trainval_paths_df["masks"]
trainval_paths_df.drop(columns=["images","masks"],inplace=True)

train_paths_df, val_paths_df = train_test_split(trainval_paths_df, train_size=TRAIN_SIZE, random_state=SEED)



transform = A.Compose(
    [
        A.Resize(int(IMG_SIZE*1.1), int(IMG_SIZE*1.1)),  
        A.RandomCrop(width=IMG_SIZE, height=IMG_SIZE, p=1.0), # Randomly crop image
        # A.Rotate(limit=35, p=1.0), # Rotate image
        # A.HorizontalFlip(p=0.5), # Flip image horizontally
        A.RGBShift(
            r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5
        ),  # Shift RGB colors
        A.RandomBrightnessContrast(p=0.5),  # Randomly change brightness and contrast
        # A.RandomGamma(p=0.5), # Randomly change image gamma
        # A.Blur(p=0.5), # Randomly blur image
        # A.GaussNoise(p=0.5), # Randomly add gaussian noise
        A.HorizontalFlip(p=0.5),
        ToTensorV2(),
    ]
)


class SegmentationDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.color_map = {
            (128, 0, 0): 1,  # Cat
            (0, 128, 0): 2,  # Dog
            (0, 0, 0): 0,  # Background
            (255, 255, 255): 3,  # Border
        }

    def __len__(self):
        return len(self.dataframe)

    def rgb_to_class(self, mask):
        """Convert an RGB mask to a single-channel class index mask."""
        h, w, _ = mask.shape
        class_mask = np.zeros((h, w), dtype=np.uint8)

        # Iterate over color map and assign class values
        for rgb, class_idx in self.color_map.items():
            mask_match = np.all(mask == np.array(rgb, dtype=np.uint8), axis=-1)
            class_mask[mask_match] = class_idx
        return class_mask

    def __getitem__(self, idx):
        image_path = self.dataframe.iloc[idx]["path_images"]
        mask_path = self.dataframe.iloc[idx]["path_masks"]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Image
        if image is None:
            raise FileNotFoundError(f"Could not load image at path: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)  # Mask
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        # Convert RGB mask to class index mask
        mask = self.rgb_to_class(mask)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]
        mask = mask.clone().detach().long()
        image = image / 255.0 
        return image, mask, image*0 # image*0 is a placeholder for the heatmap in the data providers for prompt based unet

train_dataset = SegmentationDataset(train_paths_df, transform=transform)
val_dataset = SegmentationDataset(val_paths_df, transform=transform)
test_dataset = SegmentationDataset(test_paths_df, transform=transform)