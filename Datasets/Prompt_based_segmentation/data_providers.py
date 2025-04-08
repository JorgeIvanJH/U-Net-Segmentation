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
import random
random.seed(SEED)
np.random.seed(SEED)


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
        A.Rotate(limit=35, p=1.0), # Rotate image
        A.HorizontalFlip(p=0.5), # Flip image horizontally
        A.RGBShift(
            r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5
        ),  # Shift RGB colors
        A.RandomBrightnessContrast(p=0.5),  # Randomly change brightness and contrast
        A.RandomGamma(p=0.5), # Randomly change image gamma
        A.Blur(p=0.5), # Randomly blur image
        A.GaussNoise(p=0.5), # Randomly add gaussian noise
        A.HorizontalFlip(p=0.5),
    ]
)

adjust = A.Compose(
    [
        A.Resize(int(IMG_SIZE), int(IMG_SIZE)),  
    ]
)

totensor = A.Compose(
    [
        ToTensorV2(),
    ]
)

def sample_point_from_mask(mask, elem_to_point, pixel_conf=10):
    """Sample a point from the mask with a given pixel confidence."""
    ys, xs = np.where(mask == elem_to_point)
    if len(xs) == 0:
        return None

    indices = list(zip(ys, xs))
    np.random.shuffle(indices)

    for rand_y, rand_x in indices:
        for y_pad in range(-pixel_conf, pixel_conf + 1):
            for x_pad in range(-pixel_conf, pixel_conf + 1):
                y = rand_y + y_pad
                x = rand_x + x_pad
                if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                    if mask[y, x] != -1:
                        return rand_x, rand_y

    return None  # If no valid sample found after checking all

def generate_heatmap(point, mask, sigma=5):
    shape = mask.shape
    heatmap = np.zeros(shape, dtype=np.float32)
    if point:
        x, y = point
        heatmap[y, x] = 1
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigma)
        heatmap = heatmap / np.max(heatmap)

    region = mask[(heatmap > 0.5) & (mask != -1)]

    if len(region) == 0:
        return heatmap[np.newaxis, ...], None

    values, counts = np.unique(region, return_counts=True)
    filtered = [(int(v), int(c)) for v, c in zip(values, counts) if v in [1, 2]]
    
    if not filtered:
        return heatmap
    return heatmap

class SegmentationDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform if transform else adjust
        self.color_map = {
            (0, 0, 0): 0,  # Background
            (128, 0, 0): 1,  # Cat
            (0, 128, 0): 2,  # Dog
            (255, 255, 255): 3,  # Border
        }
        self.classes_preferences = {
                               0: 0.2, # Background, 
                               1: 0.5, # Cat
                               2: 0.3, # Dog
                               3:0 # Border
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
            print(f"Could not load image at path: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)  # Mask
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        # Convert RGB mask to class index mask
        mask = self.rgb_to_class(mask)


        elems_in_mask = sorted(np.unique(mask))[:-1] # Exclude the last element which is the border
        if len(elems_in_mask) == 2:
            if elems_in_mask[1] == 2: # Dog is present
                weights = [0.1,0.2]
            else: # Cat is present
                weights = [0.1, 0.5] 
        else: # Both Cat and Dog are present
            weights = [0.1, 0.5, 0.3]

        elem_to_point = random.choices(elems_in_mask, weights=weights, k=1)[0]
  
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        point = sample_point_from_mask(mask,elem_to_point, pixel_conf=10)
        if point is None:
            point = sample_point_from_mask(mask, elem_to_point, pixel_conf=0)

        heatmap = generate_heatmap(point, mask)
        mask = (mask == elem_to_point).astype(np.uint8)

        image = totensor(image=image)["image"]
        mask = totensor(image=mask)["image"]
        heatmap = totensor(image=heatmap)["image"]

        mask = mask.clone().detach().long()
        image = image / 255.0 
        return image, mask, heatmap # image*0 is a placeholder for the heatmap in the data providers for prompt based unet

train_dataset = SegmentationDataset(train_paths_df, transform=transform)
val_dataset = SegmentationDataset(val_paths_df, transform=None)
test_dataset = SegmentationDataset(test_paths_df, transform=None)