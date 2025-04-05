import os
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image


datasets_dir = os.path.dirname(os.path.dirname(os.getcwd()))
segmentation_dir = os.path.join(datasets_dir, "Datasets/Segmentation")
prompt_dir = os.path.join(datasets_dir, "Datasets/Prompt_based_segmentation")
print(prompt_dir)

folder_names = sorted([f for f in os.listdir(segmentation_dir) if os.path.isdir(os.path.join(segmentation_dir, f))])


def sample_point_from_mask(mask, pixel_conf=10):
    ys, xs = np.where(mask >= 0)
    if len(xs) == 0:
        return None

    valid_sample = False
    while not valid_sample:
        rand_y = np.random.choice(ys)
        rand_x = np.random.choice(xs)

        for y_pad in range(-pixel_conf, pixel_conf + 1):
            for x_pad in range(-pixel_conf, pixel_conf + 1):
                y = rand_y + y_pad
                x = rand_x + x_pad
                if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                    if mask[y, x] != -1:
                        valid_sample = True
                        break
            if valid_sample:
                break

    return rand_x, rand_y

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
        return heatmap, 0

    dominant_class = max(filtered, key=lambda x: x[1])[0]
    return heatmap, dominant_class

def generate_prompt_segmentation_dataset():
    # Datasets/Segmentation
    for folder_name in folder_names:

        OUTPUT_IMAGE_DIR = os.path.join(prompt_dir,folder_name, "images")
        OUTPUT_MASK_DIR = os.path.join(prompt_dir, folder_name, "masks")
        OUTPUT_PROMPT_DIR = os.path.join(prompt_dir, folder_name, "prompts")
        if not os.path.exists(OUTPUT_IMAGE_DIR):
            os.makedirs(OUTPUT_IMAGE_DIR)
        if not os.path.exists(OUTPUT_MASK_DIR):
            os.makedirs(OUTPUT_MASK_DIR)
        if not os.path.exists(OUTPUT_PROMPT_DIR):
            os.makedirs(OUTPUT_PROMPT_DIR)

        folder_path = os.path.join(segmentation_dir, folder_name)
        image_mask_dir = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))])
        image_list = sorted([os.path.join(image_mask_dir[0], f) for f in os.listdir(image_mask_dir[0]) if f.endswith(".jpg") or f.endswith(".png")])
        mask_list = sorted([os.path.join(image_mask_dir[1], f) for f in os.listdir(image_mask_dir[1]) if f.endswith(".jpg") or f.endswith(".png")])
        # print(image_list)
        # print(mask_list)
        for img_path, mask_path in tqdm(zip(image_list, mask_list), total=len(image_list)):
            img_name = os.path.basename(img_path)
            mask_name = os.path.basename(mask_path)

            img = np.array(Image.open(img_path).convert("RGB"))
            mask = np.array(Image.open(mask_path))
            mask[mask == 255] = -1  # To ignore borders (which are 255 in the mask)

            point = sample_point_from_mask(mask)
            if point is None:
                continue

            heatmap, dominant_class = generate_heatmap(point, mask)
            mask = (mask == dominant_class).astype(np.uint8)
        
            Image.fromarray(img).save(os.path.join(OUTPUT_IMAGE_DIR, img_name))
            Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(OUTPUT_MASK_DIR, mask_name))
            heatmap_img = (heatmap * 255).astype(np.uint8)
            Image.fromarray(heatmap_img).save(os.path.join(OUTPUT_PROMPT_DIR, mask_name))


if __name__ == "__main__":
    generate_prompt_segmentation_dataset()
    print("✅ Preprocessing complete: saved images, masks, and prompts.")