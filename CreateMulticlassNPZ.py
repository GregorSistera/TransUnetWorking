import os
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# === CONFIGURATION ===
IMAGE_DIR = Path(r"C:\Users\Georges\Desktop\IHCEPITHELIALSEGDATASET\images")
MASK_DIR = Path(r"C:\Users\Georges\Desktop\IHCEPITHELIALSEGDATASET\masks")
OUTPUT_DIR = Path(r"C:\Users\Georges\Desktop\IHCEPITHELIALSEGDATASET\npz_data\train_npz")
LIST_DIR = Path(r"C:\Users\Georges\Desktop\IHCEPITHELIALSEGDATASET\lists\lists_MyEpithelialDataset")

# Create folders if they don't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LIST_DIR.mkdir(parents=True, exist_ok=True)

# === COLOR MAP: Map RGB values to class indices ===
COLOR_MAP = {
    (255, 255, 255): 0,  # Background
    (255, 0, 0): 1,      # Red
    (0, 255, 0): 2,      # Green
    (0, 0, 255): 3,      # Blue
    # Add more classes/colors as needed
}
TOLERANCE = 30  # Maximum color distance allowed


# === HELPERS ===
def clean_filename(name):
    return ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in name)


def convert_mask_to_multiclass_fuzzy(mask_img: Image.Image, color_map: dict, tolerance: int = 30) -> np.ndarray:
    """
    Convert an RGB mask to a 2D array of class indices using fuzzy color matching.
    """
    mask_arr = np.array(mask_img.convert("RGB"))
    h, w, _ = mask_arr.shape
    class_mask = np.zeros((h, w), dtype=np.uint8)

    known_colors = np.array(list(color_map.keys()))  # (N, 3)
    class_indices = list(color_map.values())

    flat_pixels = mask_arr.reshape(-1, 3)
    output_flat = np.zeros((flat_pixels.shape[0],), dtype=np.uint8)

    for i, pixel in enumerate(flat_pixels):
        dists = np.linalg.norm(known_colors - pixel, axis=1)
        min_idx = np.argmin(dists)
        if dists[min_idx] <= tolerance:
            output_flat[i] = class_indices[min_idx]
        else:
            output_flat[i] = 0  # Default to background

    return output_flat.reshape(h, w)


# === LOAD FILES ===
image_files = sorted([f for f in IMAGE_DIR.glob("*") if f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif'}])
image_mask_pairs = []

for image_path in image_files:
    base = image_path.stem
    cleaned = clean_filename(base)
    mask_path = MASK_DIR / f"{image_path.stem}.png"  # Adjust extension if needed
    if mask_path.exists():
        image_mask_pairs.append((image_path, mask_path, cleaned))
    else:
        print(f"[WARN] No mask found for {image_path.name}")

print(f"✅ Found {len(image_mask_pairs)} valid image-mask pairs.")

# === SPLIT ===
train_pairs, val_pairs = train_test_split(image_mask_pairs, test_size=0.2, random_state=42)


# === SAVE FUNCTION ===
def save_pairs(pairs, split: str):
    list_path = LIST_DIR / f"{split}.txt"
    with open(list_path, "w") as f:
        for image_path, mask_path, cleaned_name in tqdm(pairs, desc=f"Saving {split}"):
            try:
                image = np.array(Image.open(image_path).convert("RGB"))
                mask_img = Image.open(mask_path)
                mask = convert_mask_to_multiclass_fuzzy(mask_img, COLOR_MAP, tolerance=TOLERANCE)

                npz_path = OUTPUT_DIR / f"{cleaned_name}.npz"
                np.savez_compressed(npz_path, image=image.astype(np.uint8), mask=mask.astype(np.uint8))

                f.write(f"{cleaned_name}\n")
            except Exception as e:
                print(f"[ERROR] Failed {cleaned_name}: {e}")


# === PROCESS AND SAVE ===
save_pairs(train_pairs, "train")
save_pairs(val_pairs, "val")

print("✅ Dataset processing complete.")
