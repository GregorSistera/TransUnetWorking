import os
import re
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # for progress bar

# === PATH CONFIGURATION ===
image_dir = r"C:\Users\Georges\Desktop\IHCEPITHELIALSEGDATASET\images"
mask_dir = r"C:\Users\Georges\Desktop\IHCEPITHELIALSEGDATASET\masks"
save_dir = r"C:\Users\Georges\Desktop\IHCEPITHELIALSEGDATASET\npz_data\train_npz"
list_dir = r"C:\Users\Georges\Desktop\IHCEPITHELIALSEGDATASET\lists\lists_MyEpithelialDataset"

os.makedirs(save_dir, exist_ok=True)
os.makedirs(list_dir, exist_ok=True)

# === IMAGE SIZE ===
img_size = 512  # Keep original tile size

# === CLEAN FILENAME FUNCTION ===
def clean_filename(name: str) -> str:
    # Replace any character not a-z, A-Z, 0-9, -, or _ with underscore
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name)

# === MASK PROCESSING ===
def process_mask(mask_img):
    mask_arr = np.array(mask_img)
    if mask_arr.ndim == 3 and mask_arr.shape[2] == 3:
        # Background: exactly white pixels
        is_background = np.all(mask_arr == 255, axis=-1)
        binary_mask = (~is_background).astype(np.uint8)
    else:
        raise ValueError("Mask must be RGB with 3 channels")
    return binary_mask

# === IMAGE & MASK PROCESSING ===
filenames = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
processed_count = 0

cleaned_name_map = {}  # maps original filename stem -> cleaned filename

for fname in tqdm(filenames, desc="Processing images"):
    image_path = os.path.join(image_dir, fname)
    mask_path = os.path.join(mask_dir, fname)

    if not os.path.exists(mask_path):
        print(f"[WARN] Mask missing for {fname}, skipping.")
        continue

    try:
        base_name = os.path.splitext(fname)[0]
        cleaned_name = clean_filename(base_name)
        cleaned_name_map[base_name] = cleaned_name  # save mapping

        # Open and resize both image and mask
        img = Image.open(image_path).convert('RGB').resize((img_size, img_size))
        mask = Image.open(mask_path).convert('RGB').resize((img_size, img_size))

        img_np = np.array(img) / 255.0  # Normalize image to [0,1]
        mask_np = process_mask(mask)

        # Save image and mask as compressed npz
        npz_name = cleaned_name + '.npz'
        np.savez_compressed(os.path.join(save_dir, npz_name),
                            image=img_np.astype(np.float32),
                            mask=mask_np.astype(np.uint8))
        processed_count += 1

    except Exception as e:
        print(f"[ERROR] Failed to process {fname}: {e}")

print(f"\n✅ Processed {processed_count} image-mask pairs.")

# === TRAIN / VAL SPLIT ===
# Only keep file IDs for which masks exist and were processed
valid_file_ids = list(cleaned_name_map.keys())
train_ids_orig, val_ids_orig = train_test_split(valid_file_ids, test_size=0.2, random_state=42)

# Map original names to cleaned names for saving lists
train_ids_cleaned = [cleaned_name_map[n] for n in train_ids_orig]
val_ids_cleaned = [cleaned_name_map[n] for n in val_ids_orig]

with open(os.path.join(list_dir, 'train.txt'), 'w') as f:
    f.writelines(f"{item}\n" for item in train_ids_cleaned)

with open(os.path.join(list_dir, 'val.txt'), 'w') as f:
    f.writelines(f"{item}\n" for item in val_ids_cleaned)

print(f"📂 Saved train.txt with {len(train_ids_cleaned)} IDs.")
print(f"📂 Saved val.txt with {len(val_ids_cleaned)} IDs.")
