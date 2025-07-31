import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # <--- import tqdm

# === PATH CONFIGURATION ===
image_dir = r"C:\Users\Georges\Desktop\IHCEPITHELIALSEGDATASET\images"
mask_dir = r"C:\Users\Georges\Desktop\IHCEPITHELIALSEGDATASET\masks"
save_dir = r"C:\Users\Georges\Desktop\IHCEPITHELIALSEGDATASET\npz_data\train_npz"
list_dir = r"C:\Users\Georges\Desktop\IHCEPITHELIALSEGDATASET\lists\lists_MyEpithelialDataset"

os.makedirs(save_dir, exist_ok=True)
os.makedirs(list_dir, exist_ok=True)

# === IMAGE SIZE NOTE ===
# You said your tiles are 512x512 — if your model accepts that, set img_size=512.
# Using 224 was likely just an example or for models like ResNet, but
# keeping original tile size (512) avoids losing detail.
img_size = 512  # <-- updated

# === MASK PROCESSING ===
def process_mask(mask_img, white_thresh=240):
    mask_arr = np.array(mask_img)
    if mask_arr.ndim == 3 and mask_arr.shape[2] == 3:
        is_white = np.all(mask_arr >= white_thresh, axis=-1)
        binary_mask = (~is_white).astype(np.uint8)
    else:
        raise ValueError("Mask must be RGB with 3 channels")
    return binary_mask

# === IMAGE & MASK PROCESSING ===
filenames = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
processed_count = 0

for fname in tqdm(filenames, desc="Processing images"):  # <--- tqdm here
    image_path = os.path.join(image_dir, fname)
    mask_path = os.path.join(mask_dir, fname)

    if not os.path.exists(mask_path):
        print(f"[WARN] Mask missing for {fname}, skipping.")
        continue

    try:
        img = Image.open(image_path).convert('RGB').resize((img_size, img_size))
        mask = Image.open(mask_path).convert('RGB').resize((img_size, img_size))

        img_np = np.array(img) / 255.0
        mask_np = process_mask(mask)

        npz_name = os.path.splitext(fname)[0] + '.npz'
        np.savez_compressed(os.path.join(save_dir, npz_name),
                            image=img_np.astype(np.float32),
                            mask=mask_np.astype(np.uint8))
        processed_count += 1

    except Exception as e:
        print(f"[ERROR] Failed to process {fname}: {e}")

print(f"\n✅ Processed {processed_count} image-mask pairs.")

# === TRAIN / VAL SPLIT ===
file_ids = [os.path.splitext(f)[0] for f in filenames if os.path.exists(os.path.join(mask_dir, f))]
train_ids, val_ids = train_test_split(file_ids, test_size=0.2, random_state=42)

with open(os.path.join(list_dir, 'train.txt'), 'w') as f:
    f.writelines(f"{item}\n" for item in train_ids)

with open(os.path.join(list_dir, 'val.txt'), 'w') as f:
    f.writelines(f"{item}\n" for item in val_ids)

print(f"📂 Saved train.txt with {len(train_ids)} IDs.")
print(f"📂 Saved val.txt with {len(val_ids)} IDs.")
