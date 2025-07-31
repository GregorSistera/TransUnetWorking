import os
import re
import numpy as np
from PIL import Image
from skimage import io
from skimage.color import rgb2hed, hed2rgb
from skimage.util import img_as_ubyte
from tqdm import tqdm

# === CONFIG ===
original_folder = r"C:\Users\Georges\Desktop\TESTMULTIPLEXHEMATOXYLINEXTRACTION\Original"
hematoxylin_folder = r"C:\Users\Georges\Desktop\TESTMULTIPLEXHEMATOXYLINEXTRACTION\Hematoxylin"
mask_folder = r"C:\Users\Georges\Desktop\TESTMULTIPLEXHEMATOXYLINEXTRACTION\Labels"
npz_save_folder = r"C:\Users\Georges\Desktop\TESTMULTIPLEXHEMATOXYLINEXTRACTION\npz_data"
list_save_path = r"C:\Users\Georges\Desktop\TESTMULTIPLEXHEMATOXYLINEXTRACTION\lists\infer.txt"


tile_size = 512
y_start, x_start = 0, 0  # fixed tile position from your example

os.makedirs(hematoxylin_folder, exist_ok=True)
os.makedirs(npz_save_folder, exist_ok=True)
os.makedirs(os.path.dirname(list_save_path), exist_ok=True)

valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')

def clean_filename(name: str) -> str:
    # replace anything except a-z, A-Z, 0-9, dash, underscore with _
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name)

def extract_hematoxylin_tile(image_path, save_path):
    ihc_rgb = io.imread(image_path)
    height, width = ihc_rgb.shape[:2]

    y_end = y_start + tile_size
    x_end = x_start + tile_size

    if y_end > height or x_end > width:
        raise ValueError(f"Image too small for tile at ({y_start},{x_start})")

    tile = ihc_rgb[y_start:y_end, x_start:x_end, :]

    ihc_hed = rgb2hed(tile)
    h_channel = ihc_hed[:, :, 0]
    null = np.zeros_like(h_channel)
    h_rgb = hed2rgb(np.stack((h_channel, null, null), axis=-1))

    h_uint8 = img_as_ubyte(np.clip(h_rgb, 0, 1))
    io.imsave(save_path, h_uint8)

def process_mask(mask_img):
    mask_arr = np.array(mask_img)
    if mask_arr.ndim == 3 and mask_arr.shape[2] == 3:
        is_background = np.all(mask_arr == 255, axis=-1)
        binary_mask = (~is_background).astype(np.uint8)
    else:
        raise ValueError("Mask must be RGB with 3 channels")
    return binary_mask

# === STEP 1: Extract Hematoxylin tiles ===
print("=== STEP 1: Extract Hematoxylin tiles ===")
for fname in tqdm(os.listdir(original_folder)):
    if not fname.lower().endswith(valid_extensions):
        continue
    orig_path = os.path.join(original_folder, fname)
    base_name = os.path.splitext(fname)[0]
    h_path = os.path.join(hematoxylin_folder, base_name + "_hematoxylin.png")
    try:
        extract_hematoxylin_tile(orig_path, h_path)
    except Exception as e:
        print(f"[ERROR] Could not extract hematoxylin from {fname}: {e}")

# === STEP 2: Create NPZ files ===
print("=== STEP 2: Create NPZ files ===")

hematoxylin_files = [f for f in os.listdir(hematoxylin_folder) if f.endswith("_hematoxylin.png")]
cleaned_names = []

for hfname in tqdm(hematoxylin_files):
    base_name = hfname[:-len("_hematoxylin.png")]  # remove suffix
    mask_name = base_name + ".png"  # mask has original base name + .png in Labels folder
    mask_path = os.path.join(mask_folder, mask_name)
    h_path = os.path.join(hematoxylin_folder, hfname)

    if not os.path.exists(mask_path):
        print(f"[WARN] Mask missing for {mask_name}, skipping.")
        continue

    try:
        img = Image.open(h_path).convert('RGB').resize((tile_size, tile_size))
        mask = Image.open(mask_path).convert('RGB').resize((tile_size, tile_size))

        img_np = np.array(img) / 255.0
        mask_np = process_mask(mask)

        cleaned_name = clean_filename(base_name)
        cleaned_names.append(cleaned_name)

        npz_path = os.path.join(npz_save_folder, cleaned_name + ".npz")
        np.savez_compressed(npz_path,
                            image=img_np.astype(np.float32),
                            mask=mask_np.astype(np.uint8))
    except Exception as e:
        print(f"[ERROR] Failed to process {base_name}: {e}")

print(f"✅ Processed {len(cleaned_names)} image-mask pairs.")

# === STEP 3: Save infer.txt list ===
with open(list_save_path, 'w') as f:
    for name in cleaned_names:
        f.write(name + "\n")

print(f"📂 Saved infer.txt with {len(cleaned_names)} entries at {list_save_path}")
