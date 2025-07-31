import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

npz_dir = r"C:\Users\Georges\PycharmProjects\TransUNet\inference_data"
predictions_dir = r"C:\Users\Georges\PycharmProjects\TransUNet\predictions\TU_MyEpithelialDataset_512"
infer_list_file = r"C:\Users\Georges\PycharmProjects\TransUNet\lists\lists_MyEpithelialDataset\infer.txt"

# Load inference list filenames (assume without extension)
with open(infer_list_file, 'r') as f:
    infer_filenames = [line.strip() for line in f.readlines()]

def load_npz_image(npz_path):
    data = np.load(npz_path)
    img_array = data[list(data.files)[0]]  # fallback to first entry
    if img_array.dtype != np.uint8:
        img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array)

def load_prediction_mask(pred_path):
    pred = Image.open(pred_path).convert("L")
    pred_np = np.array(pred)
    binary_pred = (pred_np > 127).astype(np.uint8)
    return binary_pred

def overlay_mask_on_image(image, mask, color=(255, 0, 0), alpha=0.4):
    image = image.convert("RGBA")
    mask_img = Image.new("RGBA", image.size, color + (0,))
    mask_data = mask.astype(np.uint8) * 255
    mask_img.putalpha(Image.fromarray(mask_data))
    mask_img = Image.blend(Image.new("RGBA", image.size, (0, 0, 0, 0)), mask_img, alpha)
    blended = Image.alpha_composite(image, mask_img)
    return blended.convert("RGB")

# Prepare overlays and titles
overlay_images = []
overlay_titles = []

for idx, infer_name in enumerate(infer_filenames):
    base_name = infer_name
    if base_name.endswith('.npz'):
        base_name = base_name[:-4]
    # Remove any trailing underscores just in case
    base_name = base_name.rstrip('_')

    npz_path = os.path.join(npz_dir, base_name + '.npz')

    if not os.path.exists(npz_path):
        print(f"❌ Missing npz file: {npz_path}")
        continue

    pred_filename = f"pred_{idx:03d}.png"
    pred_path = os.path.join(predictions_dir, pred_filename)
    if not os.path.exists(pred_path):
        print(f"❌ Missing prediction mask: {pred_filename}")
        continue

    try:
        image = load_npz_image(npz_path)
        pred_mask = load_prediction_mask(pred_path)
        if pred_mask.shape != (image.height, image.width):
            pred_mask = np.array(Image.fromarray(pred_mask).resize((image.width, image.height), resample=Image.NEAREST))
        overlay_img = overlay_mask_on_image(image, pred_mask)
        overlay_images.append(overlay_img)
        overlay_titles.append(f"{base_name} ←→ {pred_filename}")
    except Exception as e:
        print(f"⚠️ Error with {base_name}: {e}")

if not overlay_images:
    print("No valid overlays to display.")
    exit()

# Interactive Viewer with Arrow Keys
index = 0
fig, ax = plt.subplots()
img_display = ax.imshow(overlay_images[index])
title = ax.set_title(overlay_titles[index])
plt.axis('off')

def on_key(event):
    global index
    if event.key == 'right':
        index = (index + 1) % len(overlay_images)
    elif event.key == 'left':
        index = (index - 1) % len(overlay_images)
    elif event.key == 'q':
        plt.close()
        return
    img_display.set_data(overlay_images[index])
    title.set_text(overlay_titles[index])
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()
