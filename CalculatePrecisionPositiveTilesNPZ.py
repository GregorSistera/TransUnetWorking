import os
import difflib
from PIL import Image
import numpy as np

def find_closest_filename(expected_name, file_list, cutoff=0.6):
    matches = difflib.get_close_matches(expected_name, file_list, n=1, cutoff=cutoff)
    return matches[0] if matches else None

def load_ground_truth_mask_from_npz(npz_path):
    try:
        data = np.load(npz_path)
        # Adjust this according to your npz structure, e.g., 'mask' key
        mask = data['mask']
        binary_mask = (mask > 0).astype(np.uint8)
        return binary_mask
    except Exception as e:
        print(f"❌ Error loading GT mask from {npz_path}: {e}")
        return None

def load_prediction_mask(pred_path):
    pred = Image.open(pred_path).convert("L")
    pred_np = np.array(pred)
    binary_pred = (pred_np > 127).astype(np.uint8)
    return binary_pred

def calculate_precision(gt_mask, pred_mask):
    tp = np.logical_and(pred_mask == 1, gt_mask == 1).sum()
    fp = np.logical_and(pred_mask == 1, gt_mask == 0).sum()
    if (tp + fp) == 0:
        return None
    return tp / (tp + fp)

# --- Directories ---
npz_data_dir = r"C:\Users\Georges\Desktop\IHCEPITHELIALSEGDATASET\npz_data\train_npz"
prediction_dir = r"C:\Users\Georges\PycharmProjects\TransUNet\predictions\TU_MyEpithelialDataset_512"
infer_list_file = r"C:\Users\Georges\PycharmProjects\TransUNet\lists\lists_MyEpithelialDataset\infer.txt"

# Load the infer.txt file (filenames without underscores and extensions)
with open(infer_list_file, 'r') as f:
    infer_filenames = [line.strip() for line in f.readlines()]

npz_files_actual = os.listdir(npz_data_dir)
prediction_files_actual = os.listdir(prediction_dir)

precision_scores = []
skipped_bg_only = 0
skipped_no_prediction = 0

for idx, infer_name in enumerate(infer_filenames):
    # Expected npz filename has a trailing underscore before .npz, e.g. 'xxx_..._.npz'
    expected_npz_name = infer_name + "_.npz"
    closest_npz = find_closest_filename(expected_npz_name, npz_files_actual)

    if not closest_npz:
        print(f"❌ Missing npz file: {expected_npz_name}")
        continue
    npz_path = os.path.join(npz_data_dir, closest_npz)

    gt_mask = load_ground_truth_mask_from_npz(npz_path)
    if gt_mask is None:
        continue

    # Skip background-only tiles
    if gt_mask.sum() == 0:
        skipped_bg_only += 1
        continue

    # Prediction filename (e.g. pred_000.png)
    pred_filename = f"pred_{idx:03d}.png"
    if pred_filename not in prediction_files_actual:
        print(f"❌ Missing prediction file: {pred_filename}")
        continue
    pred_path = os.path.join(prediction_dir, pred_filename)

    pred_mask = load_prediction_mask(pred_path)

    # Skip tiles with no prediction
    if pred_mask.sum() == 0:
        skipped_no_prediction += 1
        continue

    precision = calculate_precision(gt_mask, pred_mask)
    if precision is not None:
        precision_scores.append(precision)
        print(f"{pred_filename}: Precision = {precision:.4f}")

if precision_scores:
    mean_precision = sum(precision_scores) / len(precision_scores)
    print(f"\n✅ Mean precision on positive predicted tiles: {mean_precision:.4f}")
    print(f"🔹 Tiles evaluated: {len(precision_scores)}")
    print(f"🔸 Skipped (GT background-only): {skipped_bg_only}")
    print(f"🔸 Skipped (no prediction): {skipped_no_prediction}")
else:
    print("⚠️ No tiles qualified for precision evaluation.")
