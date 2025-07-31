import os
import numpy as np

npz_dir = 'path/to/your/npz_files'
val_list_path = 'path/to/val.txt'

with open(val_list_path, 'r') as f:
    val_files = [line.strip() for line in f.readlines() if line.strip()]

for fname in val_files:
    if not fname.endswith('.npz'):
        fname += '.npz'
    fpath = os.path.join(npz_dir, fname)
    if not os.path.exists(fpath):
        print(f"File not found: {fpath}")
        continue
    data = np.load(fpath)
    if 'mask' not in data or 'image' not in data:
        print(f"Missing keys in {fname}: {data.files}")
