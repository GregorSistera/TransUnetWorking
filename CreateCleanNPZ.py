import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image
import argparse


def clean_filename(name):
    return name.replace(" ", "_").replace("[", "_").replace("]", "_").replace("=", "_").replace(",", "_")


def create_npz_dataset(images_dir, masks_dir, output_dir, list_dir, test_size=0.2, seed=42):
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    output_dir = Path(output_dir)
    list_dir = Path(list_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    list_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.tif"))
    print(f"Found {len(image_files)} images.")

    image_mask_pairs = []

    for image_path in image_files:
        base_name = image_path.stem
        cleaned_name = clean_filename(base_name)
        mask_path = masks_dir / f"{base_name}.png"  # or .jpg depending on your mask format
        if not mask_path.exists():
            print(f"Mask not found for image {image_path.name}, skipping.")
            continue
        image_mask_pairs.append((image_path, mask_path, cleaned_name))

    print(f"Found {len(image_mask_pairs)} valid image-mask pairs.")

    train_pairs, val_pairs = train_test_split(image_mask_pairs, test_size=test_size, random_state=seed)

    def save_npz(pairs, split):
        txt_path = list_dir / f"{split}.txt"
        with open(txt_path, "w") as f:
            for image_path, mask_path, cleaned_name in pairs:
                # Load image and mask
                image = np.array(Image.open(image_path).convert("RGB"))
                mask = np.array(Image.open(mask_path).convert("L"))  # Grayscale

                # Normalize mask to 0 or 1 if needed
                mask = (mask > 0).astype(np.uint8)

                # Save to .npz
                npz_path = output_dir / f"{cleaned_name}.npz"
                np.savez_compressed(npz_path, image=image, mask=mask)

                # Save cleaned filename to txt
                f.write(f"{cleaned_name}\n")

        print(f"Saved {len(pairs)} {split} samples to {txt_path}")

    save_npz(train_pairs, "train")
    save_npz(val_pairs, "val")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True, help="Path to original images folder")
    parser.add_argument("--masks_dir", required=True, help="Path to corresponding masks folder")
    parser.add_argument("--output_dir", required=True, help="Output directory to save .npz files")
    parser.add_argument("--list_dir", required=True, help="Directory to save train.txt and val.txt")
    args = parser.parse_args()

    create_npz_dataset(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        output_dir=args.output_dir,
        list_dir=args.list_dir
    )
