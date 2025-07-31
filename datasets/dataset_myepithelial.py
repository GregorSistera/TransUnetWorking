import os
import numpy as np
import torch
from torch.utils.data import Dataset

class EpithelialNPZDataset(Dataset):
    def __init__(self, npz_dir, list_file, transform=None):
        """
        npz_dir: directory with .npz files
        list_file: txt file listing npz filenames (one per line)
        transform: albumentations transform (image and mask)
        """
        self.npz_dir = npz_dir
        with open(list_file, 'r') as f:
            self.file_list = [line.strip() for line in f.readlines()]
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        if not filename.endswith('.npz'):
            filename += '.npz'

        data = np.load(os.path.join(self.npz_dir, filename))
        image = data['image']  # HWC
        mask = data['mask']  # HW, values might be 0 and 255

        # ✅ Normalize mask values to 0 and 1
        mask = (mask > 0).astype(np.uint8)  # Ensures mask only contains 0 and 1

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        else:
            image = image.float()
            if image.max() > 1.0:
                image = image / 255.0
            if image.shape[0] != 3 and image.dim() == 3 and image.shape[-1] == 3:
                image = image.permute(2, 0, 1)

        # ✅ Ensure mask is torch tensor and long
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()

        return image, mask


