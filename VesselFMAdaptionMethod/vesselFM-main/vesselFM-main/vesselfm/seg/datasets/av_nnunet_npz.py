# vesselfm/seg/datasets/av_nnunet_npz.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class AVNnUNetNPZDataset(Dataset):
    """
    Dataset that reads nnUNet-preprocessed .npz cases:
      - image: (C, X, Y, Z), float32
      - label: (1, X, Y, Z), int16
    Optionally applies a MONAI-style transform on the dict.
    """
    def __init__(self, npz_dir, case_ids=None, transform=None):
        self.npz_dir = npz_dir
        self.transform = transform

        if case_ids is None:
            # discover all case_ids from filenames
            self.case_ids = sorted(
                os.path.splitext(os.path.basename(f))[0]
                for f in os.listdir(npz_dir)
                if f.endswith(".npz")
            )
        else:
            self.case_ids = list(case_ids)

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        path = os.path.join(self.npz_dir, f"{case_id}.npz")

        data = np.load(path, allow_pickle=True)
        image = data["image"]  # (C, X, Y, Z)
        label = data["label"]  # (1, X, Y, Z)

        # Convert to torch tensors
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)

        sample = {
            "image": image,
            "label": label,
            "case_id": case_id,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
