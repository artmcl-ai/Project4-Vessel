import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset


class NiftiVolume(Dataset):
    """
    Simple NIfTI dataset that:
    - loads image/label volumes from disk,
    - applies basic CT preprocessing (HU clipping + scaling),
    - samples 3D patches for training,
    - returns tensors in MONAI / DynUNet-friendly format:
        image: (1, D, H, W), float32
        label: (D, H, W),     long (class indices 0,1,2)
    """

    def __init__(self, items, cfg, train: bool = True):
        """
        Args:
            items: list of (image_path, label_path) tuples.
            cfg:   the full av_ct.yaml config (dict-like).
            train: True for training, False for validation.
        """
        self.items = items
        self.cfg = cfg
        self.train = train

        data_cfg = cfg.get("data", {})

        # Patch sampling / preprocessing hyperparams (from av_ct.yaml)
        self.patch_size = tuple(data_cfg.get("patch_size", [96, 96, 96]))
        self.samples_per_volume = int(data_cfg.get("samples_per_volume", 4))
        self.min_fg_fraction = float(data_cfg.get("min_fg_fraction", 0.0))
        self.clip_hu = data_cfg.get("clip_hu", None)   # e.g. [-1000, 600]
        self.zscore = bool(data_cfg.get("zscore", False))

        # For training, we define length as (#volumes * samples_per_volume)
        # so each epoch sees multiple patches per volume.
        if self.train:
            self.length = len(self.items) * self.samples_per_volume
        else:
            # For validation, one sample per volume (still patch-based).
            self.length = len(self.items)

        # Simple in-memory cache so we don't reload NIfTI every time
        self._cache = {}  # img_path -> (image_array, label_array)

        # Optional transform hook (kept to match train_av.py API)
        self._transform = None

    def set_transform(self, transform):
        """Keep a hook if you later want to plug extra transforms."""
        self._transform = transform

    def __len__(self):
        return self.length

    # ---- Helpers ----

    def _load_image_label(self, vol_idx):
        img_path, lab_path = self.items[vol_idx]

        if img_path in self._cache:
            return self._cache[img_path]

        img_nii = nib.load(img_path)
        lab_nii = nib.load(lab_path)

        image = img_nii.get_fdata().astype(np.float32)
        label = lab_nii.get_fdata().astype(np.int16)

        # Intensity preprocessing: HU clipping + scaling to [0,1]
        if self.clip_hu is not None:
            lo, hi = float(self.clip_hu[0]), float(self.clip_hu[1])
            image = np.clip(image, lo, hi)
            image = (image - lo) / (hi - lo + 1e-8)

        # Optional z-score after HU scaling (if enabled)
        if self.zscore:
            m = image.mean()
            s = image.std()
            if s > 0:
                image = (image - m) / s

        self._cache[img_path] = (image, label)
        return image, label

    def _sample_patch(self, image, label):
        """
        Randomly sample a 3D patch of size patch_size.
        For training, we try to enforce a minimum vessel foreground fraction.
        For validation, we just take a random patch (no fg constraint).
        """
        dz, dy, dx = self.patch_size
        zdim, ydim, xdim = image.shape

        # Pad if volume is smaller than patch
        pad_z = max(0, dz - zdim)
        pad_y = max(0, dy - ydim)
        pad_x = max(0, dx - xdim)
        if pad_z or pad_y or pad_x:
            image = np.pad(
                image,
                ((0, pad_z), (0, pad_y), (0, pad_x)),
                mode="constant",
            )
            label = np.pad(
                label,
                ((0, pad_z), (0, pad_y), (0, pad_x)),
                mode="constant",
            )
            zdim, ydim, xdim = image.shape

        # Try several times to get a vessel-rich patch (for training)
        for _ in range(10):
            z0 = np.random.randint(0, zdim - dz + 1)
            y0 = np.random.randint(0, ydim - dy + 1)
            x0 = np.random.randint(0, xdim - dx + 1)

            patch_lab = label[z0:z0 + dz, y0:y0 + dy, x0:x0 + dx]

            if not self.train:
                patch_img = image[z0:z0 + dz, y0:y0 + dy, x0:x0 + dx]
                return patch_img, patch_lab

            if self.min_fg_fraction <= 0:
                patch_img = image[z0:z0 + dz, y0:y0 + dy, x0:x0 + dx]
                return patch_img, patch_lab

            fg_fraction = (patch_lab > 0).mean()
            if fg_fraction >= self.min_fg_fraction:
                patch_img = image[z0:z0 + dz, y0:y0 + dy, x0:x0 + dx]
                return patch_img, patch_lab

        # Fallback: last sampled patch, even if mostly background
        patch_img = image[z0:z0 + dz, y0:y0 + dy, x0:x0 + dx]
        patch_lab = label[z0:z0 + dz, y0:y0 + dy, x0:x0 + dx]
        return patch_img, patch_lab

    def _augment(self, image, label):
        """Very lightweight spatial augmentation (flips)."""
        if not self.train:
            return image, label

        # Random flips along each axis
        if np.random.rand() < 0.5:
            image = image[::-1, :, :]
            label = label[::-1, :, :]
        if np.random.rand() < 0.5:
            image = image[:, ::-1, :]
            label = label[:, ::-1, :]
        if np.random.rand() < 0.5:
            image = image[:, :, ::-1]
            label = label[:, :, ::-1]

        return image, label

    def __getitem__(self, idx):
        # Map global index to volume index
        if self.train:
            vol_idx = idx // self.samples_per_volume
        else:
            vol_idx = idx
        vol_idx = int(vol_idx % len(self.items))

        # Load and preprocess volume
        image, label = self._load_image_label(vol_idx)

        # Sample patch + augment (if training)
        image, label = self._sample_patch(image, label)
        image, label = self._augment(image, label)

        sample = {"image": image, "label": label}

        if self._transform is not None:
            sample = self._transform(sample)
            image = sample["image"]
            label = sample["label"]

        # Ensure positive strides / contiguous arrays
        image = np.ascontiguousarray(image)
        label = np.ascontiguousarray(label)

        # Convert to tensors, channel-first for image
        img_tensor = torch.as_tensor(image[None, ...], dtype=torch.float32)  # (1, D, H, W)
        lab_tensor = torch.as_tensor(label, dtype=torch.long)                # (D, H, W)

        return {"image": img_tensor, "label": lab_tensor}



def make_aug_transforms(cfg, train: bool = True):
    """
    Placeholder hook for additional transforms.

    """
    def _identity(sample):
        return sample

    return _identity