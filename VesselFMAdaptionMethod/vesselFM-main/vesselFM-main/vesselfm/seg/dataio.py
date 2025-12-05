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

        # Flag to indicate images are already preprocessed by preprocess_av.py
        self.preprocessed = bool(data_cfg.get("preprocessed", False))

        if self.preprocessed:
            # Images are already resampled + HU-clipped + scaled to [0,1],
            # Do NOT apply any further intensity preprocessing here.
            self.clip_hu = None
            self.zscore = False
        else:
            # Fallback: keep old online intensity preprocessing behavior
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
        D, H, W = image.shape
        pD, pH, pW = self.cfg["data"]["patch_size"]
        min_fg = self.cfg["data"].get("min_fg_fraction", 0.0)
        art_prob = self.cfg["data"].get("artery_patch_prob", 0.0)
        min_art = self.cfg["data"].get("min_artery_fraction", 0.0)
        max_tries = 32

        pD = min(pD, D); pH = min(pH, H); pW = min(pW, W)

        want_artery = (np.random.rand() < art_prob)

        fallback_img = None
        fallback_lab = None

        for attempt in range(max_tries):
            z = np.random.randint(0, max(1, D - pD + 1))
            y = np.random.randint(0, max(1, H - pH + 1))
            x = np.random.randint(0, max(1, W - pW + 1))

            img_patch = image[z:z+pD, y:y+pH, x:x+pW]
            lab_patch = label[z:z+pD, y:y+pH, x:x+pW]

            if fallback_img is None:
                fallback_img, fallback_lab = img_patch, lab_patch

            if min_fg <= 0.0 and not want_artery:
                return img_patch, lab_patch

            fg_fraction = (lab_patch > 0).mean()
            art_fraction = (lab_patch == 1).mean()

            if want_artery:
                if art_fraction >= min_art:
                    return img_patch, lab_patch
            else:
                if fg_fraction >= min_fg:
                    return img_patch, lab_patch

        # fallback if we fail max_tries
        return fallback_img, fallback_lab


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

        # For training: sample patch + augment
        if self.train:
            image, label = self._sample_patch(image, label)
            image, label = self._augment(image, label)
        # For validation: use full volume (no patching, no aug)

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