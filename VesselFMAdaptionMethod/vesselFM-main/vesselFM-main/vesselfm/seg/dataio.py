import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from scipy.ndimage import rotate as ndi_rotate, zoom as ndi_zoom


def _center_crop_or_pad(arr, target_shape):
    """
    Center-crop or pad a 3D array to target_shape = (D,H,W).
    Pads with zeros if arr is smaller; crops centrally if larger.
    """
    out = np.zeros(target_shape, dtype=arr.dtype)

    in_shape = arr.shape
    in_slices = []
    out_slices = []

    for in_size, out_size in zip(in_shape, target_shape):
        if in_size >= out_size:
            # crop in the center
            start_in = (in_size - out_size) // 2
            end_in = start_in + out_size
            start_out = 0
            end_out = out_size
        else:
            # pad in the center
            start_in = 0
            end_in = in_size
            start_out = (out_size - in_size) // 2
            end_out = start_out + in_size

        in_slices.append(slice(start_in, end_in))
        out_slices.append(slice(start_out, end_out))

    out[tuple(out_slices)] = arr[tuple(in_slices)]
    return out


def rotate_3d(image, label, angle_deg, axis="z"):
    """
    Small 3D rotation around one axis.
    image, label: (D,H,W)
    axis: 'x', 'y', or 'z' (CT-wise, 'z' = axial plane rotation).
    """
    if axis == "z":
        axes = (1, 2)  # rotate in (H,W)
    elif axis == "y":
        axes = (0, 2)
    elif axis == "x":
        axes = (0, 1)
    else:
        raise ValueError(f"Unknown axis {axis}, expected 'x','y','z'.")

    img_rot = ndi_rotate(
        image, angle_deg, axes=axes, reshape=False,
        order=1, mode="nearest"
    )
    lab_rot = ndi_rotate(
        label, angle_deg, axes=axes, reshape=False,
        order=0, mode="nearest"
    ).astype(label.dtype)
    return img_rot, lab_rot


def zoom_3d(image, label, zoom_factor):
    """
    Isotropic zoom in 3D, then center-crop / pad back to original shape.
    """
    orig_shape = image.shape

    img_zoom = ndi_zoom(image, zoom_factor, order=1)
    lab_zoom = ndi_zoom(label, zoom_factor, order=0).astype(label.dtype)

    img_zoom = _center_crop_or_pad(img_zoom, orig_shape)
    lab_zoom = _center_crop_or_pad(lab_zoom, orig_shape)

    return img_zoom, lab_zoom


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

        # A/V-specific sampling hyperparams
        self.artery_prob = float(data_cfg.get("artery_patch_prob", 0.0))
        self.vein_prob   = float(data_cfg.get("vein_patch_prob", 0.0))
        self.mixed_prob  = float(data_cfg.get("mixed_av_patch_prob", 0.0))
        self.min_artery_fraction = float(data_cfg.get("min_artery_fraction", 0.0))
        self.min_vein_fraction   = float(data_cfg.get("min_vein_fraction", 0.0))


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

        # Augmentation hyperparameters from config av_ct.yaml
        aug_cfg = cfg.get("augment", {})

        # Flip probability per axis
        self.flip_prob = float(aug_cfg.get("flip_prob", 0.5))

        # Small-angle rotation & zoom probabilities
        self.rotate_prob = float(aug_cfg.get("rotate_prob", 0.3))
        self.zoom_prob = float(aug_cfg.get("zoom_prob", 0.3))

        # Gaussian noise std (on image only, after all geometric augs)
        self.noise_std = float(aug_cfg.get("noise_std", 0.0))


    def set_transform(self, transform):
        """Keep a hook if you later want to plug extra transforms."""
        self._transform = transform

    def __len__(self):
        return self.length

    # Helpers

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
        pD, pH, pW = self.patch_size
        min_fg      = self.min_fg_fraction
        art_prob    = self.artery_prob
        mixed_prob  = self.mixed_prob
        min_art     = self.min_artery_fraction
        min_vein = self.min_vein_fraction
        max_tries   = 32

        # Never request a patch bigger than the volume
        pD = min(pD, D); pH = min(pH, H); pW = min(pW, W)

        # Turn (artery_prob, vein_prob, mixed_prob) into a proper distribution
        probs = np.array(
            [self.artery_prob, self.vein_prob, self.mixed_prob],
            dtype=float,
        )
        if probs.sum() > 0:
            probs = probs / probs.sum()
            modes = ["art", "vein", "mix"]
        else:
            # No special A/V preference, just foreground-biased
            probs = None
            modes = ["fg"]

        fallback_img = None
        fallback_lab = None

        for _ in range(max_tries):
            # Random crop coords
            z = np.random.randint(0, max(1, D - pD + 1))
            y = np.random.randint(0, max(1, H - pH + 1))
            x = np.random.randint(0, max(1, W - pW + 1))

            img_patch = image[z:z+pD, y:y+pH, x:x+pW]
            lab_patch = label[z:z+pD, y:y+pH, x:x+pW]

            if fallback_img is None:
                fallback_img, fallback_lab = img_patch, lab_patch

            # Decide what type of patch we want
            if probs is not None:
                mode = np.random.choice(modes, p=probs)
            else:
                mode = "fg"

            fg_fraction   = (lab_patch > 0).mean()
            art_fraction  = (lab_patch == 1).mean()
            vein_fraction = (lab_patch == 2).mean()

            if mode == "mix":
                # want both artery and vein
                if art_fraction >= min_art and vein_fraction >= min_vein:
                    return img_patch, lab_patch
            elif mode == "art":
                if art_fraction >= min_art:
                    return img_patch, lab_patch
            elif mode == "vein":
                # reuse min_art as min_vein_fraction; you can split later if needed
                if vein_fraction >= min_vein:
                    return img_patch, lab_patch
            else:  # "fg" (fallback foreground-biased)
                if fg_fraction >= min_fg:
                    return img_patch, lab_patch

        # Fallback if constraints not met
        return fallback_img, fallback_lab



    def _augment(self, image, label):
        """Lightweight spatial + intensity augmentation."""
        if not self.train:
            return image, label

        # Random flips along each axis
        if np.random.rand() < self.flip_prob:
            image = image[::-1, :, :]
            label = label[::-1, :, :]
        if np.random.rand() < self.flip_prob:
            image = image[:, ::-1, :]
            label = label[:, ::-1, :]
        if np.random.rand() < self.flip_prob:
            image = image[:, :, ::-1]
            label = label[:, :, ::-1]

        # Small random rotation around z-axis (axial plane)
        if np.random.rand() < self.rotate_prob:
            angle = np.random.uniform(-7.0, 7.0)  # degrees
            image, label = rotate_3d(
                image,
                label,
                angle_deg=angle,
                axis="z",
            )

        # Small isotropic zoom
        if np.random.rand() < self.zoom_prob:
            zoom_factor = np.random.uniform(0.9, 1.1)
            image, label = zoom_3d(
                image,
                label,
                zoom_factor,
            )

        # Add Gaussian noise on image only
        if self.noise_std > 0.0 and np.random.rand() < 0.5:
            noise = np.random.normal(0.0, self.noise_std, size=image.shape).astype(image.dtype)
            image = image + noise

            # If your preprocessed intensities are in [0,1], keep them there:
            image = np.clip(image, 0.0, 1.0)

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