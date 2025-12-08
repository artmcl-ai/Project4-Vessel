#!/usr/bin/env python

import os
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import zoom


def resample_to_spacing(data, src_spacing, tgt_spacing, order):
    """
    data: np.ndarray with shape (X, Y, Z) or (X, Y, Z, C)
    src_spacing, tgt_spacing: iterable of length 3
    order: interpolation order (3 = cubic for image, 0 = nearest for label)
    """
    src_spacing = np.array(src_spacing, dtype=np.float32)
    tgt_spacing = np.array(tgt_spacing, dtype=np.float32)
    zoom_factors = src_spacing / tgt_spacing

    if data.ndim == 3:
        factors = zoom_factors
    elif data.ndim == 4:
        factors = (*zoom_factors, 1.0)  # don't scale channels
    else:
        raise ValueError(f"Unsupported data ndim {data.ndim}, expected 3 or 4.")

    resampled = zoom(data, factors, order=order)
    return resampled


def compute_label_bbox(label, margin=0):
    """
    Compute bounding box of non-zero labels, with margin in voxels.
    label: np.ndarray (X, Y, Z) integer labels
    returns: slices or None if no foreground
    """
    if np.max(label) == 0:
        return None

    coords = np.where(label > 0)
    xmin, xmax = coords[0].min(), coords[0].max()
    ymin, ymax = coords[1].min(), coords[1].max()
    zmin, zmax = coords[2].min(), coords[2].max()

    xmin = max(xmin - margin, 0)
    ymin = max(ymin - margin, 0)
    zmin = max(zmin - margin, 0)

    xmax = min(xmax + margin, label.shape[0] - 1)
    ymax = min(ymax + margin, label.shape[1] - 1)
    zmax = min(zmax + margin, label.shape[2] - 1)

    return (slice(xmin, xmax + 1),
            slice(ymin, ymax + 1),
            slice(zmin, zmax + 1))


def zscore_normalize(img, mask=None, eps=1e-8):
    """
    Z-score normalization with mask.
    img: np.ndarray float
    mask: boolean array or None
    """
    if mask is None:
        mask = np.ones_like(img, dtype=bool)
    vals = img[mask]
    if vals.size == 0:
        return img
    mean = vals.mean()
    std = vals.std()
    if std < eps:
        std = eps
    img = (img - mean) / std
    return img


def percentile_normalize(img, p_lo=0.5, p_hi=99.5, mask=None, eps=1e-8):
    """
    Map intensities between [p_lo, p_hi] percentiles to [0,1].
    Everything below p_lo goes to 0, above p_hi to 1.
    Usually mask = (img != 0) to ignore the air background.
    """
    if mask is None:
        mask = np.ones_like(img, dtype=bool)

    vals = img[mask]
    if vals.size == 0:
        return img

    lo = np.percentile(vals, p_lo)
    hi = np.percentile(vals, p_hi)

    if hi - lo < eps:
        # almost constant volume, nothing sensible to do
        return img

    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + eps)
    return img


def preprocess_pair(
    img_path,
    lbl_path,
    out_img_dir,
    out_lbl_dir,
    target_spacing,
    clip_range=None,
    do_zscore=False,
    crop_mode="label",
    crop_margin=10,
    percentile_norm=False,
    percentile_range=(0.5, 99.5),
):
    """
    Preprocess one image+label pair and save as .nii.gz.

    img_path, lbl_path: Path objects (input .nii/.nii.gz)
    out_img_dir, out_lbl_dir: Path objects (output directories)
    """
    print(f"Processing: {img_path.name}")

    img_nii = nib.load(str(img_path))
    img = img_nii.get_fdata().astype(np.float32)
    src_spacing = img_nii.header.get_zooms()[:3]

    # Labels
    if lbl_path is not None:
        lbl_nii = nib.load(str(lbl_path))
        lbl = lbl_nii.get_fdata().astype(np.int16)
    else:
        lbl_nii = None
        lbl = None

    # Resample image and label to target spacing
    if target_spacing is not None:
        img = resample_to_spacing(img, src_spacing, target_spacing, order=3)
        if lbl is not None:
            lbl = resample_to_spacing(lbl, src_spacing, target_spacing, order=0)
        spacing = target_spacing
    else:
        spacing = src_spacing

    # Intensity normalisation
    # HU clipping
    if clip_range is not None:
        lo, hi = clip_range
    else:
        lo, hi = -1000.0, 600.0  # default chest CT window

    img = np.clip(img, lo, hi)

    # Choose one of: percentile mapping, z-score, or simple [lo,hi] to [0,1]
    if percentile_norm:
        # Ignore pure-air voxels when computing percentiles
        mask = img != lo
        p_lo, p_hi = percentile_range
        img = percentile_normalize(img, p_lo=p_lo, p_hi=p_hi, mask=mask)
    elif do_zscore:
        mask = img != lo
        img = zscore_normalize(img, mask=mask)
    else:
        # Simple linear windowing [lo,hi] to [0,1]
        img = (img - lo) / (hi - lo + 1e-8)

    img = img.astype(np.float32)

    # Cropping
    # For inference, use crop_mode="none" so volume size stays global.
    if crop_mode == "label" and lbl is not None:
        bbox = compute_label_bbox(lbl, margin=crop_margin)
        if bbox is not None:
            img = img[bbox]
            lbl = lbl[bbox]
    elif crop_mode == "none":
        # No cropping
        pass
    elif crop_mode == "body":
        # Simple body mask
        mask = img != 0
        if np.any(mask):
            coords = np.where(mask)
            xmin, xmax = coords[0].min(), coords[0].max()
            ymin, ymax = coords[1].min(), coords[1].max()
            zmin, zmax = coords[2].min(), coords[2].max()
            xmin = max(xmin - crop_margin, 0)
            ymin = max(ymin - crop_margin, 0)
            zmin = max(zmin - crop_margin, 0)
            xmax = min(xmax + crop_margin, img.shape[0] - 1)
            ymax = min(ymax + crop_margin, img.shape[1] - 1)
            zmax = min(zmax + crop_margin, img.shape[2] - 1)
            bbox = (slice(xmin, xmax + 1),
                    slice(ymin, ymax + 1),
                    slice(zmin, zmax + 1))
            img = img[bbox]
            if lbl is not None:
                lbl = lbl[bbox]
    else:
        raise ValueError(f"Unknown crop_mode: {crop_mode}")

    # Build a simple affine with the (possibly new) spacing
    affine = np.eye(4, dtype=np.float32)
    affine[0, 0] = spacing[0]
    affine[1, 1] = spacing[1]
    affine[2, 2] = spacing[2]

    out_img_dir.mkdir(parents=True, exist_ok=True)
    if out_lbl_dir is not None:
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

    out_img_path = out_img_dir / img_path.name
    img_out = nib.Nifti1Image(img.astype(np.float32), affine)
    nib.save(img_out, str(out_img_path))

    if lbl is not None and out_lbl_dir is not None:
        out_lbl_path = out_lbl_dir / lbl_path.name
        lbl_out = nib.Nifti1Image(lbl.astype(np.int16), affine)
        nib.save(lbl_out, str(out_lbl_path))

    print(f"  -> saved image to {out_img_path}")
    if lbl is not None and out_lbl_dir is not None:
        print(f"  -> saved label to {out_lbl_path}")


def preprocess_dataset(
    images_dir,
    labels_dir=None,
    out_images_dir=None,
    out_labels_dir=None,
    target_spacing=(1.0, 1.0, 1.0),
    clip_range=None,
    do_zscore=False,
    crop_mode="label",
    crop_margin=10,
    percentile_norm=False,
    percentile_range=(0.5, 99.5),
):
    """
    Programmatic entry point: preprocess all images (and labels if provided)
    in a directory.

    images_dir: path-like
    labels_dir: path-like or None
    out_images_dir: path-like
    out_labels_dir: path-like or None
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir) if labels_dir is not None else None
    out_images_dir = Path(out_images_dir)
    out_labels_dir = Path(out_labels_dir) if out_labels_dir is not None else None

    if labels_dir is None and crop_mode == "label":
        raise ValueError("crop_mode='label' requires labels_dir to be provided.")

    img_files = sorted(
        [
            p
            for p in images_dir.iterdir()
            if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz"))
        ]
    )

    if not img_files:
        raise RuntimeError(f"No NIfTI files found in {images_dir}")

    for img_path in img_files:
        # Strip extension to get a clean base name image_001
        img_name = img_path.name
        base = img_name
        if base.endswith(".nii.gz"):
            base = base[:-7]
        elif base.endswith(".nii"):
            base = base[:-4]

        lbl_path = None
        if labels_dir is not None:
            candidates = []

            # 1) Same base name in labels dir
            candidates.append(labels_dir / (base + ".nii.gz"))
            candidates.append(labels_dir / (base + ".nii"))

            # 2) image_001 to label_001 pattern
            if base.startswith("image_"):
                idx = base[len("image_"):]  # "001"
                candidates.append(labels_dir / f"label_{idx}.nii.gz")
                candidates.append(labels_dir / f"label_{idx}.nii")

            # Pick the first existing candidate
            for cand in candidates:
                if cand.exists():
                    lbl_path = cand
                    break

            if lbl_path is None:
                raise FileNotFoundError(
                    f"Missing label for {img_path.name}. "
                    f"Tried: {[str(c) for c in candidates]}"
                )

        preprocess_pair(
            img_path=img_path,
            lbl_path=lbl_path,
            out_img_dir=out_images_dir,
            out_lbl_dir=out_labels_dir,
            target_spacing=target_spacing,
            clip_range=clip_range,
            # Don't z-score if doing percentile_norm
            do_zscore=do_zscore and not percentile_norm,
            crop_mode=crop_mode,
            crop_margin=crop_margin,
            percentile_norm=percentile_norm,
            percentile_range=tuple(percentile_range),
        )


def cli_main():
    parser = argparse.ArgumentParser(
        description="Preprocessing for VesselFM Adaptation"
    )
    parser.add_argument("--images", type=str, required=True,
                        help="Directory with input .nii/.nii.gz images")
    parser.add_argument("--labels", type=str, default=None,
                        help="Directory with input .nii/.nii.gz labels (optional)")
    parser.add_argument("--out_images", type=str, required=True,
                        help="Output directory for preprocessed images")
    parser.add_argument("--out_labels", type=str, default=None,
                        help="Output directory for preprocessed labels (optional)")
    parser.add_argument(
        "--target_spacing",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        help="Target voxel spacing (sx sy sz). Use 1.0 1.0 1.0 or dataset median."
    )
    parser.add_argument(
        "--percentile_norm",
        action="store_true",
        help="Use per-volume percentile mapping after HU clipping instead of plain [lo,hi]->[0,1] or z-score."
    )
    parser.add_argument(
        "--percentile_range",
        type=float,
        nargs=2,
        default=[0.5, 99.5],
        help="Lower/upper percentiles for --percentile_norm (default 0.5 99.5)."
    )
    parser.add_argument(
        "--clip",
        type=float,
        nargs=2,
        default=None,
        help="Intensity clip range, e.g. --clip -1000 600 for CT HU."
    )
    parser.add_argument(
        "--zscore",
        action="store_true",
        help="Apply z-score normalization after clipping."
    )
    parser.add_argument(
        "--crop_mode",
        type=str,
        default="label",
        choices=["none", "label", "body"],
        help="Cropping mode: 'label' (crop to label bbox), 'body' (non-zero img), or 'none'."
    )
    parser.add_argument(
        "--crop_margin",
        type=int,
        default=10,
        help="Margin (in voxels) to add around the crop bounding box."
    )

    args = parser.parse_args()

    preprocess_dataset(
        images_dir=args.images,
        labels_dir=args.labels,
        out_images_dir=args.out_images,
        out_labels_dir=args.out_labels,
        target_spacing=tuple(args.target_spacing),
        clip_range=args.clip,
        do_zscore=args.zscore,
        crop_mode=args.crop_mode,
        crop_margin=args.crop_margin,
        percentile_norm=args.percentile_norm,
        percentile_range=tuple(args.percentile_range),
    )


if __name__ == "__main__":
    cli_main()