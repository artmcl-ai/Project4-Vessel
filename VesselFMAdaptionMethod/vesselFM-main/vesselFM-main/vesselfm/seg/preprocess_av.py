#!/usr/bin/env python
"""
Preprocessing for A/V CT dataset.

- Reads NIfTI volumes from an image directory (and optional label directory).
- Optionally resamples to a target spacing.
- Applies HU clipping (e.g. [-1000, 600]).
- Scales intensities to [0, 1].
- Writes out preprocessed images (and labels, if provided) to new directories.

Labels are resampled with nearest-neighbor interpolation and kept as uint8.
"""

import argparse
import pathlib
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", required=True, type=str,
                   help="Input directory with CT images (NIfTI).")
    p.add_argument("--label_dir", type=str, default=None,
                   help="Optional directory with label NIfTIs. "
                        "If given, labels will be resampled and saved too.")
    p.add_argument("--out_image_dir", required=True, type=str,
                   help="Output directory for preprocessed images.")
    p.add_argument("--out_label_dir", type=str, default=None,
                   help="Output directory for preprocessed labels "
                        "(required if label_dir is given).")
    p.add_argument("--target_spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                   help="Target voxel spacing in mm (z y x).")
    p.add_argument("--hu_min", type=float, default=-1000.0,
                   help="Lower HU clip bound.")
    p.add_argument("--hu_max", type=float, default=600.0,
                   help="Upper HU clip bound.")
    return p.parse_args()


def map_label_name(img_name: str) -> str:
    """
    Match the same logic you use in train_av.py:
      - if image name starts with 'image_', use 'label_###.nii.gz'
      - else, assume same filename for label.
    """
    if img_name.startswith("image_"):
        return "label_" + img_name[len("image_"):]
    return img_name


def resample_volume(vol: np.ndarray,
                    orig_spacing,
                    target_spacing,
                    order: int) -> np.ndarray:
    """
    Resample a 3D volume to target_spacing using scipy.ndimage.zoom.
    order=1 for images, order=0 for labels.
    """
    zoom_factors = tuple(os / ts for os, ts in zip(orig_spacing, target_spacing))
    return ndi.zoom(vol, zoom=zoom_factors, order=order)


def preprocess_one(
    img_path: pathlib.Path,
    out_img_path: pathlib.Path,
    label_path: pathlib.Path = None,
    out_label_path: pathlib.Path = None,
    target_spacing=(1.0, 1.0, 1.0),
    hu_min=-1000.0,
    hu_max=600.0,
):
    # --- Image ---
    img_nii = nib.load(str(img_path))
    img = img_nii.get_fdata().astype(np.float32)
    header = img_nii.header.copy()
    affine = img_nii.affine
    orig_spacing = header.get_zooms()[:3]

    # Resample image to target spacing (linear)
    if target_spacing is not None:
        img = resample_volume(img, orig_spacing, target_spacing, order=1)
        # Update spacing in header (approx; keeps orientation)
        header.set_zooms(target_spacing + header.get_zooms()[3:])

    # HU clipping
    img = np.clip(img, hu_min, hu_max)

    # Scale to [0, 1]
    img = (img - hu_min) / float(hu_max - hu_min)
    img = np.clip(img, 0.0, 1.0)

    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(img.astype(np.float32), affine, header),
             str(out_img_path))

    # --- Label (optional) ---
    if label_path is not None and out_label_path is not None:
        lab_nii = nib.load(str(label_path))
        lab = lab_nii.get_fdata()
        lab_header = lab_nii.header.copy()
        lab_affine = lab_nii.affine
        lab_spacing = lab_header.get_zooms()[:3]

        # Resample labels with nearest-neighbor using same factor
        if target_spacing is not None:
            lab = resample_volume(lab, lab_spacing, target_spacing, order=0)
            lab_header.set_zooms(target_spacing + lab_header.get_zooms()[3:])

        lab = lab.astype(np.uint8)
        out_label_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(lab, lab_affine, lab_header),
                 str(out_label_path))


def main():
    args = parse_args()

    image_dir = pathlib.Path(args.image_dir)
    out_img_dir = pathlib.Path(args.out_image_dir)
    label_dir = pathlib.Path(args.label_dir) if args.label_dir is not None else None
    out_lbl_dir = pathlib.Path(args.out_label_dir) if args.out_label_dir is not None else None

    target_spacing = tuple(args.target_spacing)

    if label_dir is not None and out_lbl_dir is None:
        raise ValueError("If --label_dir is given, you must also provide --out_label_dir.")

    img_paths = sorted(image_dir.glob("*.nii*"))
    print(f"Found {len(img_paths)} images in {image_dir}")

    for img_path in img_paths:
        img_name = img_path.name
        out_img_path = out_img_dir / img_name

        if label_dir is not None:
            lbl_name = map_label_name(img_name)
            lab_path = label_dir / lbl_name
            if not lab_path.exists():
                print(f"WARNING: label not found for {img_name} (expected {lab_path}), skipping.")
                continue
            out_label_path = out_lbl_dir / lbl_name
        else:
            lab_path = None
            out_label_path = None

        print(f"Preprocessing {img_name}...")
        preprocess_one(
            img_path=img_path,
            out_img_path=out_img_path,
            label_path=lab_path,
            out_label_path=out_label_path,
            target_spacing=target_spacing,
            hu_min=args.hu_min,
            hu_max=args.hu_max,
        )

    print("Done.")


if __name__ == "__main__":
    main()