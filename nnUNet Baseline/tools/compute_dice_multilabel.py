#!/usr/bin/env python3
"""
Compute Dice scores between prediction masks and ground-truth masks for multi-label segmentation.

Usage examples:
  python compute_dice_multilabel.py --pred-dir data/vesselFM/raw \
      --gt-dir data/nnUNet_raw/Dataset001_nnunet/labelsTs --ext nii \
      --labels '{"artery": 1, "vein": 2}'

The script supports .nii, .nii.gz, .png, .jpg, .tif and .npy files.
"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import csv
import re
import json

try:
    import nibabel as nib
except Exception:
    nib = None

try:
    from imageio import imread
except Exception:
    from skimage.io import imread  # type: ignore


def list_mask_files(d: Path, exts=None):
    if exts is None:
        exts = [".nii", ".nii.gz", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy"]
    files = []
    for p in d.rglob("*"):
        if p.is_file():
            for e in exts:
                if str(p.name).lower().endswith(e):
                    files.append(p)
                    break
    return sorted(files)


def stem_without_suffix(p: Path):
    name = p.name
    # handle .nii.gz
    if name.lower().endswith('.nii.gz'):
        return name[:-7]
    return os.path.splitext(name)[0]


def load_mask(path: Path):
    p = str(path)
    low = p.lower()
    if low.endswith('.npy'):
        arr = np.load(p)
    elif low.endswith('.nii') or low.endswith('.nii.gz'):
        if nib is None:
            raise RuntimeError('nibabel is required to load NIfTI files (pip install nibabel)')
        arr = nib.load(p).get_fdata()
    else:
        arr = imread(p)
    arr = np.asarray(arr)
    return arr


def dice_score(a, b):
    """Compute Dice score between two binary masks."""
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    s = a.sum() + b.sum()
    if s == 0:
        return 1.0 
    return 2.0 * inter / s


def dice_score_per_label(pred, gt, label_map):
    scores = {}
    for label_name, label_value in label_map.items():
        pred_binary = (pred == label_value).astype(np.uint8)
        gt_binary = (gt == label_value).astype(np.uint8)
        scores[label_name] = dice_score(pred_binary, gt_binary)
    return scores


def compute_mean_dice(label_scores, exclude_background=True):
    values = []
    for name, score in label_scores.items():
        if exclude_background and name.lower() == 'background':
            continue
        values.append(score)
    if not values:
        return 0.0
    return np.mean(values)


def find_match(pred_path: Path, gt_map: dict):
    base = stem_without_suffix(pred_path)
    # try exact
    if base in gt_map:
        return gt_map[base]

    m = re.match(r"(.+)_\d{1,4}$", base)
    if m:
        cand = m.group(1)
        if cand in gt_map:
            return gt_map[cand]

    for suffix in ['_pred', '-pred', '_mask', '-mask', '_seg', '-seg']:
        if base.endswith(suffix):
            cand = base[: -len(suffix)]
            if cand in gt_map:
                return gt_map[cand]
            m2 = re.match(r"(.+)_\d{1,4}$", cand)
            if m2 and m2.group(1) in gt_map:
                return gt_map[m2.group(1)]

    for k, v in gt_map.items():
        if k in base or base in k:
            return v
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pred-dir', required=True, help='Directory with predicted masks')
    p.add_argument('--gt-dir', required=True, help='Directory with ground-truth masks')
    p.add_argument('--labels', type=str, default='{"artery": 1, "vein": 2}',
                   help='JSON string mapping label names to integer values, e.g. \'{"artery": 1, "vein": 2}\'')
    p.add_argument('--out-csv', default=None, help='Optional CSV path to save per-file scores')
    p.add_argument('--ext', default=None, help='Restrict extensions (e.g. nii or png). If omitted, auto-detects common types')
    p.add_argument('--include-background', action='store_true', 
                   help='Include background (label 0) in mean Dice calculation')
    args = p.parse_args()

    # Parse label mapping
    try:
        label_map = json.loads(args.labels)
        if not isinstance(label_map, dict):
            raise ValueError("Labels must be a JSON object/dict")
        # Validate values are integers
        for k, v in label_map.items():
            if not isinstance(v, int):
                raise ValueError(f"Label value for '{k}' must be an integer, got {type(v).__name__}")
    except json.JSONDecodeError as e:
        print(f'Invalid --labels JSON: {e}', file=sys.stderr)
        sys.exit(3)
    except ValueError as e:
        print(f'Invalid --labels: {e}', file=sys.stderr)
        sys.exit(3)

    print(f"Label mapping: {label_map}")
    print("-" * 60)

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    if args.ext:
        exts = [('.' + args.ext).lower()]
    else:
        exts = None

    preds = list_mask_files(pred_dir, exts)
    gts = list_mask_files(gt_dir, exts)

    if len(preds) == 0:
        print('No prediction files found in', pred_dir, file=sys.stderr)
        sys.exit(2)
    if len(gts) == 0:
        print('No ground-truth files found in', gt_dir, file=sys.stderr)
        sys.exit(2)

    gt_map = {stem_without_suffix(p): p for p in gts}

    # Storage for results
    rows = []  # [(pred_file, gt_file, {label: score}, mean_dice)]
    all_label_scores = {label: [] for label in label_map.keys()}
    all_mean_scores = []
    missed = []

    for pred in preds:
        gt = find_match(pred, gt_map)
        if gt is None:
            missed.append(str(pred))
            continue
        try:
            arr_pred = load_mask(pred)
            arr_gt = load_mask(gt)
        except Exception as e:
            print(f'Error loading {pred} or {gt}: {e}', file=sys.stderr)
            continue

        if arr_pred.shape != arr_gt.shape:
            # try to squeeze singleton dims
            sp = np.squeeze(arr_pred)
            sg = np.squeeze(arr_gt)
            if sp.shape != sg.shape:
                print(f'Shape mismatch for {pred} vs {gt}: {arr_pred.shape} vs {arr_gt.shape}', file=sys.stderr)
                continue
            arr_pred = sp
            arr_gt = sg

        # Convert to integer labels if floating point
        if np.issubdtype(arr_pred.dtype, np.floating):
            arr_pred = np.round(arr_pred).astype(np.int32)
        if np.issubdtype(arr_gt.dtype, np.floating):
            arr_gt = np.round(arr_gt).astype(np.int32)

        # Compute per-label Dice scores
        label_scores = dice_score_per_label(arr_pred, arr_gt, label_map)
        mean_dice = compute_mean_dice(label_scores, exclude_background=not args.include_background)

        rows.append((str(pred), str(gt), label_scores, mean_dice))
        all_mean_scores.append(mean_dice)
        for label, score in label_scores.items():
            all_label_scores[label].append(score)

    # Print per-file results
    for pred_file, gt_file, label_scores, mean_dice in rows:
        pred_name = Path(pred_file).name
        gt_name = Path(gt_file).name
        print(f'{pred_name}  <->  {gt_name}:')
        for label, score in label_scores.items():
            print(f'    {label}: {score:.4f}')
        print(f'    Mean Dice: {mean_dice:.4f}')
        print()

    # Print summary
    if all_mean_scores:
        print('=' * 60)
        print('SUMMARY')
        print('=' * 60)
        print(f'Files compared: {len(all_mean_scores)}')
        print()
        print('Per-label statistics:')
        for label in label_map.keys():
            scores = all_label_scores[label]
            if scores:
                print(f'  {label}:')
                print(f'    Mean:   {np.mean(scores):.4f}')
                print(f'    Std:    {np.std(scores):.4f}')
                print(f'    Min:    {np.min(scores):.4f}')
                print(f'    Max:    {np.max(scores):.4f}')
        print()
        print(f'Overall Mean Dice: {np.mean(all_mean_scores):.4f} (std: {np.std(all_mean_scores):.4f})')

    if missed:
        print('\nWarning: No matching GT for the following prediction files:', file=sys.stderr)
        for m in missed:
            print('  ' + m, file=sys.stderr)

    # Save CSV
    if args.out_csv and rows:
        with open(args.out_csv, 'w', newline='') as f:
            w = csv.writer(f)
            # Header
            header = ['prediction', 'ground_truth'] + list(label_map.keys()) + ['mean_dice']
            w.writerow(header)
            # Data rows
            for pred_file, gt_file, label_scores, mean_dice in rows:
                row = [pred_file, gt_file]
                row += [label_scores[label] for label in label_map.keys()]
                row.append(mean_dice)
                w.writerow(row)
        print(f'\nSaved per-file scores to {args.out_csv}')


if __name__ == '__main__':
    main()