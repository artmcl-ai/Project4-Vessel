#!/usr/bin/env python3

import os
import json
import math
import csv
from pathlib import Path

import numpy as np
import nibabel as nib
from skimage.morphology import skeletonize


PRED_DIR = Path("/projectnb/ec500kb/projects/Fall_2025_Projects/Project_4_VesselFM/data/nnUNet_results/Dataset003_vesselfm_good/preds/seed_1")
GT_DIR   = Path("/projectnb/ec500kb/projects/Fall_2025_Projects/Project_4_VesselFM/data/nnUNet_raw/Dataset003_vesselfm_good/labelsTs")
OUT_DIR  = PRED_DIR / "cldice_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

USE_BBOX_CROP = True

EPS = 1e-8
CLASS_NAMES = {1: "Artery", 2: "Vein"}



def countnz(x) -> int:
    return int(np.count_nonzero(x))


def cl_score(skeleton: np.ndarray, volume: np.ndarray) -> float:
    """Centerline score: coverage of skeleton within volume"""
    denom = countnz(skeleton)
    if denom == 0:
        return 0.0
    inter = countnz(skeleton & volume.astype(bool))
    return inter / (denom + EPS)


def cldice_binary(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    """clDice for binary 3D masks; """
    pred_bool = pred_bin.astype(bool)
    gt_bool   = gt_bin.astype(bool)

    if not pred_bool.any() or not gt_bool.any():
        return float('nan') if (not pred_bool.any() and not gt_bool.any()) else 0.0

    if USE_BBOX_CROP:
        fg = pred_bool | gt_bool
        if fg.any():
            zz, yy, xx = np.where(fg)
            z0, z1 = zz.min(), zz.max() + 1
            y0, y1 = yy.min(), yy.max() + 1
            x0, x1 = xx.min(), xx.max() + 1
            pred_bool = pred_bool[z0:z1, y0:y1, x0:x1]
            gt_bool   = gt_bool[z0:z1, y0:y1, x0:x1]

    # Change skeletonize_3d to skeletonize
    pred_skel = skeletonize(pred_bool)
    gt_skel   = skeletonize(gt_bool)

    tprec = cl_score(pred_skel, gt_bool) 
    tsens = cl_score(gt_skel,   pred_bool) 

    if (tprec + tsens) == 0:
        return 0.0
    return 2.0 * tprec * tsens / (tprec + tsens + EPS)


def load_label(path: Path) -> np.ndarray:
    return nib.load(str(path)).get_fdata()


def summarize(scores: list[float]) -> dict:
    vals = [s for s in scores if not (isinstance(s, float) and math.isnan(s))]
    if not vals:
        return dict(mean=float('nan'), std=float('nan'),
                    median=float('nan'), vmin=float('nan'), vmax=float('nan'), n=0)
    arr = np.array(vals, dtype=float)
    return dict(mean=float(arr.mean()), std=float(arr.std(ddof=0)),
                median=float(np.median(arr)), vmin=float(arr.min()), vmax=float(arr.max()), n=len(arr))


def main():
    if not PRED_DIR.exists():
        print(f"ERROR: Prediction folder does not exist: {PRED_DIR}")
        return 1
    if not GT_DIR.exists():
        print(f"ERROR: Ground truth folder does not exist: {GT_DIR}")
        return 1

    print("Vessel Segmentation clDice (3D) Evaluation")
    print("Labels: 0=Background, 1=Artery, 2=Vein")
    print(f"Pred dir: {PRED_DIR}")
    print(f"GT   dir: {GT_DIR}")
    print("=" * 72)

    pred_files = sorted([p for p in PRED_DIR.glob("*.nii.gz")])
    if not pred_files:
        print(f"ERROR: No .nii.gz files in {PRED_DIR}")
        return 1

    results: list[dict] = []

    for pred_path in pred_files:
        case = pred_path.name
        gt_path = GT_DIR / case
        if not gt_path.exists():
            print(f"WARNING: Missing GT for {case}, skip.")
            continue

        pred = load_label(pred_path)
        gt   = load_label(gt_path)

        if pred.shape != gt.shape:
            print(f"WARNING: Shape mismatch for {case}: pred{pred.shape} vs gt{gt.shape}, skip.")
            continue

        row = {"case": case}

        for lid, lname in CLASS_NAMES.items():
            pred_bin = (pred == lid)
            gt_bin   = (gt == lid)

            pred_vox = countnz(pred_bin)
            gt_vox   = countnz(gt_bin)

            score = cldice_binary(pred_bin, gt_bin)
            row[f"clDice_{lname}"] = float(score)
            row[f"{lname}_pred_voxels"] = int(pred_vox)
            row[f"{lname}_gt_voxels"] = int(gt_vox)

        pred_fg = (pred > 0)
        gt_fg   = (gt > 0)
        row["clDice_Overall"] = float(cldice_binary(pred_fg, gt_fg))
        
        av_vals = [row["clDice_Artery"], row["clDice_Vein"]]
        av_vals = [v for v in av_vals if not (isinstance(v, float) and math.isnan(v))]
        row["clDice_mean_AV"] = float(np.mean(av_vals)) if av_vals else float('nan')

        print(f"{case}: Overall={row['clDice_Overall']:.4f}, "
              f"Artery={row['clDice_Artery']:.4f} "
              f"Vein={row['clDice_Vein']:.4f} "
              f"mean_AV={row['clDice_mean_AV']:.4f}")

        results.append(row)

    print("\n" + "=" * 72)

    if not results:
        print("ERROR: No cases processed.")
        return 1

    artery_stats  = summarize([r["clDice_Artery"]  for r in results])
    vein_stats    = summarize([r["clDice_Vein"]    for r in results])
    overall_stats = summarize([r["clDice_Overall"] for r in results])
    meanav_stats  = summarize([r["clDice_mean_AV"] for r in results])

    summary = {
        "num_cases": len(results),
        "artery": artery_stats,
        "vein": vein_stats,
        "overall": overall_stats,
        "mean_AV": meanav_stats,
        "use_bbox_crop": USE_BBOX_CROP,
    }

    json_path = OUT_DIR / "cldice_metrics.json"
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "per_case": results}, f, indent=2)
    print(f"[OK] JSON saved -> {json_path}")

    csv_path = OUT_DIR / "cldice_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        fields = ["case",
                  "clDice_Artery", "Artery_pred_voxels", "Artery_gt_voxels",
                  "clDice_Vein",   "Vein_pred_voxels",   "Vein_gt_voxels",
                  "clDice_Overall", "clDice_mean_AV"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"[OK] CSV saved  -> {csv_path}")


    def fmt(s):
        return (f"mean={s['mean']:.4f}±{s['std']:.4f}, "
                f"median={s['median']:.4f}, "
                f"range=[{s['vmin']:.4f}, {s['vmax']:.4f}], n={s['n']}")

    print("\n==== Summary ====")
    print(f"Artery : {fmt(artery_stats)}")
    print(f"Vein   : {fmt(vein_stats)}")
    print(f"Overall: {fmt(overall_stats)}")
    print(f"MeanAV : {fmt(meanav_stats)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
