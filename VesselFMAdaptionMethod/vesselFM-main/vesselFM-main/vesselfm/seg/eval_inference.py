#!/usr/bin/env python
"""
End-to-end evaluation script for VesselFM A/V model.

Typical usage inside Docker (as module):

    python -m vesselfm.seg.eval_inference /input /output

Both /input and /output can be either:
  * a directory (multi-volume mode), or
  * a single NIfTI file (single-volume mode).
In single-volume mode you can also pass a single NIfTI path for output to control
the exact prediction filename.
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gdown
import hydra
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from monai.inferers import SlidingWindowInfererAdapt
from skimage.morphology import remove_small_objects
from skimage.exposure import equalize_hist

from .cldice_utils import hard_cldice
from .eval_preprocessing_av_ct_nii import preprocess_dataset, resample_to_spacing

from vesselfm.seg.utils.data import generate_transforms
from vesselfm.seg.utils.io import determine_reader_writer

warnings.filterwarnings("ignore")
logger = logging.getLogger("eval_inference")

# Google Drive ID for your av_ct_best_cldice checkpoint
GDRIVE_FILE_ID = "1FIz4lJl2vsYWwr10pDaE11buKWOJ6jNV"


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] - %(name)s - %(message)s",
        level=logging.INFO,
    )


def ensure_checkpoint(ckpt_path_str: str) -> Path:
    """
    Ensure that the checkpoint file exists at ckpt_path_str.
    If not, download it from Google Drive into that path.
    """
    ckpt_path = Path(ckpt_path_str)

    if ckpt_path.is_file():
        logger.info(f"Using existing checkpoint at: {ckpt_path}")
        return ckpt_path

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    logger.info(
        f"Checkpoint not found at {ckpt_path}. "
        f"Downloading from Google Drive ({url}) ..."
    )

    gdown.download(url, str(ckpt_path), quiet=False)

    if not ckpt_path.is_file():
        raise RuntimeError(
            f"Failed to download checkpoint to {ckpt_path}. "
            "Please check network access and the Google Drive link."
        )

    logger.info(f"Successfully downloaded checkpoint to {ckpt_path}")
    return ckpt_path


def get_paths(cfg):
    """
    Collect image and mask paths from the Hydra config.

    Supports config layouts:
      - cfg.image_dir / cfg.mask_dir
      - cfg.image_path / cfg.mask_path
      - cfg.data.image_dir / cfg.data.mask_dir
      - cfg.data.image_path / cfg.data.mask_path

    Each of these can be EITHER a directory or a single NIfTI file.

    Filename conventions for directory-based masks:
      1) Same-name masks:
           image_004.nii.gz -> image_004.nii.gz
      2) image/label naming:
           image_004.nii.gz -> label_004.nii.gz
    """
    # --- 1. Read locations from config with fallbacks ---
    image_root_str = (
        OmegaConf.select(cfg, "data.image_dir")
        or OmegaConf.select(cfg, "image_dir")
        or OmegaConf.select(cfg, "data.image_path")
        or OmegaConf.select(cfg, "image_path")
    )

    mask_root_str = (
        OmegaConf.select(cfg, "data.mask_dir")
        or OmegaConf.select(cfg, "mask_dir")
        or OmegaConf.select(cfg, "data.mask_path")
        or OmegaConf.select(cfg, "mask_path")
    )

    if image_root_str is None:
        raise RuntimeError(
            "Image location not set in config "
            "(looked for 'image_dir', 'data.image_dir', "
            "'image_path', and 'data.image_path')."
        )

    image_root = Path(image_root_str)

    # --- 2. Collect images as Path objects ---
    if image_root.is_dir():
        # Use *.nii* so it works for .nii and .nii.gz
        image_paths = sorted(image_root.glob("*.nii*"))
        if not image_paths:
            raise RuntimeError(f"No images found in directory {image_root}")
    elif image_root.is_file():
        image_paths = [image_root]
    else:
        raise RuntimeError(f"Image path {image_root} does not exist.")

    # If no mask_root (pure inference), just return images
    if mask_root_str is None or mask_root_str in ("", "null"):
        return image_paths, None

    mask_root = Path(mask_root_str)

    # --- 3. Build mask paths, supporting both file and directory cases ---
    if mask_root.is_file():
        # Single GT mask; only valid if there's one image
        if len(image_paths) != 1:
            raise RuntimeError(
                "Mask path points to a single file but there are "
                f"{len(image_paths)} images; cannot match uniquely."
            )
        return image_paths, [mask_root]

    if not mask_root.is_dir():
        raise RuntimeError(f"Mask path {mask_root} is neither a file nor a directory.")

    mask_dir = mask_root
    mask_paths = []

    for img_path in image_paths:
        img_name = img_path.name  # e.g. "image_004.nii.gz"

        # (a) First try: mask has EXACT same basename as image
        same_name_mask = mask_dir / img_name
        if same_name_mask.exists():
            mask_paths.append(same_name_mask)
            continue

        # (b) Second try: image_XXX.nii.gz -> label_XXX.nii.gz
        alt_mask = None
        if img_name.startswith("image_"):
            suffix = img_name[len("image_"):]  # "004.nii.gz"
            alt_mask = mask_dir / f"label_{suffix}"
            if alt_mask.exists():
                mask_paths.append(alt_mask)
                continue

        # If we get here, no matching mask was found for this image
        msg = (
            f"Could not find a mask for image:\n  {img_path}\n"
            f"Tried:\n  {same_name_mask}"
        )
        if alt_mask is not None:
            msg += f"\n  {alt_mask}"
        raise FileNotFoundError(msg)

    return image_paths, mask_paths


def resample(image, factor=None, target_shape=None):
    """
    Simple 3D trilinear resampling helper used for TTA scaling.
    Mirrors logic from inference.py.
    """
    if factor == 1:
        return image

    if target_shape:
        _, _, new_d, new_h, new_w = target_shape
    else:
        _, _, d, h, w = image.shape
        new_d = int(round(d / factor))
        new_h = int(round(h / factor))
        new_w = int(round(w / factor))

    return F.interpolate(
        image, size=(new_d, new_h, new_w), mode="trilinear", align_corners=False
    )


def load_model(cfg, device):
    """
    Load the final A/V model (including Stage-3 av_refine_head) from cfg.ckpt_path.

    Assumes ckpt_path points to the av_ct_best_cldice checkpoint written by train_av.py.
    If the file is missing, it will be downloaded from Google Drive.
    If that fails, falls back to the public vesselFM_base.pt.
    """
    # 1) Ensure local checkpoint (download if needed)
    try:
        ckpt_path = ensure_checkpoint(cfg.ckpt_path)
        logger.info(f"Loading model from {ckpt_path}.")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception as e:
        logger.info(
            f"Could not load {cfg.ckpt_path} ({e}). "
            "Falling back to Hugging Face vesselFM_base.pt."
        )
        hf_hub_download(repo_id="bwittmann/vesselFM", filename="meta.yaml")
        ckpt = torch.load(
            hf_hub_download(
                repo_id="bwittmann/vesselFM",
                filename="vesselFM_base.pt",
            ),
            map_location=device,
            weights_only=True,
        )

    # 2) Instantiate the same backbone as in training
    model = hydra.utils.instantiate(cfg.model)

    # 3) Figure out how many output channels the backbone has (should be 3)
    if "out_channels" in cfg.model:
        out_ch = cfg.model.out_channels
    elif "num_classes" in cfg.model:
        out_ch = cfg.model.num_classes
    else:
        out_ch = 3  # sensible default for your A/V/BG setup

    # 4) Attach heads exactly like in train_av.py
    # Vessel head (not strictly needed for inference right now, but harmless)
    if not hasattr(model, "vessel_head"):
        logger.info("[load_model] Adding vessel_head for union-of-vessels output.")
        model.vessel_head = nn.Conv3d(out_ch, 1, kernel_size=1)

    # AV refine head: Stage-3 2-class A/V classifier on top of logits
    if not hasattr(model, "av_refine_head"):
        logger.info("[load_model] Adding av_refine_head (A/V refine) on top of logits.")
        model.av_refine_head = nn.Conv3d(out_ch, 2, kernel_size=1)

    # 5) Load weights into this full architecture
    if isinstance(ckpt, dict):
        # Works for both raw state_dict and {'state_dict': ...}
        state = ckpt.get("state_dict", ckpt)
    else:
        state = ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info(
        f"[load_model] Loaded checkpoint with {len(missing)} missing and "
        f"{len(unexpected)} unexpected keys."
    )

    return model


def run_inference(cfg, raw_image_source=None, explicit_output_path=None):
    """
    Core inference loop, adapted from vesselfm.seg.inference.main(),
    but taking a composed Hydra config as input.

    raw_image_source:
        Either a directory containing the original (raw) CT NIfTI files or
        a single NIfTI file in single-volume mode. Used to map predictions back
        to the raw image geometry (spacing, shape, affine).

    explicit_output_path:
        If not None and there is exactly one input volume, the prediction will
        be written exactly to this file (overriding the default naming scheme).
    """
    # Seed libraries
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # Set device
    logger.info(f"Using device {cfg.device}.")
    device = cfg.device

    # Load model and ckpt
    model = load_model(cfg, device)
    model.to(device)
    model.eval()

    # Init pre-processing transforms
    transforms = generate_transforms(cfg.transforms_config)

    # I/O
    output_folder = Path(cfg.output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    image_paths, mask_paths = get_paths(cfg)
    logger.info(f"Found {len(image_paths)} images in {image_paths[0].parent}.")

    file_ending = (
        cfg.image_file_ending if cfg.image_file_ending else image_paths[0].suffix
    )
    image_reader_writer = determine_reader_writer(file_ending)()
    _ = determine_reader_writer(file_ending)()  # save_writer (kept for parity)

    # Init sliding window inferer
    logger.debug(f"Sliding window patch size: {cfg.patch_size}")
    logger.debug(f"Sliding window batch size: {cfg.batch_size}.")
    logger.debug(f"Sliding window overlap: {cfg.overlap}.")
    inferer = SlidingWindowInfererAdapt(
        roi_size=cfg.patch_size,
        sw_batch_size=cfg.batch_size,
        overlap=cfg.overlap,
        mode=cfg.mode,
        sigma_scale=cfg.sigma_scale,
        padding_mode=cfg.padding_mode,
    )

    # Prepare raw source Path (can be None, dir, or single file)
    raw_source = Path(raw_image_source) if raw_image_source is not None else None

    # Loop over images
    metrics_dict = {}
    with torch.no_grad():
        for idx, image_path in tqdm(
            enumerate(image_paths),
            total=len(image_paths),
            desc="Processing images.",
        ):
            preds = []  # per-scale logits (kept on device)

            for scale in cfg.tta.scales:
                # Read image
                image_np = image_reader_writer.read_images(image_path)[0].astype(
                    np.float32
                )
                image = transforms(image_np)[None].to(device)  # (1,1,D,H,W) on device

                # TTA intensity transforms
                if cfg.tta.invert:
                    if image.mean() > cfg.tta.invert_mean_thresh:
                        image = 1 - image
                if cfg.tta.equalize_hist:
                    image_np = image.cpu().squeeze().numpy()
                    image_equal_hist_np = equalize_hist(
                        image_np, nbins=cfg.tta.hist_bins
                    )
                    image = (
                        torch.from_numpy(image_equal_hist_np)
                        .to(device)[None][None]
                    )

                # Resample for scale, run model, resample back
                original_shape = image.shape
                image_scaled = resample(image, factor=scale)  # on device
                logits = inferer(image_scaled, model)  # (1,3,D,H,W) on device
                logits = resample(logits, target_shape=original_shape)
                preds.append(logits.squeeze(0))  # (3,D,H,W) on device

            # Preds is a list of per-scale logits, each (3,D,H,W) on device
            logits_ensemble = torch.stack(preds).mean(dim=0)  # (3,D,H,W) on device

            if hasattr(model, "av_refine_head"):
                # Stage-3 A/V refinement (same as eval_epoch(use_av_refine=True))
                base_probs = F.softmax(
                    logits_ensemble.unsqueeze(0), dim=1
                )  # (1,3,D,H,W)
                p_bg = base_probs[:, 0:1, ...]
                p_union = (
                    base_probs[:, 1:3, ...].sum(dim=1, keepdim=True).clamp(0.0, 1.0)
                )

                av_logits = model.av_refine_head(
                    logits_ensemble.unsqueeze(0)
                )  # (1,2,D,H,W)
                av_probs = F.softmax(av_logits, dim=1)
                p_art_cond = av_probs[:, 0:1, ...]
                p_vein_cond = av_probs[:, 1:2, ...]

                p_art = p_union * p_art_cond
                p_vein = p_union * p_vein_cond

                denom = p_bg + p_art + p_vein + 1e-8
                probs_final = torch.cat(
                    [p_bg / denom, p_art / denom, p_vein / denom],
                    dim=1,
                )[0]  # (3,D,H,W) on device
            else:
                # Fallback: original vesselFM A/V logits fusion
                if cfg.merging.max:
                    probs_final = torch.stack(
                        [F.softmax(p, dim=0) for p in preds]
                    ).max(dim=0)[0]
                else:
                    probs_final = torch.stack(
                        [F.softmax(p, dim=0) for p in preds]
                    ).mean(dim=0)

            # Prediction in preprocessed image space
            label = probs_final.argmax(0).cpu().numpy().astype(np.uint8)

            # Class-wise CC cleanup
            if cfg.post.apply:
                cleaned = np.zeros_like(label, dtype=np.uint8)
                for c in (1, 2):  # artery, vein
                    cm = label == c
                    cm = remove_small_objects(
                        cm,
                        min_size=cfg.post.small_objects_min_size,
                        connectivity=cfg.post.small_objects_connectivity,
                    )
                    cleaned[cm] = c
                label = cleaned

            # Label is a numpy array (D, H, W) in model/reader order
            label_np = label.astype(np.uint8)

            # Use the preprocessed image as current reference (for orientation)
            pre_nii = nib.load(str(image_path))
            pre_shape = pre_nii.shape

            # If shape is (D,H,W)=(234,186,247) but ref is (247,186,234),
            # Swap axes 0 and 2 so we match nibabel's (X,Y,Z) convention.
            if (
                label_np.shape != pre_shape
                and label_np.shape[0] == pre_shape[2]
                and label_np.shape[1] == pre_shape[1]
                and label_np.shape[2] == pre_shape[0]
            ):
                label_np = np.transpose(label_np, (2, 1, 0))
                logger.info(
                    f"Transposed prediction from original shape to match preprocessed shape {pre_shape}."
                )
            
            # --- Metrics in PREPROCESSED space (paper metrics) ---
            if mask_paths is not None:
                # GT in preprocessed space (from /tmp/av_preprocessed/labels)
                gt_pre_nii = nib.load(str(mask_paths[idx]))
                gt_pre = gt_pre_nii.get_fdata().astype(np.int16)

                if gt_pre.shape != label_np.shape:
                    logger.warning(
                        f"[metrics-pre] GT shape {gt_pre.shape} != pred shape {label_np.shape}; "
                        "skipping metrics for this case in preprocessed space."
                    )
                else:
                    pred_pre = label_np
                    gt = gt_pre

                    # --- VESSEL (A ∪ V) ---
                    union_pred = pred_pre > 0
                    union_gt = gt > 0

                    inter_u = np.logical_and(union_pred, union_gt).sum()
                    denom_u = union_pred.sum() + union_gt.sum()
                    dice_vessel = (
                        2.0 * inter_u / (denom_u + 1e-5) if denom_u > 0 else 0.0
                    )
                    cldice_vessel = hard_cldice(
                        union_pred.astype(bool), union_gt.astype(bool)
                    )

                    # --- ARTERY (class = 1) ---
                    g_art = gt == 1
                    if g_art.any():
                        p_art = pred_pre == 1
                        inter_a = np.logical_and(p_art, g_art).sum()
                        denom_a = p_art.sum() + g_art.sum()
                        dice_art = (
                            2.0 * inter_a / (denom_a + 1e-5) if denom_a > 0 else 0.0
                        )
                        cldice_art = hard_cldice(
                            p_art.astype(bool), g_art.astype(bool)
                        )
                    else:
                        dice_art = 0.0
                        cldice_art = 0.0

                    # --- VEIN (class = 2) ---
                    g_vein = gt == 2
                    if g_vein.any():
                        p_vein = pred_pre == 2
                        inter_v = np.logical_and(p_vein, g_vein).sum()
                        denom_v = p_vein.sum() + g_vein.sum()
                        dice_vein = (
                            2.0 * inter_v / (denom_v + 1e-5) if denom_v > 0 else 0.0
                        )
                        cldice_vein = hard_cldice(
                            p_vein.astype(bool), g_vein.astype(bool)
                        )
                    else:
                        dice_vein = 0.0
                        cldice_vein = 0.0

                    case_name = image_path.name.split(".")[0]
                    logger.info(
                        f"[pre] {case_name}: "
                        f"Dice(vessel)={dice_vessel:.4f} clDice(vessel)={cldice_vessel:.4f} "
                        f"Dice(art)={dice_art:.4f} clDice(art)={cldice_art:.4f} "
                        f"Dice(vein)={dice_vein:.4f} clDice(vein)={cldice_vein:.4f}"
                    )

                    # Store metrics (these are what the mean summary uses)
                    metrics_dict[case_name] = {
                        "dice_vessel": torch.tensor(dice_vessel),
                        "cldice_vessel": torch.tensor(cldice_vessel),
                        "dice_art": torch.tensor(dice_art),
                        "cldice_art": torch.tensor(cldice_art),
                        "dice_vein": torch.tensor(dice_vein),
                        "cldice_vein": torch.tensor(cldice_vein),
                        # keep these aliases for backwards compatibility if needed
                        "dice": torch.tensor(dice_vessel),
                        "cldice": torch.tensor(cldice_vessel),
                    }
            # --- END metrics in PREPROCESSED space ---

            # If we know the raw image location, resample label to raw geometry
            raw_nii = None
            if raw_source is not None and raw_source.is_file():
                raw_nii = nib.load(str(raw_source))
                raw_shape = raw_nii.shape

                pre_spacing = pre_nii.header.get_zooms()[:3]
                raw_spacing = raw_nii.header.get_zooms()[:3]

                if (label_np.shape != raw_shape) or (pre_spacing != raw_spacing):
                    logger.info(
                        f"Resampling prediction from preprocessed spacing {pre_spacing} "
                        f"to raw spacing {raw_spacing}."
                    )
                    # Resample prediction to raw spacing
                    label_resampled = resample_to_spacing(
                        label_np.astype(np.float32),
                        src_spacing=pre_spacing,
                        tgt_spacing=raw_spacing,
                        order=0,  # nearest-neighbor for labels
                    ).astype(np.uint8)

                    label_np = label_resampled

                    # Additional safety: if shapes still differ slightly due to rounding,
                    # pad (but do NOT crop) to match raw_shape.
                    if label_np.shape != raw_shape:
                        logger.warning(
                            f"Resampled label shape {label_np.shape} != raw shape {raw_shape}; "
                            f"padding (no cropping) to match."
                        )

                        # If any dimension is larger than raw_shape, raise an error instead of cropping
                        too_large = any(a > b for a, b in zip(label_np.shape, raw_shape))
                        if too_large:
                            raise RuntimeError(
                                f"Resampled label shape {label_np.shape} is larger than raw shape "
                                f"{raw_shape} in at least one dimension; cannot match without cropping."
                            )

                        # Compute symmetric padding per dimension
                        pad_width = []
                        for a, b in zip(label_np.shape, raw_shape):
                            diff = b - a
                            before = diff // 2
                            after = diff - before
                            pad_width.append((before, after))

                        label_np = np.pad(
                            label_np,
                            pad_width=pad_width,
                            mode="constant",
                            constant_values=0,
                        )

            # Choose reference NIfTI for saving (raw if available, else preprocessed)
            if raw_nii is not None:
                ref_nii = raw_nii
            else:
                ref_nii = pre_nii

            pred_nii = nib.Nifti1Image(
                label_np,
                affine=ref_nii.affine,
                header=ref_nii.header,
            )

            # Keep sform/qform consistent
            sform, sform_code = ref_nii.get_sform(coded=True)
            qform, qform_code = ref_nii.get_qform(coded=True)
            pred_nii.set_sform(sform, code=sform_code or 1)
            pred_nii.set_qform(qform, code=qform_code or 1)

            # Name the output based on the raw image name if available
            if explicit_output_path is not None and len(image_paths) == 1:
                out_path = Path(explicit_output_path)
            else:
                if explicit_output_path is not None and len(image_paths) > 1:
                    logger.warning(
                        "Explicit output path was provided but multiple input images "
                        "are present; ignoring explicit path and using default naming."
                    )
                # Default naming scheme
                if ref_nii.get_filename() is not None:
                    base_name = Path(ref_nii.get_filename()).name.split(".")[0]
                else:
                    # Fallback if header lacks filename
                    base_name = image_path.name.split(".")[0]
                out_name = f"{base_name}_{cfg.file_app}pred.nii.gz"
                out_path = output_folder / out_name

            nib.save(pred_nii, str(out_path))
            logger.info(f"Saved prediction to {out_path}")

    # Summarize over all images
    if mask_paths is not None and len(metrics_dict) > 0:
        metric_names = list(next(iter(metrics_dict.values())).keys())
        mean_metrics = {}
        for m in metric_names:
            vals = [metrics_dict[k][m].item() for k in metrics_dict]
            mean_metrics[m] = float(np.mean(vals))

        # Per-class means
        logger.info(f"Mean Dice(vessel): {mean_metrics['dice_vessel']:.4f}")
        logger.info(f"Mean clDice(vessel): {mean_metrics['cldice_vessel']:.4f}")
        logger.info(f"Mean Dice(art): {mean_metrics['dice_art']:.4f}")
        logger.info(f"Mean clDice(art): {mean_metrics['cldice_art']:.4f}")
        logger.info(f"Mean Dice(vein): {mean_metrics['dice_vein']:.4f}")
        logger.info(f"Mean clDice(vein): {mean_metrics['cldice_vein']:.4f}")

        with open(output_folder / "metrics_per_volume.json", "w") as f:
            json.dump(
                {k: {m: float(v[m].item()) for m in v} for k, v in metrics_dict.items()},
                f,
                indent=2,
            )

        with open(output_folder / "metrics_mean.json", "w") as f:
            json.dump(mean_metrics, f, indent=2)


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description=(
            "End-to-end evaluation: preprocess CTs and run VesselFM AV inference.\n"
            "Typical container usage:\n"
            "  python -m vesselfm.seg.eval_inference /input /output\n\n"
            "Both /input and /output can be a directory or a single NIfTI file."
        )
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory OR single .nii/.nii.gz with raw CT image(s).",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help=(
            "Directory OR single .nii/.nii.gz where predictions will be written. "
            "If a single file is given, only valid in single-volume mode."
        ),
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        default=None,
        help="Optional directory OR single NIfTI file with GT labels (for local metric computation).",
    )
    parser.add_argument(
        "--tmp_preproc_dir",
        type=str,
        default="/tmp/av_preprocessed",
        help="Working directory for preprocessed images/labels.",
    )
    parser.add_argument(
        "--skip_preprocess",
        action="store_true",
        help="Skip preprocessing (assume input_dir already contains preprocessed images).",
    )

    args = parser.parse_args()

    raw_input_path = Path(args.input_dir)
    raw_labels_path = Path(args.labels_dir) if args.labels_dir is not None else None

    # Interpret output path: directory vs explicit file
    output_arg = Path(args.output_dir)
    if output_arg.suffix in (".nii", ".nii.gz"):
        # Single-volume explicit output file path
        explicit_output_file = output_arg
        out_dir = explicit_output_file.parent
    else:
        explicit_output_file = None
        out_dir = output_arg

    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_root = Path(args.tmp_preproc_dir)
    preproc_images_dir = tmp_root / "images"
    preproc_labels_dir = tmp_root / "labels" if raw_labels_path is not None else None

    if not args.skip_preprocess:
        logger.info("Starting preprocessing of input volumes...")
        preprocess_dataset(
            images_dir=raw_input_path,
            labels_dir=raw_labels_path,
            out_images_dir=preproc_images_dir,
            out_labels_dir=preproc_labels_dir,
            target_spacing=(1.0, 1.0, 1.0),
            clip_range=(-1000.0, 400.0),
            do_zscore=False,
            crop_mode="none",  # keep full FOV for inference
            crop_margin=10,
            percentile_norm=True,
            percentile_range=(0.5, 99.5),
        )
        logger.info("Preprocessing finished.")
    else:
        # If skipping preprocessing, treat input_dir as already preprocessed
        preproc_images_dir = raw_input_path
        preproc_labels_dir = raw_labels_path

    # Compose Hydra config from the original config directory
    here = Path(__file__).resolve().parent
    config_dir = here / "configs"

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), job_name="eval_inference"):
        cfg = compose(config_name="eval_inference")

    # Allow dynamic insertion of new keys
    OmegaConf.set_struct(cfg, False)

    if "data" not in cfg:
        cfg.data = OmegaConf.create({})
    # Override paths in config to point to preprocessed data and desired output
    cfg.data.image_dir = str(preproc_images_dir)
    cfg.data.mask_dir = (
        str(preproc_labels_dir) if preproc_labels_dir is not None else None
    )

    cfg.output_folder = str(out_dir)

    logger.info(f"Using config:\n{OmegaConf.to_yaml(cfg)}")

    # raw_input_path can be directory or single file; run_inference handles both
    run_inference(
        cfg,
        raw_image_source=str(raw_input_path),
        explicit_output_path=str(explicit_output_file) if explicit_output_file is not None else None,
    )


if __name__ == "__main__":
    main()