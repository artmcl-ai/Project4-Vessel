""" Script to perform inference with vesselFM."""

import logging
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra

import numpy as np
import json
import nibabel as nib
from .cldice_utils import hard_cldice
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from monai.inferers import SlidingWindowInfererAdapt
from skimage.morphology import remove_small_objects
from skimage.exposure import equalize_hist
from scipy.ndimage import zoom as nd_zoom

from vesselfm.seg.utils.data import generate_transforms
from vesselfm.seg.utils.io import determine_reader_writer
from vesselfm.seg.utils.evaluation import Evaluator, calculate_mean_metrics

from omegaconf import OmegaConf
from pathlib import Path


warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

def build_model(num_classes=3, dropout=0.0):
    # Load inference config to get the same model definition with ckpt_path
    here = Path(__file__).resolve().parent
    config_dir = here / "configs"

    # Compose the full inference config
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), job_name="av_model"):
        cfg_inf = compose(config_name="inference")

    if "model" not in cfg_inf:
        raise ValueError(
            f"'model' key not found in composed config. Top-level keys: {list(cfg_inf.keys())}"
        )

    mcfg = cfg_inf.model

    # Set number of output channels to num_classes
    if "out_channels" in mcfg:
        logger.info(f"[build_model] Setting model.out_channels -> {num_classes}")
        mcfg.out_channels = num_classes
    elif "num_classes" in mcfg:
        logger.info(f"[build_model] Setting model.num_classes -> {num_classes}")
        mcfg.num_classes = num_classes
    else:
        logger.warning(
            "[build_model] Neither 'out_channels' nor 'num_classes' found in model config; "
            "leaving output channels as-is."
        )

    # Dropout override
    if "dropout" in mcfg:
        logger.info(f"[build_model] Setting model.dropout -> {dropout}")
        mcfg.dropout = dropout

    # Instantiate MONAI DynUNet
    model = hydra.utils.instantiate(cfg_inf.model)

    # Load pretrained VesselFM weights
    try:
        logger.info(f"[build_model] Loading pretrained weights from {cfg_inf.ckpt_path}.")
        ckpt = torch.load(Path(cfg_inf.ckpt_path), map_location="cpu", weights_only=True)
    except Exception as e:
        logger.info(
            f"[build_model] Could not load ckpt from cfg_inf.ckpt_path ({e}). "
            "Falling back to Hugging Face vesselFM_base.pt."
        )
        hf_hub_download(repo_id="bwittmann/vesselFM", filename="meta.yaml")
        ckpt = torch.load(
            hf_hub_download(repo_id="bwittmann/vesselFM", filename="vesselFM_base.pt"),
            map_location="cpu",
            weights_only=True,
        )

    # Drop old head weights (1-channel) to avoid size mismatch
    head_keys = [k for k in ckpt.keys() if k.startswith("output_block.")]
    if head_keys:
        logger.info(
            f"[build_model] Removing {len(head_keys)} head params from checkpoint "
            f"to accommodate new 3-class head: {head_keys}"
        )
        for k in head_keys:
            ckpt.pop(k)

    # Now load backbone weights (strict=False allows missing head params)
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    logger.info(
        f"[build_model] Loaded pretrained VesselFM weights with "
        f"{len(missing)} missing and {len(unexpected)} unexpected keys "
        f"(expected when swapping to a 3-class head)."
    )

    # Add an explicit vessel head as a second physical head.
    if not hasattr(model, "vessel_head"):
        logger.info("[build_model] Adding 1x1x1 vessel_head on top of A/V logits.")
        model.vessel_head = nn.Conv3d(num_classes, 1, kernel_size=1)

    return model


def load_model(cfg, device):
    """
    Load the final A/V model (including Stage-3 av_refine_head) from ckpt_path.
    Assumes ckpt_path points to the av_ct_best_cldice.pt written by train_av.py.
    """
    # Load checkpoint
    try:
        logger.info(f"Loading model from {cfg.ckpt_path}.")
        ckpt = torch.load(Path(cfg.ckpt_path), map_location=device, weights_only=True)
    except Exception as e:
        logger.info(
            f"Could not load {cfg.ckpt_path} ({e}). "
            "Falling back to Hugging Face vesselFM_base.pt."
        )
        hf_hub_download(repo_id="bwittmann/vesselFM", filename="meta.yaml")
        ckpt = torch.load(
            hf_hub_download(repo_id="bwittmann/vesselFM", filename="vesselFM_base.pt"),
            map_location=device,
            weights_only=True,
        )

    # nstantiate the same backbone as in training
    model = hydra.utils.instantiate(cfg.model)

    # Figure out how many output channels the backbone has (should be 3)
    if "out_channels" in cfg.model:
        out_ch = cfg.model.out_channels
    elif "num_classes" in cfg.model:
        out_ch = cfg.model.num_classes
    else:
        out_ch = 3  # sensible default for your A/V/BG setup

    # Attach heads exactly like in train_av.py
    # Vessel head (not strictly needed for inference right now, but harmless)
    if not hasattr(model, "vessel_head"):
        logger.info("[load_model] Adding vessel_head for union-of-vessels output.")
        model.vessel_head = nn.Conv3d(out_ch, 1, kernel_size=1)

    # AV refine head: Stage-3 2-class A/V classifier on top of logits
    if not hasattr(model, "av_refine_head"):
        logger.info("[load_model] Adding av_refine_head (A/V refine) on top of logits.")
        model.av_refine_head = nn.Conv3d(out_ch, 2, kernel_size=1)

    # Load weights into this full architecture
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


def get_paths(cfg):
    """
    Collect image and mask paths.

    Supports config layouts:
      - cfg.image_dir / cfg.mask_dir
      - cfg.image_path / cfg.mask_path
      - cfg.data.image_dir / cfg.data.mask_dir
      - cfg.data.image_path / cfg.data.mask_path

    Supports filename conventions:
      1) Same-name masks:
           image_004.nii.gz -> image_004.nii.gz
      2) image/label naming:
           image_004.nii.gz -> label_004.nii.gz
    """
    import os

    # Read directories from config with fallbacks
    image_dir_str = (
        OmegaConf.select(cfg, "data.image_dir")
        or OmegaConf.select(cfg, "image_dir")
        or OmegaConf.select(cfg, "data.image_path")
        or OmegaConf.select(cfg, "image_path")
    )

    mask_dir_str = (
        OmegaConf.select(cfg, "data.mask_dir")
        or OmegaConf.select(cfg, "mask_dir")
        or OmegaConf.select(cfg, "data.mask_path")
        or OmegaConf.select(cfg, "mask_path")
    )

    if image_dir_str is None:
        raise RuntimeError(
            "image directory not set in config "
            "(looked for 'image_dir', 'data.image_dir', "
            "'image_path', and 'data.image_path')."
        )

    image_dir = Path(image_dir_str)

    # Normalize mask_dir: either a Path or None
    if mask_dir_str is None or mask_dir_str in ("", "null"):
        mask_dir = None
    else:
        mask_dir = Path(mask_dir_str)

    # Collect images as Path objects
    # Use *.nii* so it works for .nii and .nii.gz
    image_paths = sorted(image_dir.glob("*.nii*"))
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    # If no mask_dir (pure inference), just return images
    if mask_dir is None:
        return image_paths, None

    # --- 3. Build mask paths with both naming schemes (also as Path objects) ---
    mask_paths = []

    for img_path in image_paths:
        # img_path is a Path
        img_name = img_path.name  # e.g. "image_004.nii.gz"

        # First try: mask has EXACT same basename as image
        same_name_mask = mask_dir / img_name
        if same_name_mask.exists():
            mask_paths.append(same_name_mask)
            continue

        # Second try: image_XXX.nii.gz -> label_XXX.nii.gz
        alt_mask = None
        if img_name.startswith("image_"):
            suffix = img_name[len("image_"):]          # "004.nii.gz"
            alt_mask = mask_dir / f"label_{suffix}"
            if alt_mask.exists():
                mask_paths.append(alt_mask)
                continue

        # No matching mask was found for this image
        msg = (
            f"Could not find a mask for image:\n  {img_path}\n"
            f"Tried:\n  {same_name_mask}"
        )
        if alt_mask is not None:
            msg += f"\n  {alt_mask}"
        raise FileNotFoundError(msg)

    return image_paths, mask_paths



def resample(image, factor=None, target_shape=None):
    if factor == 1:
        return image
    
    if target_shape:
        _, _, new_d, new_h, new_w = target_shape
    else:
        _, _, d, h, w = image.shape
        new_d, new_h, new_w = int(round(d / factor)), int(round(h / factor)), int(round(w / factor))
    return F.interpolate(image, size=(new_d, new_h, new_w), mode="trilinear", align_corners=False)


def resample_mask_to_spacing(mask, src_spacing, tgt_spacing, order=0):
    """
    Resample a 3D mask from src_spacing to tgt_spacing using nearest-neighbor (order=0).

    mask: (X, Y, Z) np.ndarray (integer labels)
    src_spacing, tgt_spacing: iterables of length 3 (sx, sy, sz)
    """
    src_spacing = np.array(src_spacing, dtype=np.float32)
    tgt_spacing = np.array(tgt_spacing, dtype=np.float32)
    zoom_factors = src_spacing / tgt_spacing  # Same convention as preprocess_av_ct_nii
    return nd_zoom(mask, zoom_factors, order=order)


@hydra.main(config_path="configs", config_name="inference", version_base="1.3.2")
def main(cfg):
    # seed libraries
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # set device
    logger.info(f"Using device {cfg.device}.")
    device = cfg.device

    # load model and ckpt
    model = load_model(cfg, device)
    model.to(device)
    model.eval()

    # init pre-processing transforms
    transforms = generate_transforms(cfg.transforms_config)

    # i/o
    output_folder = Path(cfg.output_folder)
    output_folder.mkdir(exist_ok=True)

    image_paths, mask_paths = get_paths(cfg)
    logger.info(f"Found {len(image_paths)} images in {cfg.image_path}.")

    file_ending = (cfg.image_file_ending if cfg.image_file_ending else image_paths[0].suffix)
    image_reader_writer = determine_reader_writer(file_ending)()
    save_writer = determine_reader_writer(file_ending)()

    # init sliding window inferer
    logger.debug(f"Sliding window patch size: {cfg.patch_size}")
    logger.debug(f"Sliding window batch size: {cfg.batch_size}.")
    logger.debug(f"Sliding window overlap: {cfg.overlap}.")
    inferer = SlidingWindowInfererAdapt(
        roi_size=cfg.patch_size, sw_batch_size=cfg.batch_size, overlap=cfg.overlap, 
        mode=cfg.mode, sigma_scale=cfg.sigma_scale, padding_mode=cfg.padding_mode
    )

    # loop over images
    metrics_dict = {}
    with torch.no_grad():
        for idx, image_path in tqdm(
            enumerate(image_paths),
            total=len(image_paths),
            desc="Processing images.",
        ):
            preds = []  # per-scale logits (kept on device)
            mask_np = None

            for scale in cfg.tta.scales:
                # read image (and mask if available)
                image_np = image_reader_writer.read_images(image_path)[0].astype(np.float32)
                image = transforms(image_np)[None].to(device)  # (1,1,D,H,W) on device

                if mask_paths is not None and mask_np is None:
                    # Load 3-class GT: 0=bg,1=artery,2=vein, keep on CPU
                    mask_np = image_reader_writer.read_images(mask_paths[idx])[0].astype(np.int16)

                # TTA intensity transforms
                if cfg.tta.invert:
                    if image.mean() > cfg.tta.invert_mean_thresh:
                        image = 1 - image
                if cfg.tta.equalize_hist:
                    image_np = image.cpu().squeeze().numpy()
                    image_equal_hist_np = equalize_hist(image_np, nbins=cfg.tta.hist_bins)
                    image = torch.from_numpy(image_equal_hist_np).to(device)[None][None]

                # resample for scale, run model, resample back
                original_shape = image.shape
                image_scaled = resample(image, factor=scale)          # on device
                logits = inferer(image_scaled, model)                 # (1,3,D,H,W) on device
                logits = resample(logits, target_shape=original_shape)
                preds.append(logits.squeeze(0))                       # (3,D,H,W) on device

            # Preds is a list of per-scale logits, each (3,D,H,W) on device
            logits_ensemble = torch.stack(preds).mean(dim=0)          # (3,D,H,W) on device

            if hasattr(model, "av_refine_head"):
                # Stage-3 A/V refinement (same as eval_epoch(use_av_refine=True))
                base_probs = F.softmax(logits_ensemble.unsqueeze(0), dim=1)  # (1,3,D,H,W)
                p_bg = base_probs[:, 0:1, ...]
                p_union = base_probs[:, 1:3, ...].sum(dim=1, keepdim=True).clamp(0.0, 1.0)

                av_logits = model.av_refine_head(logits_ensemble.unsqueeze(0))  # (1,2,D,H,W)
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
                    probs_final = torch.stack([F.softmax(p, dim=0) for p in preds]).max(dim=0)[0]
                else:
                    probs_final = torch.stack([F.softmax(p, dim=0) for p in preds]).mean(dim=0)

            # Move to CPU only when converting to numpy for saving / metrics
            label = probs_final.argmax(0).cpu().numpy().astype(np.uint8)

            # Class-wise CC cleanup
            if cfg.post.apply:
                cleaned = np.zeros_like(label, dtype=np.uint8)
                for c in (1, 2):  # artery, vein
                    cm = (label == c)
                    cm = remove_small_objects(
                        cm,
                        min_size=cfg.post.small_objects_min_size,
                        connectivity=cfg.post.small_objects_connectivity,
                    )
                    cleaned[cm] = c
                label = cleaned

            # Label is a numpy array (D, H, W) in model/reader order
            label_np = label.astype(np.uint8)

            # Geometry handling
            # Preprocessed reference (the volume actually fed into the model)
            pre_nii = nib.load(str(image_path))
            pre_shape = pre_nii.shape
            pre_spacing = pre_nii.header.get_zooms()[:3]

            # RAW reference: same filename but in cfg.raw_image_dir
            raw_image_dir = getattr(cfg, "raw_image_dir", None)
            raw_nii = None
            label_for_save = label_np  # default: save in preprocessed space
            ref_nii = pre_nii          # default reference

            if raw_image_dir not in (None, "", "null"):
                raw_dir = Path(raw_image_dir)
                raw_path = raw_dir / image_path.name

                if raw_path.exists():
                    raw_nii = nib.load(str(raw_path))
                    raw_spacing = raw_nii.header.get_zooms()[:3]

                    # Resample mask from pre spacing -> raw spacing
                    mask_raw = resample_mask_to_spacing(
                        label_np,
                        src_spacing=pre_spacing,
                        tgt_spacing=raw_spacing,
                        order=0,  # nearest neighbor for labels
                    ).astype(np.uint8)

                    # If small size mismatch (rounding), crop/pad to raw_nii.shape
                    if mask_raw.shape != raw_nii.shape:
                        logger.warning(
                            f"Resampled mask shape {mask_raw.shape} != raw image shape {raw_nii.shape}; "
                            "cropping/padding to match."
                        )
                        out = np.zeros(raw_nii.shape, dtype=np.uint8)
                        common = tuple(min(a, b) for a, b in zip(mask_raw.shape, out.shape))
                        slices_out = tuple(slice(0, c) for c in common)
                        slices_in = tuple(slice(0, c) for c in common)
                        out[slices_out] = mask_raw[slices_in]
                        mask_raw = out

                    label_for_save = mask_raw
                    ref_nii = raw_nii
                else:
                    logger.warning(
                        f"Raw image not found at {raw_path}; "
                        "saving prediction in preprocessed space instead."
                    )
                    ref_nii = pre_nii
                    label_for_save = label_np
            else:
                # No raw_image_dir provided: save in preprocessed space,
                if label_np.shape != pre_shape:
                    # Common case: (D,H,W) vs (X,Y,Z) where only axis 0 and 2 differ
                    if (
                        label_np.shape[0] == pre_shape[2]
                        and label_np.shape[1] == pre_shape[1]
                        and label_np.shape[2] == pre_shape[0]
                    ):
                        label_np = np.transpose(label_np, (2, 1, 0))
                        logger.info(
                            f"Transposed prediction to match preprocessed shape {pre_shape}."
                        )
                    else:
                        raise RuntimeError(
                            f"Predicted mask shape {label_np.shape} does not match preprocessed image shape "
                            f"{pre_shape} and is not a simple 0<->2 axis swap."
                        )
                label_for_save = label_np
                ref_nii = pre_nii

            # Save prediction in ref_nii space (raw if available, otherwise preprocessed)
            pred_nii = nib.Nifti1Image(
                label_for_save,
                affine=ref_nii.affine,
                header=ref_nii.header,
            )

            # Keep sform/qform consistent with reference image
            pred_nii.set_sform(
                ref_nii.get_sform(),
                code=ref_nii.get_sform(coded=True)[1] or 1
            )
            pred_nii.set_qform(
                ref_nii.get_qform(),
                code=ref_nii.get_qform(coded=True)[1] or 1
            )

            out_path = output_folder / f"{image_path.name.split('.')[0]}_{cfg.file_app}pred.nii.gz"
            nib.save(pred_nii, str(out_path))

            # Metrics if GT masks are available
            if mask_paths is not None and mask_np is not None:
                """
                label:   (D, H, W) with {0:bg, 1:artery, 2:vein}
                mask_np: (D, H, W) with {0:bg, 1:artery, 2:vein}

                We report:
                  - Dice / clDice for union, artery, vein
                  - FP% and FN%:
                      * FP%_union_pred:   FP / (TP + FP)   for union (A∪V)
                      * FN%_union_gt:     FN / (TP + FN)   for union (A∪V)
                    and analogous metrics for artery and vein.
                """

                # -------------------------
                # UNION (A ∪ V) as binary
                # -------------------------
                union_pred = label > 0    # (D,H,W) bool
                union_gt   = mask_np > 0  # (D,H,W) bool

                tp_u = np.logical_and(union_pred, union_gt).sum()
                fp_u = np.logical_and(union_pred, np.logical_not(union_gt)).sum()
                fn_u = np.logical_and(np.logical_not(union_pred), union_gt).sum()
                tn_u = np.logical_and(np.logical_not(union_pred), np.logical_not(union_gt)).sum()

                denom_u = union_pred.sum() + union_gt.sum()
                dice_union = 2.0 * tp_u / (denom_u + 1e-5) if denom_u > 0 else 0.0
                cldice_union = hard_cldice(union_pred.astype(bool), union_gt.astype(bool))

                pred_pos_u = tp_u + fp_u
                gt_pos_u   = tp_u + fn_u

                fp_pct_union_pred = 100.0 * fp_u / (pred_pos_u + 1e-5) if pred_pos_u > 0 else 0.0
                fn_pct_union_gt   = 100.0 * fn_u / (gt_pos_u + 1e-5)   if gt_pos_u > 0 else 0.0

                # -------------------------
                # ARTERY (class = 1)
                # -------------------------
                g_art = (mask_np == 1)
                if g_art.any():
                    p_art = (label == 1)

                    tp_a = np.logical_and(p_art, g_art).sum()
                    fp_a = np.logical_and(p_art, np.logical_not(g_art)).sum()
                    fn_a = np.logical_and(np.logical_not(p_art), g_art).sum()

                    denom_a = p_art.sum() + g_art.sum()
                    dice_art = 2.0 * tp_a / (denom_a + 1e-5) if denom_a > 0 else 0.0
                    cldice_art = hard_cldice(p_art.astype(bool), g_art.astype(bool))

                    pred_pos_a = tp_a + fp_a
                    gt_pos_a   = tp_a + fn_a

                    fp_pct_art_pred = 100.0 * fp_a / (pred_pos_a + 1e-5) if pred_pos_a > 0 else 0.0
                    fn_pct_art_gt   = 100.0 * fn_a / (gt_pos_a + 1e-5)   if gt_pos_a > 0 else 0.0
                else:
                    dice_art = 0.0
                    cldice_art = 0.0
                    fp_pct_art_pred = 0.0
                    fn_pct_art_gt = 0.0

                # -------------------------
                # VEIN (class = 2)
                # -------------------------
                g_vein = (mask_np == 2)
                if g_vein.any():
                    p_vein = (label == 2)

                    tp_v = np.logical_and(p_vein, g_vein).sum()
                    fp_v = np.logical_and(p_vein, np.logical_not(g_vein)).sum()
                    fn_v = np.logical_and(np.logical_not(p_vein), g_vein).sum()

                    denom_v = p_vein.sum() + g_vein.sum()
                    dice_vein = 2.0 * tp_v / (denom_v + 1e-5) if denom_v > 0 else 0.0
                    cldice_vein = hard_cldice(p_vein.astype(bool), g_vein.astype(bool))

                    pred_pos_v = tp_v + fp_v
                    gt_pos_v   = tp_v + fn_v

                    fp_pct_vein_pred = 100.0 * fp_v / (pred_pos_v + 1e-5) if pred_pos_v > 0 else 0.0
                    fn_pct_vein_gt   = 100.0 * fn_v / (gt_pos_v + 1e-5)   if gt_pos_v > 0 else 0.0
                else:
                    dice_vein = 0.0
                    cldice_vein = 0.0
                    fp_pct_vein_pred = 0.0
                    fn_pct_vein_gt = 0.0

                case_name = image_path.name.split(".")[0]

                logger.info(
                    f"{case_name}: "
                    f"Dice(A∪V)={dice_union:.4f} clDice(A∪V)={cldice_union:.4f} "
                    f"Dice(art)={dice_art:.4f} clDice(art)={cldice_art:.4f} "
                    f"Dice(vein)={dice_vein:.4f} clDice(vein)={cldice_vein:.4f} | "
                    f"FP%(A∪V|pred)={fp_pct_union_pred:.2f} FN%(A∪V|gt)={fn_pct_union_gt:.2f} | "
                    f"FP%(art|pred)={fp_pct_art_pred:.2f} FN%(art|gt)={fn_pct_art_gt:.2f} | "
                    f"FP%(vein|pred)={fp_pct_vein_pred:.2f} FN%(vein|gt)={fn_pct_vein_gt:.2f}"
                )

                # Store metrics for JSON + summary
                metrics_dict[case_name] = {
                    # union
                    "dice":               torch.tensor(dice_union),
                    "cldice":             torch.tensor(cldice_union),
                    "fp_pct_union_pred":  torch.tensor(fp_pct_union_pred),
                    "fn_pct_union_gt":    torch.tensor(fn_pct_union_gt),
                    # artery
                    "dice_art":           torch.tensor(dice_art),
                    "cldice_art":         torch.tensor(cldice_art),
                    "fp_pct_art_pred":    torch.tensor(fp_pct_art_pred),
                    "fn_pct_art_gt":      torch.tensor(fn_pct_art_gt),
                    # vein
                    "dice_vein":          torch.tensor(dice_vein),
                    "cldice_vein":        torch.tensor(cldice_vein),
                    "fp_pct_vein_pred":   torch.tensor(fp_pct_vein_pred),
                    "fn_pct_vein_gt":     torch.tensor(fn_pct_vein_gt),
                }

    # Summarize over all images
    if mask_paths is not None and len(metrics_dict) > 0:
        # Compute mean for every metric key we stored
        metric_names = list(next(iter(metrics_dict.values())).keys())
        mean_metrics = {}
        for m in metric_names:
            vals = [metrics_dict[k][m].item() for k in metrics_dict]
            mean_metrics[m] = float(np.mean(vals))

        logger.info(f"Mean Dice(A∪V): {mean_metrics['dice']:.4f}")
        logger.info(f"Mean clDice(A∪V): {mean_metrics['cldice']:.4f}")
        logger.info(f"Mean Dice(art): {mean_metrics['dice_art']:.4f}")
        logger.info(f"Mean clDice(art): {mean_metrics['cldice_art']:.4f}")
        logger.info(f"Mean Dice(vein): {mean_metrics['dice_vein']:.4f}")
        logger.info(f"Mean clDice(vein): {mean_metrics['cldice_vein']:.4f}")

        logger.info(f"Mean FP%(A∪V|pred): {mean_metrics['fp_pct_union_pred']:.2f}")
        logger.info(f"Mean FN%(A∪V|gt):   {mean_metrics['fn_pct_union_gt']:.2f}")
        logger.info(f"Mean FP%(art|pred): {mean_metrics['fp_pct_art_pred']:.2f}")
        logger.info(f"Mean FN%(art|gt):   {mean_metrics['fn_pct_art_gt']:.2f}")
        logger.info(f"Mean FP%(vein|pred): {mean_metrics['fp_pct_vein_pred']:.2f}")
        logger.info(f"Mean FN%(vein|gt):   {mean_metrics['fn_pct_vein_gt']:.2f}")

        with open(output_folder / "metrics_per_volume.json", "w") as f:
            json.dump(
                {k: {m: float(v[m].item()) for m in v} for k, v in metrics_dict.items()},
                f,
                indent=2,
            )

        with open(output_folder / "metrics_mean.json", "w") as f:
            json.dump(mean_metrics, f, indent=2)

if __name__ == "__main__":
    main()