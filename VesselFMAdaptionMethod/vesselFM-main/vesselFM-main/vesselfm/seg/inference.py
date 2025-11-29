""" Script to perform inference with vesselFM."""

import logging
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
import hydra
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra

import numpy as np
import json
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from monai.inferers import SlidingWindowInfererAdapt
from skimage.morphology import remove_small_objects
from skimage.exposure import equalize_hist

from vesselfm.seg.utils.data import generate_transforms
from vesselfm.seg.utils.io import determine_reader_writer
from vesselfm.seg.utils.evaluation import Evaluator, calculate_mean_metrics

from omegaconf import OmegaConf


warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

def build_model(num_classes=3, dropout=0.0):
    # Load inference config to get the same model definition with ckpt_path
    here = Path(__file__).resolve().parent
    config_dir = here / "configs"  # -> vesselfm/seg/configs

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

    return model


def load_model(cfg, device):
    try:
        logger.info(f"Loading model from {cfg.ckpt_path}.")
        ckpt = torch.load(Path(cfg.ckpt_path), map_location=device, weights_only=True)
    except:
        logger.info(f"Loading model from Hugging Face.")
        hf_hub_download(repo_id='bwittmann/vesselFM', filename='meta.yaml') # required to track downloads
        ckpt = torch.load(
            hf_hub_download(repo_id='bwittmann/vesselFM', filename='vesselFM_base.pt'),
            map_location=device, weights_only=True
        )

    model = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(ckpt, strict=False)
    return model

def get_paths(cfg):
    """
    Collect image and mask paths.

    Supports config layouts:
      - cfg.image_dir / cfg.mask_dir
      - cfg.data.image_dir / cfg.data.mask_dir

    Supports filename conventions:
      1) Same-name masks:
           image_004.nii.gz -> image_004.nii.gz
      2) image/label naming:
           image_004.nii.gz -> label_004.nii.gz
    """
    import os
    import glob

    # --- 1. Read directories from config, but don't assume 'data' exists ---
    # Try nested first, then fall back to top-level.
    image_dir = OmegaConf.select(cfg, "data.image_dir")
    if image_dir is None:
        image_dir = OmegaConf.select(cfg, "image_dir")

    mask_dir = OmegaConf.select(cfg, "data.mask_dir")
    if mask_dir is None:
        mask_dir = OmegaConf.select(cfg, "mask_dir")

    if image_dir is None:
        raise RuntimeError(
            "image_dir not set in config (looked for 'image_dir' and 'data.image_dir')."
        )

    # --- 2. Collect images ---
    # Use *.nii* so it works for .nii and .nii.gz
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.nii*")))
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    # If no mask_dir (pure inference), just return images
    if mask_dir in (None, "", "null"):
        return image_paths, None

    # --- 3. Build mask paths with both naming schemes ---
    mask_paths = []

    for img_path in image_paths:
        img_name = os.path.basename(img_path)          # e.g. "image_004.nii.gz"

        # (a) First try: mask has EXACT same basename as image
        same_name_mask = os.path.join(mask_dir, img_name)
        if os.path.exists(same_name_mask):
            mask_paths.append(same_name_mask)
            continue

        # (b) Second try: image_XXX.nii.gz -> label_XXX.nii.gz
        alt_mask = None
        if img_name.startswith("image_"):
            suffix = img_name[len("image_"):]          # "004.nii.gz"
            alt_mask = os.path.join(mask_dir, "label_" + suffix)
            if os.path.exists(alt_mask):
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
    if factor == 1:
        return image
    
    if target_shape:
        _, _, new_d, new_h, new_w = target_shape
    else:
        _, _, d, h, w = image.shape
        new_d, new_h, new_w = int(round(d / factor)), int(round(h / factor)), int(round(w / factor))
    return F.interpolate(image, size=(new_d, new_h, new_w), mode="trilinear", align_corners=False)

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
            preds = []  # per-scale logits
            mask = None

            for scale in cfg.tta.scales:
                # read image (and mask if available)
                image_np = image_reader_writer.read_images(image_path)[0].astype(np.float32)
                image = transforms(image_np)[None].to(device)

                if mask_paths is not None:
                    mask_np = image_reader_writer.read_images(mask_paths[idx])[0]
                    mask = torch.tensor(mask_np).bool()

                # TTA intensity transforms
                if cfg.tta.invert:
                    if image.mean() > cfg.tta.invert_mean_thresh:
                        image = 1 - image
                if cfg.tta.equalize_hist:
                    image_np = image.cpu().squeeze().numpy()
                    image_equal_hist_np = equalize_hist(image_np, nbins=cfg.tta.hist_bins)
                    image = torch.from_numpy(image_equal_hist_np).to(image.device)[None][None]

                # resample for scale, run model, resample back
                original_shape = image.shape
                image_scaled = resample(image, factor=scale)
                logits = inferer(image_scaled, model)                     # (1,3,D,H,W)
                logits = resample(logits, target_shape=original_shape)    # back to original patch grid
                preds.append(logits.cpu().squeeze())                      # (3,D,H,W)

            # Merge TTA scales (multiclass A/V/BG)
            if cfg.merging.max:
                probs = torch.stack([F.softmax(p, dim=0) for p in preds]).max(dim=0)[0]   # (3,D,H,W)
            else:
                probs = torch.stack([F.softmax(p, dim=0) for p in preds]).mean(dim=0)    # (3,D,H,W)

            # Argmax -> labelmap {0:bg, 1:artery, 2:vein}
            label = probs.argmax(0).cpu().numpy().astype(np.uint8)                        # (D,H,W)

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

            # Save final labelmap
            save_writer.write_seg(
                label,
                output_folder / f"{image_path.name.split('.')[0]}_{cfg.file_app}pred.{file_ending}",
            )

            # Metrics (if GT masks available)
            if mask_paths is not None and mask is not None:
                # Union vessel probability = 1 - P(background)
                union_prob = 1.0 - probs[0]  # probs[0] is class 0 = background
                metrics = Evaluator().estimate_metrics(
                    union_prob, mask, threshold=cfg.merging.threshold
                )
                logger.info(f"Dice of {image_path.name.split('.')[0]}: {metrics['dice'].item()}")
                logger.info(f"clDice of {image_path.name.split('.')[0]}: {metrics['cldice'].item()}")
                metrics_dict[image_path.name.split('.')[0]] = metrics

    # Summarize over all images
    if mask_paths is not None and len(metrics_dict) > 0:
        mean_metrics = calculate_mean_metrics(metrics_dict)
        logger.info(f"Mean Dice: {mean_metrics['dice']:.4f}")
        logger.info(f"Mean clDice: {mean_metrics['cldice']:.4f}")
        with open(output_folder / "metrics_per_volume.json", "w") as f:
            json.dump(
                {k: {m: float(v[m].item()) for m in v} for k, v in metrics_dict.items()},
                f,
                indent=2,
            )
        with open(output_folder / "metrics_mean.json", "w") as f:
            json.dump(
                {m: float(mean_metrics[m]) for m in mean_metrics},
                f,
                indent=2,
            )

if __name__ == "__main__":
    main()