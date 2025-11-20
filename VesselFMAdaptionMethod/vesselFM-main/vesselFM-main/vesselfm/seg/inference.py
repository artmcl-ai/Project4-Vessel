""" Script to perform inference with vesselFM."""

import logging
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
import hydra
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from monai.inferers import SlidingWindowInfererAdapt
from skimage.morphology import remove_small_objects
from skimage.exposure import equalize_hist

from vesselfm.seg.utils.data import generate_transforms
from vesselfm.seg.utils.io import determine_reader_writer
from vesselfm.seg.utils.evaluation import Evaluator, calculate_mean_metrics


warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

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
    image_paths = list(Path(cfg.image_path).iterdir())
    if cfg.mask_path:
        mask_paths = [Path(cfg.mask_path) / f"{p.name}" for p in image_paths]
        assert all(
            mask_path.exists() for mask_path in mask_paths
        ), "All mask paths must exist mask name has to be the same as the image name."
    else:
        mask_paths = None
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



if __name__ == "__main__":
    main()