import os, random, argparse, json, pathlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from monai.inferers import sliding_window_inference
import multiprocessing as mp
from multiprocessing import Pool, cpu_count

from .inference import build_model
from .losses import CompositeLoss
from .dataio import make_aug_transforms, NiftiVolume
from .cldice_utils import SoftCLDiceLoss, hard_cldice
from torch.utils.data import Dataset


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def random_multi_crop_3d(
    img,
    lab,
    roi_size,
    num_samples,
    fg_prob: float = 0.5,
):
    """
    Randomly crop 3D patches from volumes, with optional vessel-biased sampling.

    img: (B, C, D, H, W)
    lab: (B, D, H, W)
    roi_size: [pD, pH, pW]
    num_samples: patches per volume
    fg_prob: probability a given patch is forced to contain foreground (label > 0)
    """
    B, C, D, H, W = img.shape
    pD, pH, pW = roi_size
    assert pD <= D and pH <= H and pW <= W, (
        f"Patch size {roi_size} is larger than volume {(D, H, W)}"
    )

    device = img.device
    img_p = torch.empty((B * num_samples, C, pD, pH, pW),
                        dtype=img.dtype, device=device)
    lab_p = torch.empty((B * num_samples, pD, pH, pW),
                        dtype=lab.dtype, device=lab.device)

    out_idx = 0
    for b in range(B):
        # foreground voxel indices for this volume (vessel union)
        fg_mask = (lab[b] > 0)
        fg_idx = torch.nonzero(fg_mask, as_tuple=False)  # (N_fg, 3) [z,y,x]

        for _ in range(num_samples):
            use_fg = (fg_idx.numel() > 0) and (torch.rand(1, device=device) < fg_prob)

            if use_fg:
                # pick a random foreground voxel as (approximate) patch center
                rand_idx = torch.randint(fg_idx.shape[0], (1,), device=device).item()
                zc, yc, xc = fg_idx[rand_idx].tolist()  # now zc,yc,xc are ints

                # compute start coords so patch stays in bounds
                z = max(0, min(zc - pD // 2, D - pD))
                y = max(0, min(yc - pH // 2, H - pH))
                x = max(0, min(xc - pW // 2, W - pW))
            else:
                # uniform random crop
                z = torch.randint(0, D - pD + 1, (1,), device=device).item()
                y = torch.randint(0, H - pH + 1, (1,), device=device).item()
                x = torch.randint(0, W - pW + 1, (1,), device=device).item()

            img_p[out_idx] = img[b, :, z:z+pD, y:y+pH, x:x+pW]
            lab_p[out_idx] = lab[b, z:z+pD, y:y+pH, x:x+pW]
            out_idx += 1

    return img_p, lab_p


def _compute_cldice_worker(p_np, g_np, cldice_metric):
    """
    Worker for multiprocessing: computes clDice for one case.
    p_np, g_np are downsampled boolean numpy arrays (D, H, W).
    """
    return float(cldice_metric(p_np, g_np))


def freeze_backbone(model):
    """
    For MONAI DynUNet:
      - freeze all encoder/decoder weights
      - leave only the final_conv (classification head) trainable
    """
    for name, p in model.named_parameters():
        if "output_block" in name or "deep_supervision_heads" in name:
            p.requires_grad = True    # head stays trainable
        else:
            p.requires_grad = False   # backbone frozen


def unfreeze_encoder_tail(model, n_stages=2):
    """
    To respect the no backbone retrain constraint, keep this as a no-op.
    Stage 2 just continues training the head with a lower LR.
    """
    # keep head trainable
    if hasattr(model, "output_block"):
        for p in model.output_block.parameters():
            p.requires_grad = True
    if hasattr(model, "deep_supervision_heads"):
        for p in model.deep_supervision_heads.parameters():
            p.requires_grad = True

    # example for DynUNet: unfreeze last decoder level
    if hasattr(model, "decoder"):
        # decoder is typically a nn.ModuleList of stages
        for p in model.decoder[-1].parameters():
            p.requires_grad = True


def one_epoch(
    model,
    loader,
    loss_fn,
    opt,
    scaler,
    device,
    amp=True,
    cldice_loss_fn=None,
    cldice_weight: float = 0.0,
    patch_size=None,
    samples_per_volume: int = 1,
):
    model.train()
    running = []

    for i, batch in enumerate(loader, start=1):
        img, lab = batch["image"].to(device), batch["label"].to(device).long()
        opt.zero_grad(set_to_none=True)

        # Patch-based training
        if patch_size is not None and samples_per_volume > 0:
            img, lab = random_multi_crop_3d(
                img,
                lab,
                roi_size=patch_size,
                num_samples=samples_per_volume,
                fg_prob=0.5,  # 50% of patches vessel-focused
            )

        with autocast(enabled=amp):
            logits = model(img)  # (B*,3,pD,pH,pW)

            # Ensure labels are in [0, num_classes-1]
            n_classes = logits.shape[1]
            invalid = (lab < 0) | (lab >= n_classes)
            if invalid.any():
                lab = lab.clone()
                lab[invalid] = 0

            base_loss = loss_fn(logits, lab)
            loss = base_loss

            # Soft clDice on union-of-vessels (A ∪ V) if enabled
            if cldice_loss_fn is not None and cldice_weight > 0.0:
                probs = F.softmax(logits, dim=1)  # (B*,3,...)

                vessel_lab = (lab > 0).long()           # (B*,D,H,W)
                vessel_gt = vessel_lab.unsqueeze(1).float()  # (B*,1,D,H,W)

                vessel_probs = probs[:, 1:3].sum(dim=1, keepdim=True)
                vessel_probs = vessel_probs.clamp(0.0, 1.0)

                cl_loss = cldice_loss_fn(vessel_gt, vessel_probs)
                loss = loss + cldice_weight * cl_loss

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        running.append(loss.item())

        if i == 1:
            # One-time debug: make sure patches are not full volume
            print("DEBUG train patch shape:", img.shape, lab.shape, flush=True)

        if i % 50 == 0 or i == 1:
            print(f"  [train] batch {i}/{len(loader)}  loss={loss.item():.4f}")

    return float(np.mean(running))


@torch.no_grad()
def eval_epoch(
    model,
    loader,
    device,
    cldice_metric=None,
    patch_size=None,
    num_metric_workers: int = 0,
):
    """
    Evaluation on full volumes via sliding-window inference.

    Returns:
      - mean Dice (union A∪V)
      - mean hard clDice (union A∪V)
      - mean Dice (artery, class=1)
      - mean hard clDice (artery)
      - mean Dice (vein, class=2)
      - mean hard clDice (vein)
    """
    model.eval()

    # Dice accumulators
    dice_union_scores = []
    dice_art_scores = []
    dice_vein_scores = []

    # clDice accumulators
    cldice_union_scores = []
    cldice_art_scores = []
    cldice_vein_scores = []

    # Downsampled masks for clDice
    cldice_cases_union = []
    cldice_cases_art = []
    cldice_cases_vein = []

    ds_factor = 1  # Spatial downsample factor for clDice masks

    for batch in loader:
        img, lab = batch["image"].to(device), batch["label"].to(device).long()

        if patch_size is not None:
            logits = sliding_window_inference(
                img,                 # (B, C, D, H, W)
                roi_size=patch_size, # e.g. [96, 96, 96]
                sw_batch_size=2,
                predictor=model,
            )
        else:
            logits = model(img)

        probs = F.softmax(logits, dim=1)
        pred = probs.argmax(1)  # (B, D, H, W)

        B = pred.shape[0]
        for b in range(B):
            pb = pred[b]
            gb = lab[b]

            # UNION OF VESSELS (A∪V)
            p_union = pb > 0
            g_union = gb > 0

            inter_u = (p_union & g_union).sum().float()
            denom_u = p_union.sum().float() + g_union.sum().float()
            if denom_u > 0:
                dice_u = (2.0 * inter_u) / (denom_u + 1e-5)
                dice_union_scores.append(dice_u.item())

            if cldice_metric is not None:
                p_u_small = p_union[::ds_factor, ::ds_factor, ::ds_factor]
                g_u_small = g_union[::ds_factor, ::ds_factor, ::ds_factor]
                cldice_cases_union.append(
                    (
                        p_u_small.cpu().numpy().astype(bool),
                        g_u_small.cpu().numpy().astype(bool),
                    )
                )

            # ARTERY (class = 1)
            g_art = gb == 1
            if g_art.any():  # Only evaluate arteries if GT has some artery voxels
                p_art = pb == 1

                inter_a = (p_art & g_art).sum().float()
                denom_a = p_art.sum().float() + g_art.sum().float()
                if denom_a > 0:
                    dice_a = (2.0 * inter_a) / (denom_a + 1e-5)
                    dice_art_scores.append(dice_a.item())

                    if cldice_metric is not None:
                        p_a_small = p_art[::ds_factor, ::ds_factor, ::ds_factor]
                        g_a_small = g_art[::ds_factor, ::ds_factor, ::ds_factor]
                        cldice_cases_art.append(
                            (
                                p_a_small.cpu().numpy().astype(bool),
                                g_a_small.cpu().numpy().astype(bool),
                            )
                        )

            # VEIN (class = 2)
            g_vein = gb == 2
            if g_vein.any():  # Only evaluate veins if GT has some vein voxels
                p_vein = pb == 2

                inter_v = (p_vein & g_vein).sum().float()
                denom_v = p_vein.sum().float() + g_vein.sum().float()
                if denom_v > 0:
                    dice_v = (2.0 * inter_v) / (denom_v + 1e-5)
                    dice_vein_scores.append(dice_v.item())

                    if cldice_metric is not None:
                        p_v_small = p_vein[::ds_factor, ::ds_factor, ::ds_factor]
                        g_v_small = g_vein[::ds_factor, ::ds_factor, ::ds_factor]
                        cldice_cases_vein.append(
                            (
                                p_v_small.cpu().numpy().astype(bool),
                                g_v_small.cpu().numpy().astype(bool),
                            )
                        )

    # Helper to compute clDice for a list of (pred, gt) cases
    def _compute_cldice_list(cases):
        if cldice_metric is None or not cases:
            return []
        if num_metric_workers > 0:
            print(
                f"[eval_epoch] clDice over {len(cases)} cases "
                f"with {num_metric_workers} workers",
                flush=True,
            )
            with mp.Pool(processes=num_metric_workers) as pool:
                return pool.starmap(
                    _compute_cldice_worker,
                    [(p_np, g_np, cldice_metric) for (p_np, g_np) in cases],
                )
        else:
            return [
                _compute_cldice_worker(p_np, g_np, cldice_metric)
                for (p_np, g_np) in cases
            ]

    # Compute clDice for union, artery, vein
    cldice_union_scores.extend(_compute_cldice_list(cldice_cases_union))
    cldice_art_scores.extend(_compute_cldice_list(cldice_cases_art))
    cldice_vein_scores.extend(_compute_cldice_list(cldice_cases_vein))

    # Means (fall back to 0.0 if no samples)
    mean_dice_union = float(np.mean(dice_union_scores)) if dice_union_scores else 0.0
    mean_dice_art = float(np.mean(dice_art_scores)) if dice_art_scores else 0.0
    mean_dice_vein = float(np.mean(dice_vein_scores)) if dice_vein_scores else 0.0

    mean_cldice_union = (
        float(np.mean(cldice_union_scores)) if cldice_union_scores else 0.0
    )
    mean_cldice_art = (
        float(np.mean(cldice_art_scores)) if cldice_art_scores else 0.0
    )
    mean_cldice_vein = (
        float(np.mean(cldice_vein_scores)) if cldice_vein_scores else 0.0
    )

    return (
        mean_dice_union,
        mean_cldice_union,
        mean_dice_art,
        mean_cldice_art,
        mean_dice_vein,
        mean_cldice_vein,
    )



def make_items_from_dirs(image_dir, label_dir):
    image_dir = pathlib.Path(image_dir)
    label_dir = pathlib.Path(label_dir)
    items = []

    for img_path in sorted(image_dir.glob("*.nii*")):
        img_name = img_path.name

        # Handle pattern: image_###.nii.gz -> label_###.nii.gz
        if img_name.startswith("image_"):
            lbl_name = "label_" + img_name[len("image_"):]
        else:
            # Fallback: same filename if matching names
            lbl_name = img_name

        lab_path = label_dir / lbl_name
        if not lab_path.exists():
            print(f"WARNING: no label for {img_name}, expected {lab_path}")
            continue

        items.append((str(img_path), str(lab_path)))

    if not items:
        raise RuntimeError(
            f"No image/label pairs found in {image_dir} and {label_dir}. "
            f"Check that filenames follow image_### / label_### pattern."
        )

    return items


def make_loader(kind, cfg, train=True):
    if kind == "train":
        items = make_items_from_dirs(cfg["data"]["train_images"],
                                     cfg["data"]["train_labels"])
    else:
        items = make_items_from_dirs(cfg["data"]["val_images"],
                                     cfg["data"]["val_labels"])

    ds = NiftiVolume(items, cfg, train=train)
    aug = make_aug_transforms(cfg, train=train)
    ds.set_transform(aug)

    return DataLoader(
        ds,
        batch_size=cfg["optim"]["batch_size"],
        shuffle=train,
        num_workers=cfg["data"].get("num_workers", 8),
        pin_memory=True,
    )


def main(cfg):
    history = {
    "epoch": [],
    "stage": [],
    "train_loss": [],
    # Union-of-vessels (A∪V)
    "val_dice": [],
    "val_clDice": [],
    # Class-wise metrics
    "val_dice_artery": [],
    "val_clDice_artery": [],
    "val_dice_vein": [],
    "val_clDice_vein": [],
    }

    val_cldice_workers = cfg["optim"].get("val_cldice_workers", 0)
    print("val_cldice_workers =", val_cldice_workers, flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg["seed"])

    # NiftiVolume already handles patch sampling; disable extra cropping here
    patch_size = None
    samples_per_volume = 1

    # Data
    train_loader = make_loader("train", cfg, train=True)
    val_loader = make_loader("val", cfg, train=False)

    # Debug: grab one batch to make sure loader works
    first_batch = next(iter(train_loader))
    print("DEBUG: first_batch image shape:", first_batch["image"].shape)
    print("DEBUG: first_batch label shape:", first_batch["label"].shape)

    # Model
    model = build_model(
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"].get("dropout", 0.0),
    )

    # ---- Load VesselFM backbone weights, but ignore mismatched head ----
    pre_ckpt = cfg["model"].get("pretrain_ckpt", None)
    if pre_ckpt:
        print(f"Loading pre-trained VesselFM weights from {pre_ckpt}")
        ckpt = torch.load(pre_ckpt, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)

        model_state = model.state_dict()
        filtered_state = {}

        for k, v in state.items():
            if k in model_state and v.shape == model_state[k].shape:
                filtered_state[k] = v
            else:
                # This will include the 1-channel output_block head params
                print(f"Skipping {k}: ckpt {tuple(v.shape)} vs model {tuple(model_state.get(k, torch.empty(0)).shape)}")

        # Now load only compatible weights
        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        print(f"Loaded pre-trained backbone with {len(missing)} missing and {len(unexpected)} unexpected keys")

    model.to(device)


    # Loss (Dice+CE etc.)
    loss_fn = CompositeLoss(
        num_classes=cfg["model"]["num_classes"],
        class_weights=cfg["loss"]["class_weights"],
        soft_cldice_weight=0.0,
        soft_cldice_iters=cfg["loss"]["soft_cldice_iters"],
    ).to(device)

    # clDice weights per stage (from YAML)
    use_soft = cfg["loss"].get("use_soft_cldice", False)
    cldice_weight_s1 = cfg["loss"].get("soft_cldice_weight_stage1", 0.0) if use_soft else 0.0
    cldice_weight_s2 = cfg["loss"].get("soft_cldice_weight_stage2", 0.0) if use_soft else 0.0

    cldice_loss_fn = None
    if (cldice_weight_s1 > 0.0) or (cldice_weight_s2 > 0.0):
        cldice_loss_fn = SoftCLDiceLoss(
            iter_=cfg["loss"]["soft_cldice_iters"], smooth=1.0
        ).to(device)


    # Stage 1: freeze backbone, train head and decoder
    freeze_backbone(model)
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["optim"]["lr_stage1"],
        weight_decay=cfg["optim"]["weight_decay"],
    )
    scaler = GradScaler(enabled=cfg["optim"]["amp"])

    best_cl = -1.0
    for epoch in range(cfg["optim"]["epochs_stage1"]):
        tr = one_epoch(
            model,
            train_loader,
            loss_fn,
            opt,
            scaler,
            device,
            amp=cfg["optim"]["amp"],
            cldice_loss_fn=cldice_loss_fn,
            cldice_weight=cldice_weight_s1,
            patch_size=patch_size,
            samples_per_volume=samples_per_volume,
        )
        (
            va_dice_union,
            va_cldice_union,
            va_dice_art,
            va_cldice_art,
            va_dice_vein,
            va_cldice_vein,
        ) = eval_epoch(
            model,
            val_loader,
            device,
            cldice_metric=hard_cldice,
            patch_size=patch_size,
            num_metric_workers=val_cldice_workers,
        )

        history["epoch"].append(epoch + 1)
        history["stage"].append("S1")
        history["train_loss"].append(tr)

        # union (keep old column names)
        history["val_dice"].append(va_dice_union)
        history["val_clDice"].append(va_cldice_union)

        # NEW: per-class
        history["val_dice_artery"].append(va_dice_art)
        history["val_clDice_artery"].append(va_cldice_art)
        history["val_dice_vein"].append(va_dice_vein)
        history["val_clDice_vein"].append(va_cldice_vein)

        # still pick best model by union clDice (A∪V)
        if va_cldice_union > best_cl:
            best_cl = va_cldice_union
            torch.save(
                model.state_dict(),
                f"checkpoints/{cfg['experiment']}_best_cldice.pt",
            )

        print(
            f"[S1][{epoch+1}/{cfg['optim']['epochs_stage1']}] "
            f"loss={tr:.4f} "
            f"valDice(A∪V)={va_dice_union:.4f} valClDice(A∪V)={va_cldice_union:.4f} "
            f"valDice(art)={va_dice_art:.4f} valClDice(art)={va_cldice_art:.4f} "
            f"valDice(vein)={va_dice_vein:.4f} valClDice(vein)={va_cldice_vein:.4f}"
        )

    # Stage 2: continue training head with lower LR
    unfreeze_encoder_tail(model, n_stages=2)
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["optim"]["lr_stage2"],
        weight_decay=cfg["optim"]["weight_decay"],
    )

    for epoch in range(cfg["optim"]["epochs_stage2"]):
        tr = one_epoch(
            model,
            train_loader,
            loss_fn,
            opt,
            scaler,
            device,
            amp=cfg["optim"]["amp"],
            cldice_loss_fn=cldice_loss_fn,
            cldice_weight=cldice_weight_s2,
            patch_size=patch_size,
            samples_per_volume=samples_per_volume,
        )
        (
            va_dice_union,
            va_cldice_union,
            va_dice_art,
            va_cldice_art,
            va_dice_vein,
            va_cldice_vein,
        ) = eval_epoch(
            model,
            val_loader,
            device,
            cldice_metric=hard_cldice,
            patch_size=patch_size,
            num_metric_workers=val_cldice_workers,
        )

        history["epoch"].append(epoch + 1)
        history["stage"].append("S2")
        history["train_loss"].append(tr)

        history["val_dice"].append(va_dice_union)
        history["val_clDice"].append(va_cldice_union)
        history["val_dice_artery"].append(va_dice_art)
        history["val_clDice_artery"].append(va_cldice_art)
        history["val_dice_vein"].append(va_dice_vein)
        history["val_clDice_vein"].append(va_cldice_vein)

        if va_cldice_union > best_cl:
            best_cl = va_cldice_union
            torch.save(
                model.state_dict(),
                f"checkpoints/{cfg['experiment']}_best_cldice.pt",
            )

        print(
            f"[S2][{epoch+1}/{cfg['optim']['epochs_stage2']}] "
            f"loss={tr:.4f} "
            f"valDice(A∪V)={va_dice_union:.4f} valClDice(A∪V)={va_cldice_union:.4f} "
            f"valDice(art)={va_dice_art:.4f} valClDice(art)={va_cldice_art:.4f} "
            f"valDice(vein)={va_dice_vein:.4f} valClDice(vein)={va_cldice_vein:.4f}"
        )

    try:
        import pandas as pd

        os.makedirs("checkpoints", exist_ok=True)
        pd.DataFrame(history).to_csv(f"checkpoints/{cfg['experiment']}_training_curve.csv", index=False)
    except Exception as e:
        print("Could not save training curve CSV:", e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/av_ct.yaml")
    args = ap.parse_args()
    with open(args.config) as f:
        import yaml

        cfg = yaml.safe_load(f)
    os.makedirs("checkpoints", exist_ok=True)
    main(cfg)