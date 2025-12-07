import os, random, argparse, pathlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from monai.inferers import sliding_window_inference
import multiprocessing as mp

from .inference import build_model
from .dataio import NiftiVolume, make_aug_transforms
from .cldice_utils import SoftCLDiceLoss, hard_cldice


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def _compute_cldice_worker(p_np, g_np, cldice_metric):
    return float(cldice_metric(p_np, g_np))


def make_items_from_dirs(image_dir, label_dir):
    image_dir = pathlib.Path(image_dir)
    label_dir = pathlib.Path(label_dir)
    items = []

    for img_path in sorted(image_dir.glob("*.nii*")):
        img_name = img_path.name
        if img_name.startswith("image_"):
            lbl_name = "label_" + img_name[len("image_"):]
        else:
            lbl_name = img_name

        lab_path = label_dir / lbl_name
        if not lab_path.exists():
            print(f"WARNING: no label for {img_name}, expected {lab_path}")
            continue
        items.append((str(img_path), str(lab_path)))

    if not items:
        raise RuntimeError(
            f"No image/label pairs found in {image_dir} and {label_dir}."
        )

    return items


def make_loader(kind, cfg, train=True):
    if kind == "train":
        items = make_items_from_dirs(
            cfg["data"]["train_images"], cfg["data"]["train_labels"]
        )
    else:
        items = make_items_from_dirs(
            cfg["data"]["val_images"], cfg["data"]["val_labels"]
        )

    ds = NiftiVolume(items, cfg, train=train)
    aug = make_aug_transforms(cfg, train=train)
    ds.set_transform(aug)

    batch_size = cfg["optim"]["batch_size"] if train else 1
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=cfg["data"].get("num_workers", 8),
        pin_memory=True,
    )


def vessel_loss(
    vessel_logits,
    lab,
    bce_weight: float,
    cldice_weight: float,
    cldice_loss_fn=None,
):
    """
    lab: (B,D,H,W) int {0,1,2}, union-of-vessels = (lab>0)
    vessel_logits: (B,1,D,H,W)
    """
    vessel_gt = (lab > 0).float().unsqueeze(1)    # (B,1,D,H,W)
    if vessel_gt.sum() == 0:
        return vessel_logits.new_tensor(0.0)

    loss = 0.0

    if bce_weight > 0.0:
        bce = F.binary_cross_entropy_with_logits(
            vessel_logits, vessel_gt
        )
        loss = loss + bce_weight * bce

    if cldice_loss_fn is not None and cldice_weight > 0.0:
        vessel_probs = torch.sigmoid(vessel_logits)
        cl = cldice_loss_fn(vessel_gt, vessel_probs)
        loss = loss + cldice_weight * cl

    return loss


@torch.no_grad()
def eval_epoch(model, loader, device, patch_size, cldice_metric=None, num_workers: int = 0):
    model.eval()

    dice_scores = []
    cldice_cases = []

    ds_factor = 1

    for batch in loader:
        img, lab = batch["image"].to(device), batch["label"].to(device).long()

        # Full-volume via sliding window
        logits = sliding_window_inference(
            img, roi_size=patch_size, sw_batch_size=2, predictor=model
        )

        # Vessel head
        vessel_logits = model.vessel_head(logits)      # (B,1,D,H,W)
        vessel_probs = torch.sigmoid(vessel_logits)
        pred = (vessel_probs > 0.5).squeeze(1)        # (B,D,H,W)

        gt = (lab > 0)                                # (B,D,H,W)

        B = pred.shape[0]
        for b in range(B):
            pb = pred[b]
            gb = gt[b]
            if gb.sum() == 0:
                continue

            inter = (pb & gb).sum().float()
            denom = pb.sum().float() + gb.sum().float()
            if denom > 0:
                dice = (2.0 * inter) / (denom + 1e-5)
                dice_scores.append(dice.item())

            if cldice_metric is not None:
                p_small = pb[::ds_factor, ::ds_factor, ::ds_factor]
                g_small = gb[::ds_factor, ::ds_factor, ::ds_factor]
                cldice_cases.append(
                    (p_small.cpu().numpy().astype(bool),
                     g_small.cpu().numpy().astype(bool))
                )

    def _compute_cldice_list(cases):
        if cldice_metric is None or not cases:
            return []
        if num_workers > 0:
            with mp.Pool(processes=num_workers) as pool:
                return pool.starmap(
                    _compute_cldice_worker,
                    [(p, g, cldice_metric) for (p, g) in cases],
                )
        else:
            return [
                _compute_cldice_worker(p, g, cldice_metric)
                for (p, g) in cases
            ]

    cldice_scores = _compute_cldice_list(cldice_cases)

    mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
    mean_cl   = float(np.mean(cldice_scores)) if cldice_scores else 0.0
    return mean_dice, mean_cl


def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg["seed"])

    val_cldice_workers = cfg["optim"].get("val_cldice_workers", 0)
    train_loader = make_loader("train", cfg, train=True)
    val_loader   = make_loader("val", cfg, train=False)

    # Just for sanity
    first_batch = next(iter(train_loader))
    print("DEBUG vessel: first_batch image:", first_batch["image"].shape)
    print("DEBUG vessel: first_batch label:", first_batch["label"].shape)

    # Model with 3-class head + vessel_head
    model = build_model(
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"].get("dropout", 0.0),
    )

    # Optional pretrain_ckpt from YAML
    pre_ckpt = cfg["model"].get("pretrain_ckpt", None)
    if pre_ckpt:
        print(f"[Stage1] Loading pretrain_ckpt from {pre_ckpt}")
        ckpt = torch.load(pre_ckpt, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        model_state = model.state_dict()
        filtered = {}
        for k, v in state.items():
            if k in model_state and v.shape == model_state[k].shape:
                filtered[k] = v
            else:
                print(f"[Stage1] Skipping {k}: {v.shape} vs {model_state.get(k, torch.empty(0)).shape}")
        model.load_state_dict(filtered, strict=False)

    model.to(device)

    # Loss hyperparams
    bce_w = cfg["loss"].get("vessel_bce_weight", 0.5)
    cl_w  = cfg["loss"].get("vessel_cldice_weight", 1.0)

    cldice_loss_fn = SoftCLDiceLoss(
        iter_=cfg["loss"]["soft_cldice_iters"], smooth=1.0
    ).to(device)

    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
    )
    scaler = GradScaler(enabled=cfg["optim"]["amp"])

    patch_size = cfg["data"]["patch_size"]
    best_cl = -1.0

    history = {"epoch": [], "train_loss": [], "val_dice": [], "val_clDice": []}

    for epoch in range(cfg["optim"]["epochs"]):
        model.train()
        running = []

        for i, batch in enumerate(train_loader, start=1):
            img, lab = batch["image"].to(device), batch["label"].to(device).long()
            opt.zero_grad(set_to_none=True)

            with autocast(enabled=cfg["optim"]["amp"]):
                logits = model(img)                    # (B,3,D,H,W)
                vessel_logits = model.vessel_head(logits)  # (B,1,D,H,W)

                loss = vessel_loss(
                    vessel_logits,
                    lab,
                    bce_weight=bce_w,
                    cldice_weight=cl_w,
                    cldice_loss_fn=cldice_loss_fn,
                )

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running.append(loss.item())

            if i == 1:
                print("[Stage1] DEBUG train patch:", img.shape, lab.shape, flush=True)

            if i % 50 == 0 or i == 1:
                print(f"[Stage1] epoch {epoch+1} batch {i}/{len(train_loader)} loss={loss.item():.4f}")

        mean_train = float(np.mean(running))
        val_dice, val_cl = eval_epoch(
            model, val_loader, device,
            patch_size=patch_size,
            cldice_metric=hard_cldice,
            num_workers=val_cldice_workers,
        )

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(mean_train)
        history["val_dice"].append(val_dice)
        history["val_clDice"].append(val_cl)

        print(
            f"[Stage1][{epoch+1}/{cfg['optim']['epochs']}] "
            f"train={mean_train:.4f} valDice(vessel)={val_dice:.4f} valClDice(vessel)={val_cl:.4f}"
        )

        if val_cl > best_cl:
            best_cl = val_cl
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(
                model.state_dict(),
                f"checkpoints/{cfg['experiment']}_vessel_best.pt",
            )

    try:
        import pandas as pd
        os.makedirs("checkpoints", exist_ok=True)
        pd.DataFrame(history).to_csv(
            f"checkpoints/{cfg['experiment']}_vessel_curve.csv", index=False
        )
    except Exception as e:
        print("Could not save vessel training curve CSV:", e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/vessel_ct.yaml")
    args = ap.parse_args()
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)