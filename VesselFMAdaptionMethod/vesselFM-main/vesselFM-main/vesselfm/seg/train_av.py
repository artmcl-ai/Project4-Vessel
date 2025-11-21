import os, csv, random, argparse, json
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from .inference import build_model
from ..avseg.losses import CompositeLoss, SoftClDiceLoss

# Dataset Utilities
from .dataio import NiftiVolume, make_aug_transforms

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def freeze_backbone(model):
    """
    For MONAI DynUNet, treat everything except the output (and optional deep
    supervision heads) as the backbone. Freeze all those params.
    """
    for name, p in model.named_parameters():
        # Keep only the segmentation head trainable
        if "output_block" in name or "deep_supervision_heads" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False

def unfreeze_encoder_tail(model, n_stages=2):
    """
    Stage 2 in train_av.py continues training the head with a lower LR.
    """
    return

def one_epoch(model, loader, loss_fn, opt, scaler, device, amp=True):
    model.train()
    running = []
    for batch in loader:
        img, lab = batch["image"].to(device), batch["label"].to(device).long()
        opt.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            logits = model(img)  # (B,3,D,H,W)
            loss = loss_fn(logits, lab)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        running.append(loss.item())
    return float(np.mean(running))

@torch.no_grad()
def eval_epoch(model, loader, device, soft_cl=None):
    model.eval()
    dices = []
    cldices = []
    for batch in loader:
        img, lab = batch["image"].to(device), batch["label"].to(device).long()
        logits = model(img)
        probs = F.softmax(logits, dim=1)
        pred = probs.argmax(1)

        # Dice per class (excluding background)
        eps = 1e-5
        ds = []
        for c in [1, 2]:
            inter = ((pred == c) & (lab == c)).sum().float()
            denom = (pred == c).sum().float() + (lab == c).sum().float()
            ds.append((2 * inter + eps) / (denom + eps))
        dices.append(torch.stack(ds).mean().item())

        # Soft clDice on vessel union (A ∪ V)
        if soft_cl is not None:
            onehot = torch.zeros_like(probs).scatter_(1, lab.unsqueeze(1), 1.0)
            cl_loss = soft_cl(probs, onehot)       # This returns 1 - clDice
            cldices.append(1.0 - cl_loss.item())

    mean_dice = float(np.mean(dices)) if dices else 0.0
    mean_cldice = float(np.mean(cldices)) if cldices else 0.0
    return mean_dice, mean_cldice


def make_loader(csv_path, cfg, train=True):
    items=[]
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            items.append((row["image"], row["label"]))
    ds = NiftiVolume(items, cfg)
    aug = make_aug_transforms(cfg, train=train)
    ds.set_transform(aug)
    return DataLoader(ds, batch_size=cfg["optim"]["batch_size"], shuffle=train,
                      num_workers=4, pin_memory=True)

def main(cfg):
    history = {"epoch": [], "stage": [], "train_loss": [], "val_dice": [], "val_clDice": []}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg["seed"])
    # Data
    train_loader = make_loader(cfg["data"]["train_csv"], cfg, train=True)
    val_loader   = make_loader(cfg["data"]["val_csv"],   cfg, train=False)

    # Model
    model = build_model(num_classes=cfg["model"]["num_classes"], dropout=cfg["model"].get("dropout",0.0))
    model.to(device)

    # Loss
    loss_fn = CompositeLoss(
        num_classes=cfg["model"]["num_classes"],
        class_weights=cfg["loss"]["class_weights"],
        soft_cldice_weight=cfg["loss"]["soft_cldice_weight"] if cfg["loss"]["use_soft_cldice"] else 0.0,
        soft_cldice_iters=cfg["loss"]["soft_cldice_iters"],
    ).to(device)

    # clDice metric (union vessel)
    soft_cl_metric = SoftClDiceLoss(iters=cfg["loss"]["soft_cldice_iters"]).to(device)

    # Stage 1: freeze backbone, train head and decoder
    freeze_backbone(model)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=cfg["optim"]["lr_stage1"], weight_decay=cfg["optim"]["weight_decay"])
    scaler = torch.amp.GradScaler(enabled=cfg["optim"]["amp"])

    best = -1.0
    for epoch in range(cfg["optim"]["epochs_stage1"]):
        tr = one_epoch(model, train_loader, loss_fn, opt, scaler, device, amp=cfg["optim"]["amp"])
        va_dice, va_cldice = eval_epoch(model, val_loader, device, soft_cl=soft_cl_metric)
        history["epoch"].append(epoch + 1)
        history["stage"].append("S1")
        history["train_loss"].append(tr)
        history["val_dice"].append(va_dice)
        history["val_clDice"].append(va_cldice)
        if va_dice > best:
            best = va_dice
            torch.save(model.state_dict(), f"checkpoints/{cfg['experiment']}_best_stage1.pt")
        print(f"[S1][{epoch+1}/{cfg['optim']['epochs_stage1']}] "
            f"loss={tr:.4f} valDice={va_dice:.4f} valClDice={va_cldice:.4f}")

    # Stage 2: partial unfreeze
    unfreeze_encoder_tail(model, n_stages=2)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=cfg["optim"]["lr_stage2"], weight_decay=cfg["optim"]["weight_decay"])
    for epoch in range(cfg["optim"]["epochs_stage2"]):
        history["epoch"].append(epoch + 1)
        history["stage"].append("S2")
        history["train_loss"].append(tr)
        history["val_dice"].append(va_dice)
        history["val_clDice"].append(va_cldice)
        tr = one_epoch(...)
        va_dice, va_cldice = eval_epoch(model, val_loader, device, soft_cl=soft_cl_metric)
        if va_dice > best:
            best = va_dice
            torch.save(model.state_dict(), f"checkpoints/{cfg['experiment']}_best.pt")
        print(f"[S2][{epoch+1}/{cfg['optim']['epochs_stage2']}] "
            f"loss={tr:.4f} valDice={va_dice:.4f} valClDice={va_cldice:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/av_ct.yaml")
    args = ap.parse_args()
    with open(args.config) as f:
        import yaml; cfg = yaml.safe_load(f)
    os.makedirs("checkpoints", exist_ok=True)
    main(cfg)