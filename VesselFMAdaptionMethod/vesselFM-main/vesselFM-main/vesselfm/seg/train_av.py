import os, random, argparse, json, pathlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from monai.inferers import sliding_window_inference

from .inference import build_model
from .losses import CompositeLoss
from .dataio import make_aug_transforms
from .cldice_utils import SoftCLDiceLoss, hard_cldice
from torch.utils.data import Dataset


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def random_multi_crop_3d(img, lab, roi_size, num_samples):
    """
    Randomly crop 3D patches from volumes.

    img: (B, C, D, H, W)
    lab: (B, D, H, W)
    roi_size: [pD, pH, pW]
    num_samples: patches per volume

    returns:
        img_p: (B * num_samples, C, pD, pH, pW)
        lab_p: (B * num_samples, pD, pH, pW)
    """
    B, C, D, H, W = img.shape
    pD, pH, pW = roi_size
    assert pD <= D and pH <= H and pW <= W, (
        f"Patch size {roi_size} is larger than volume {(D, H, W)}"
    )

    device = img.device
    img_p = torch.empty(
        (B * num_samples, C, pD, pH, pW),
        dtype=img.dtype,
        device=device,
    )
    lab_p = torch.empty(
        (B * num_samples, pD, pH, pW),
        dtype=lab.dtype,
        device=lab.device,
    )

    out_idx = 0
    for b in range(B):
        for _ in range(num_samples):
            z = torch.randint(0, D - pD + 1, (1,), device=device).item()
            y = torch.randint(0, H - pH + 1, (1,), device=device).item()
            x = torch.randint(0, W - pW + 1, (1,), device=device).item()

            img_p[out_idx] = img[b, :, z:z + pD, y:y + pH, x:x + pW]
            lab_p[out_idx] = lab[b, z:z + pD, y:y + pH, x:x + pW]
            out_idx += 1

    return img_p, lab_p


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
def eval_epoch(model, loader, device, cldice_metric=None, patch_size=None):
    """
    Evaluation on full volumes via sliding-window inference.
    """
    model.eval()
    dice_scores = []
    cldice_scores = []

    for batch in loader:
        img, lab = batch["image"].to(device), batch["label"].to(device).long()

        if patch_size is not None:
            logits = sliding_window_inference(
                img,                 # (B,C,D,H,W)
                roi_size=patch_size, # e.g. [96, 96, 96] from YAML
                sw_batch_size=2,
                predictor=model,
            )
        else:
            logits = model(img)

        probs = F.softmax(logits, dim=1)
        pred = probs.argmax(1)  # (B,D,H,W)

        B = pred.shape[0]
        for b in range(B):
            p = (pred[b] > 0)
            g = (lab[b] > 0)

            inter = (p & g).sum().float()
            denom = p.sum().float() + g.sum().float()
            dice = (2.0 * inter) / (denom + 1e-5)
            dice_scores.append(dice.item())

            if cldice_metric is not None:
                # Wrap as (1,1,D,H,W) so we can use interpolate
                p_t = p.float().unsqueeze(0).unsqueeze(0)
                g_t = g.float().unsqueeze(0).unsqueeze(0)

                # Choose a downsample factor (2 is usually safe; try 4 if still too slow)
                ds_factor = 2

                p_ds = torch.nn.functional.interpolate(
                    p_t,
                    scale_factor=1.0 / ds_factor,
                    mode="nearest",
                )[0, 0] > 0.5
                g_ds = torch.nn.functional.interpolate(
                    g_t,
                    scale_factor=1.0 / ds_factor,
                    mode="nearest",
                )[0, 0] > 0.5

                p_np = p_ds.cpu().numpy().astype(bool)
                g_np = g_ds.cpu().numpy().astype(bool)

                cld = cldice_metric(p_np, g_np)
                cldice_scores.append(cld)

    mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
    mean_cldice = float(np.mean(cldice_scores)) if cldice_scores else 0.0
    return mean_dice, mean_cldice


class NPZVolume(Dataset):
    """
    Dataset for nnUNet-preprocessed .npz cases.

    Expects each .npz to contain:
      - 'image': (C, X, Y, Z), float32
      - 'label': (1, X, Y, Z) or (X, Y, Z), int
    """
    def __init__(self, npz_dir, cfg, train=True, transform=None):
        self.npz_dir = pathlib.Path(npz_dir)
        self.cfg = cfg
        self.train = train
        self.transform = transform

        self.files = sorted(self.npz_dir.glob("*.npz"))
        if not self.files:
            raise RuntimeError(
                f"No .npz files found in {self.npz_dir}. "
                f"Make sure you ran the nnUNet preprocessor and pointed "
                f"cfg['data']['train_images'] / ['val_images'] to that folder."
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fpath = self.files[idx]
        data = np.load(fpath, allow_pickle=True)

        img = data["image"]  # (C, X, Y, Z)
        lab = data["label"]  # (1, X, Y, Z) or (X, Y, Z)

        # Convert to torch tensors
        img = torch.from_numpy(img).float()  # keep (C, D, H, W)

        # Remove channel dim for labels so train_av's code sees (D, H, W)
        lab = np.array(lab)
        if lab.ndim == 4 and lab.shape[0] == 1:
            lab = lab[0]  # (D, H, W)
        lab = torch.from_numpy(lab).long()

        sample = {
            "image": img,
            "label": lab,
            "case_id": fpath.stem,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample



def make_loader(kind, cfg, train=True):
    """
    Loader for nnUNet-preprocessed .npz volumes.

    cfg["data"]["train_images"] and cfg["data"]["val_images"] should point
    to folders that contain the .npz files produced by the modified
    DefaultPreprocessor (image+label+properties).
    """
    if kind == "train":
        npz_dir = cfg["data"]["train_images"]
    else:
        npz_dir = cfg["data"]["val_images"]

    aug = make_aug_transforms(cfg, train=train)  # make sure this no longer does file loading itself

    ds = NPZVolume(npz_dir, cfg, train=train, transform=aug)

    return DataLoader(
        ds,
        batch_size=cfg["optim"]["batch_size"],
        shuffle=train,
        num_workers=8,  # adjust if needed
        pin_memory=True,
    )



def main(cfg):
    history = {
        "epoch": [],
        "stage": [],
        "train_loss": [],
        "val_dice": [],      # Dice on vessel union (A∪V vs BG)
        "val_clDice": [],    # hard clDice on vessel union (A∪V vs BG)
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg["seed"])

    # Patch-based training settings from YAML (av_ct.yaml)
    patch_size = cfg["data"].get("patch_size", [96, 96, 96])
    samples_per_volume = cfg["data"].get("samples_per_volume", 1)

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
        soft_cldice_weight=0.0,  # we now handle clDice explicitly below
        soft_cldice_iters=cfg["loss"]["soft_cldice_iters"],
    ).to(device)

    # Optional clDice component on union-of-vessels (A ∪ V vs BG)
    cldice_weight = cfg["loss"]["soft_cldice_weight"] if cfg["loss"]["use_soft_cldice"] else 0.0
    cldice_loss_fn = None
    if cldice_weight > 0:
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

    best = -1.0
    for epoch in range(cfg["optim"]["epochs_stage1"]):
        tr = one_epoch(
            model, train_loader, loss_fn, opt, scaler, device,
            amp=cfg["optim"]["amp"],
            cldice_loss_fn=cldice_loss_fn,
            cldice_weight=cldice_weight,
            patch_size=patch_size,
            samples_per_volume=samples_per_volume,
        )
        va_dice, va_cldice = eval_epoch(
            model, val_loader, device,
            cldice_metric=hard_cldice,
            patch_size=patch_size,
        )

        history["epoch"].append(epoch + 1)
        history["stage"].append("S1")
        history["train_loss"].append(tr)
        history["val_dice"].append(va_dice)
        history["val_clDice"].append(va_cldice)

        if va_dice > best:
            best = va_dice
            torch.save(model.state_dict(), f"checkpoints/{cfg['experiment']}_best_stage1.pt")

        print(
            f"[S1][{epoch+1}/{cfg['optim']['epochs_stage1']}] "
            f"loss={tr:.4f} valDice(A∪V)={va_dice:.4f} valClDice(A∪V)={va_cldice:.4f}"
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
            model, train_loader, loss_fn, opt, scaler, device,
            amp=cfg["optim"]["amp"],
            cldice_loss_fn=cldice_loss_fn,
            cldice_weight=cldice_weight,
            patch_size=patch_size,
            samples_per_volume=samples_per_volume,
        )
        va_dice, va_cldice = eval_epoch(
            model, val_loader, device,
            cldice_metric=hard_cldice,
            patch_size=patch_size,
        )

        history["epoch"].append(epoch + 1)
        history["stage"].append("S2")
        history["train_loss"].append(tr)
        history["val_dice"].append(va_dice)
        history["val_clDice"].append(va_cldice)

        if va_dice > best:
            best = va_dice
            torch.save(model.state_dict(), f"checkpoints/{cfg['experiment']}_best_stage2.pt")

        print(
            f"[S2][{epoch+1}/{cfg['optim']['epochs_stage1']}] "
            f"loss={tr:.4f} valDice(A∪V)={va_dice:.4f} valClDice(A∪V)={va_cldice:.4f}"
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