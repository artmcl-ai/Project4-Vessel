#!/usr/bin/env python

import os
import time
import argparse
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRangeD,
    NormalizeIntensityd,
    RandSpatialCropSamplesd,
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    Compose,
    LambdaD,
)
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import DynUNet


# 1. Loss functions: BCE + Dice (binary)
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred: logits, shape [B,1,D,H,W]
        target: float mask in {0,1}, shape [B,1,D,H,W]
        """
        pred = torch.sigmoid(pred)
        dims = tuple(range(2, pred.ndim))  # sum over spatial dims
        intersection = 2.0 * (pred * target).sum(dim=dims)
        denominator = pred.sum(dim=dims) + target.sum(dim=dims) + self.smooth
        dice = 1.0 - intersection / denominator
        return dice.mean()


def combined_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = nn.BCEWithLogitsLoss()(pred, target)
    dice = DiceLoss()(pred, target)
    return 0.5 * bce + 0.5 * dice


# Training & validation loops
def train_one_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)  # [B,1,D,H,W]
        labels = batch["label"].to(device)  # [B,1,D,H,W], float {0,1}

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = combined_loss(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = combined_loss(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def validate(model, loader, device, roi_size, dice_metric: DiceMetric):
    model.eval()
    total_loss = 0.0
    dice_metric.reset()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = sliding_window_inference(
                images,
                roi_size=roi_size,
                sw_batch_size=1,
                predictor=model,
            )

            loss = combined_loss(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            dice_metric(y_pred=preds, y=labels)

    mean_dice = float(dice_metric.aggregate().item())
    return total_loss / max(len(loader), 1), mean_dice


def main():
    parser = argparse.ArgumentParser(
        description="Binary vessel pretraining (union-of-vessels) with DynUNet + VesselFM base."
    )

    # --- explicit train/val dirs ---
    parser.add_argument(
        "--train_image_dir",
        type=str,
        required=True,
        help="Directory with training .nii/.nii.gz images.",
    )
    parser.add_argument(
        "--train_label_dir",
        type=str,
        required=True,
        help="Directory with training .nii/.nii.gz labels (0=bg, 1=art, 2=vein).",
    )
    parser.add_argument(
        "--val_image_dir",
        type=str,
        required=True,
        help="Directory with validation .nii/.nii.gz images.",
    )
    parser.add_argument(
        "--val_label_dir",
        type=str,
        required=True,
        help="Directory with validation .nii/.nii.gz labels (0=bg, 1=art, 2=vein).",
    )

    parser.add_argument(
        "--pretrained_ckpt",
        type=str,
        default="vesselFM_base.pt",
        help="Path to VesselFM base checkpoint (.pt).",
    )
    parser.add_argument(
        "--output_ckpt",
        type=str,
        default="vesselfm_finetuned_binary.pt",
        help="Path to save best fine-tuned checkpoint.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Training batch size.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        nargs=3,
        default=[96, 96, 96],
        help="Patch size (D H W) for training & sliding window.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Num workers for DataLoader / CacheDataset.",
    )
    parser.add_argument(
        "--cache_rate",
        type=float,
        default=0.5,
        help="Fraction of data to keep in RAM in CacheDataset.",
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="If set, freeze encoder blocks (input_block + downsamples).",
    )
    parser.add_argument(
        "--a_min",
        type=float,
        default=-1000.0,
        help="Lower HU bound for ScaleIntensityRangeD.",
    )
    parser.add_argument(
        "--a_max",
        type=float,
        default=400.0,
        help="Upper HU bound for ScaleIntensityRangeD.",
    )

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_images = sorted(glob(os.path.join(args.train_image_dir, "*.nii*")))
    train_labels = sorted(glob(os.path.join(args.train_label_dir, "*.nii*")))
    assert len(train_images) == len(train_labels), "Train images and labels must match in count."
    assert len(train_images) > 0, "No training NIfTI files found."

    # --- build val file list ---
    val_images = sorted(glob(os.path.join(args.val_image_dir, "*.nii*")))
    val_labels = sorted(glob(os.path.join(args.val_label_dir, "*.nii*")))
    assert len(val_images) == len(val_labels), "Val images and labels must match in count."
    assert len(val_images) > 0, "No validation NIfTI files found."

    train_files = [{"image": i, "label": l} for i, l in zip(train_images, train_labels)]
    val_files = [{"image": i, "label": l} for i, l in zip(val_images, val_labels)]

    print(f"Train volumes: {len(train_files)} | Val volumes: {len(val_files)}")

    patch_size = tuple(args.patch_size)

    # MONAI transforms (labels: 0/1/2 -> binary union-of-vessels)
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),

        # Optional per-volume z-score on non-zero voxels
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),

        # Convert multi-class labels {0,1,2} -> binary {0,1}
        LambdaD(
            keys=["label"],
            func=lambda x: (x > 0).astype(np.float32),
        ),

        RandSpatialCropSamplesd(
            keys=["image", "label"],
            roi_size=patch_size,
            num_samples=4,
            random_center=True,
            random_size=False,
        ),

        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.5),

        RandRotate90d(keys=["image", "label"], prob=0.25, max_k=3),
        RandGaussianNoised(keys=["image"], prob=0.1),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),

        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        LambdaD(
            keys=["label"],
            func=lambda x: (x > 0).astype(np.float32),
        ),
    ])

    # Datasets & DataLoaders
    train_ds = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=args.cache_rate,
        num_workers=args.num_workers,
    )
    val_ds = CacheDataset(
        data=val_files,
        transform=val_transforms,
        cache_rate=args.cache_rate,
        num_workers=args.num_workers,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model: DynUNet with VesselFM config
    model = DynUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,  # binary vessel mask
        kernel_size=[[3, 3, 3]] * 6,
        strides=[
            [1, 1, 1],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
        ],
        upsample_kernel_size=[[2, 2, 2]] * 5,
        filters=[32, 64, 128, 256, 320, 320],
        res_block=True,
    )

    # Load VesselFM base weights
    if os.path.exists(args.pretrained_ckpt):
        print(f"Loading pretrained weights from {args.pretrained_ckpt}")
        state_dict = torch.load(args.pretrained_ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
    else:
        print(f"WARNING: Pretrained checkpoint not found: {args.pretrained_ckpt}")
        print("         Training from scratch.")

    model = model.to(device)

    # Optionally freeze encoder
    if args.freeze_encoder:
        print("Freezing encoder layers (input_block + downsamples)...")
        for p in model.input_block.parameters():
            p.requires_grad = False
        for p in model.downsamples.parameters():
            p.requires_grad = False

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-5,
    )

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # Training loop
    best_dice = -1.0

    print("Starting training...")
    for epoch in range(args.epochs):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
        val_loss, val_dice = validate(model, val_loader, device, patch_size, dice_metric)

        elapsed = time.time() - start_time
        print(
            f"[Epoch {epoch+1:03d}/{args.epochs:03d}] "
            f"TrainLoss={train_loss:.4f} | "
            f"ValLoss={val_loss:.4f} | "
            f"ValDice={val_dice:.4f} | "
            f"Time={elapsed:.1f}s"
        )

        # Save best model by Dice
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), args.output_ckpt)
            print(f"  ✓ New best model saved to {args.output_ckpt} (Dice={best_dice:.4f})")

    print("Training complete.")
    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"Best model path: {args.output_ckpt}")


if __name__ == "__main__":
    main()