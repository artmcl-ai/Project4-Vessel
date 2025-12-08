#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import time
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim

from monai.data import Dataset, DataLoader, CacheDataset, PersistentDataset
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, NormalizeIntensityd,
    RandSpatialCropSamplesd, RandFlipd, RandRotate90d,
    RandGaussianNoised, ScaleIntensityRangeD, Compose
)
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import DynUNet
import torch.nn.functional as F
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, NormalizeIntensityd,
    RandSpatialCropSamplesd, RandFlipd, RandRotate90d,
    RandGaussianNoised, ScaleIntensityRangeD, Compose
)
#from monai.losses import DiceLoss
from monai.transforms import ConcatItemsd
from monai.transforms import CropForegroundd, RandCropByPosNegLabeld

def adapt_checkpoint_for_multichannel(state_dict, model):
    model_sd = model.state_dict()
    new_state = {}

    for k, v in state_dict.items():
        if "output_block" in k:
            print(f"Skipping output head param: {k}")
            continue

        if k not in model_sd:
            print(f"Key not in model, skipping: {k}")
            continue

        tgt = model_sd[k]
        if v.shape == tgt.shape:
            new_state[k] = v
            continue
        if (
            v.ndim == 5 and tgt.ndim == 5 and
            v.shape[0] == tgt.shape[0] and     # same out_channels
            v.shape[2:] == tgt.shape[2:] and   # same kernel size
            v.shape[1] == 1 and tgt.shape[1] == 2
        ):
            print(f"Expanding in_channels 1 -> 2 for {k}: {v.shape} -> {tgt.shape}")
            w = tgt.clone()
            # copy checkpoint weights into channel 0
            w[:, 0:1, ...] = v.clone()
            # initialize channel 1 (prior seg) to zeros
            w[:, 1:2, ...] = 0.0
            new_state[k] = w
            continue
        print(f"Ignoring key due to shape mismatch: {k} {v.shape} -> {tgt.shape}")

    return new_state




class WeightedDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, class_weights=None):
        super().__init__()
        self.smooth = smooth
        self.class_weights = class_weights if class_weights is not None else torch.tensor([0.0, 1.0, 2.0])

    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        target = target.squeeze(1).long()

        target_1h = torch.nn.functional.one_hot(target, num_classes=3)
        target_1h = target_1h.permute(0, 4, 1, 2, 3).float()

        dims = (2, 3, 4)
        intersection = (pred * target_1h).sum(dims)
        union = pred.sum(dims) + target_1h.sum(dims)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        # Apply class weights
        weights = self.class_weights.to(pred.device)
        dice = dice * weights[None, :]  # match batch dimension

        # Remove background contribution: class_weights[0] = 0 ensures this naturally
        # Compute loss
        loss = 1 - dice.mean(dim=1).mean()

        return loss


def combined_loss(pred, target):
    target_ce = target.squeeze(1).long()
    ce = nn.CrossEntropyLoss()(pred, target_ce)
    dice = WeightedDiceLoss(class_weights=torch.tensor([0.0, 1.0, 2.0]))(pred, target)
    return 0.25 * ce + 0.75 * dice



def train_one_epoch(model, loader, optimizer, device, scaler=None):
    print('begin training')
    model.train()
    total_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = combined_loss(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = combined_loss(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device, roi_size, dice_metric):
    print('begin validation')
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

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1, keepdim=True)

            dice_metric(y_pred=preds, y=labels)

    mean_dice = dice_metric.aggregate().item()
    return total_loss / len(loader), mean_dice



def main():
    image_dir = "/projectnb/ec500kb/projects/Project_4/Graph_Creations/Images/"
    mask_dir  = "/projectnb/ec500kb/projects/Project_4/Graph_Creations/Masks_Truth/"
    recon_dir = "/projectnb/ec500kb/projects/Project_4/Graph_Creations/GNN_recon_step2/"
    pretrained_ckpt = "vesselfm_finetuned_multiclass.pt"

    patch_size = (96, 96, 96)
    num_epochs = 50
    batch_size = 1
    num_workers = 4
    cache_rate = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    images = sorted(glob(os.path.join(image_dir, "*.nii*")))
    masks = sorted(glob(os.path.join(mask_dir, "*.nii*")))
    recons = sorted(glob(os.path.join(recon_dir, "*.nii*")))

    val_size = int(0.2 * len(images))
    train_images, val_images = images[val_size:], images[:val_size]
    train_masks,  val_masks  = masks[val_size:], masks[:val_size]
    train_recons, val_recons = recons[val_size:], recons[:val_size]
    train_files = [{"image": i, "label": m, "recon": n} for i, m, n in zip(train_images, train_masks, train_recons)]
    val_files   = [{"image": i, "label": m, "recon": n} for i, m, n in zip(val_images, val_masks, val_recons)]
    print(f"Train volumes: {len(train_files)} | Val volumes: {len(val_files)}")

    train_transforms = Compose([
        LoadImaged(keys=["image", "label", "recon"]),
        EnsureChannelFirstd(keys=["image", "label", "recon"]),

        # CT preprocessing
        ScaleIntensityRangeD(keys=["image"], a_min=-1000, a_max=400,
                             b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),

        # Reconstruction preprocessing (keep as prior)
        ScaleIntensityRangeD(keys=["recon"], a_min=0, a_max=2,
                             b_min=0.0, b_max=1.0, clip=True),

        CropForegroundd(keys=["image", "label", "recon"], source_key="label"),

        RandCropByPosNegLabeld(
            keys=["image", "label", "recon"],
            label_key="label",
            spatial_size=patch_size,
            pos = 2,
            neg = 1,
            num_samples=3,
        ),

        RandFlipd(keys=["image", "label", "recon"], spatial_axis=[0,1,2], prob=0.5),
        RandRotate90d(keys=["image", "label", "recon"], prob=0.25, max_k=3),
        RandGaussianNoised(keys=["image"], prob=0.1),

        # Final 2-channel image
        ConcatItemsd(keys=["image", "recon"], name="image"),
    ])
    
    val_transforms = Compose([
        LoadImaged(keys=["image", "label", "recon"]),
        EnsureChannelFirstd(keys=["image", "label", "recon"]),
        ScaleIntensityRangeD(keys=["image"], a_min=-1000, a_max=400,
                             b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ScaleIntensityRangeD(keys=["recon"], a_min=0, a_max=2,
                             b_min=0.0, b_max=1.0, clip=True),
        ConcatItemsd(keys=["image", "recon"], name="image"),
    ])

    print("\n=== MODEL INPUT BLOCK TARGET SHAPES ===")
    model = DynUNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=3,
        kernel_size=[[3,3,3]] * 6,
        strides=[[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2]],
        upsample_kernel_size=[[2,2,2]]*5,
        filters=[32,64,128,256,320,320],
        res_block=True,
    )


    if os.path.exists(pretrained_ckpt):
        state_dict = torch.load(pretrained_ckpt, map_location="cpu")
        #state_dict = adapt_checkpoint_for_multichannel(state_dict, model)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
    else:
        print("WARNING: Pretrained checkpoint not found. Training from scratch.")

    model = model.to(device)
    
    freeze_encoder = True
    if freeze_encoder:
        print("Freezing encoder layers...")
        for p in model.input_block.parameters(): p.requires_grad = False
        for p in model.downsamples.parameters(): p.requires_grad = False

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=1e-5
    )

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    dice_metric = DiceMetric(include_background=False, reduction="mean")

    best_dice = -1.0
    save_path = "vesselfm_finetuned_multiclass_weighted_dice.pt"

    train_ds = PersistentDataset(data=train_files, transform=train_transforms, cache_dir="./cache_train")
    val_ds = PersistentDataset(data=val_files, transform=val_transforms, cache_dir="./cache_val")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1,
                            shuffle=False, num_workers=num_workers, pin_memory=True)

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
        val_loss, val_dice = validate(model, val_loader, device, patch_size, dice_metric)

        elapsed = time.time() - start_time
        print(f"[Epoch {epoch+1:03d}/{num_epochs:03d}] "
              f"Train={train_loss:.4f} | Val={val_loss:.4f} | Dice={val_dice:.4f} "
              f"| Time={elapsed:.1f}s")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ New best model saved ({best_dice:.4f})")

    print("Training complete.")
    print(f"Best Dice: {best_dice:.4f}")
    print(f"Model saved at: {save_path}")


if __name__ == "__main__":
    main()


# In[5]:


import os
import torch
import numpy as np
import nibabel as nib

from monai.transforms import (
    LoadImaged, EnsureChannelFirstd,
    ScaleIntensityRangeD, NormalizeIntensityd,
    ConcatItemsd, Compose
)
from monai.inferers import sliding_window_inference
from monai.networks.nets import DynUNet


def load_model(checkpoint_path, device):
    print(f"Loading model weights from {checkpoint_path}")

    model = DynUNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=3,
        kernel_size=[[3, 3, 3]] * 6,
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        upsample_kernel_size=[[2, 2, 2]] * 5,
        filters=[32, 64, 128, 256, 320, 320],
        res_block=True,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def run_segmentation(model, image_path, recon_path, output_path, device,
                     roi_size=(96, 96, 96)):
    print("Preparing transforms...")

    '''infer_transforms = Compose([
        LoadImaged(keys=["image", "recon"]),
        EnsureChannelFirstd(keys=["image", "recon"]),

        ScaleIntensityRangeD(keys=["image"], a_min=-1000, a_max=400,
                             b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),

        ScaleIntensityRangeD(keys=["recon"], a_min=0, a_max=2,
                             b_min=0.0, b_max=1.0, clip=True),

        ConcatItemsd(keys=["image", "recon"], name="image"),
    ])
'''
    
    infer_transforms = Compose([
        LoadImaged(keys=["image", "label", "recon"]),
        EnsureChannelFirstd(keys=["image", "label", "recon"]),
        ScaleIntensityRangeD(keys=["image"], a_min=-1000, a_max=400,
                             b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ScaleIntensityRangeD(keys=["recon"], a_min=0, a_max=2,
                             b_min=0.0, b_max=1.0, clip=True),
        ConcatItemsd(keys=["image", "recon"], name="image"),
    ])
    
    
    data = infer_transforms({"image": image_path, "recon": recon_path})
    image_tensor = data["image"].unsqueeze(0).to(device)

    print("Running sliding window inference...")
    with torch.no_grad():
        logits = sliding_window_inference(
            image_tensor,
            roi_size=roi_size,
            sw_batch_size=1,
            predictor=model,
        )

        probs = torch.softmax(logits, dim=1)
        seg = torch.argmax(probs, dim=1).cpu().numpy().astype(np.uint8)[0]

    # Save segmentation to NIfTI
    print(f"Saving segmentation to: {output_path}")
    img_obj = nib.load(image_path)
    nib.save(nib.Nifti1Image(seg, img_obj.affine), output_path)


def main():
    checkpoint = "vesselfm_finetuned_multiclass_weighted_dice.pt"
    image_path = "/projectnb/ec500kb/projects/Project_4/Graph_Creations/Images/case_001_0000.nii.gz"
    recon_path = "/projectnb/ec500kb/projects/Project_4/Graph_Creations/GNN_recon_step2/case_001_nearest_label_propagated.nii.gz"
    output_path = "prediction1.nii.gz"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(checkpoint, device)
    run_segmentation(model, image_path, recon_path, output_path, device)


if __name__ == "__main__":
    main()


# In[ ]:




