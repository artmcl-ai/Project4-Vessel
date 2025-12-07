# -*- coding: utf-8 -*-
import os
import torch
import nibabel as nib
import numpy as np
from monai.transforms import (
    LoadImage, EnsureChannelFirst, NormalizeIntensity,
    RandSpatialCropSamples, RandFlipd, RandRotate90d,
    RandGaussianNoised, ScaleIntensityRange, Compose
)
from monai.inferers import sliding_window_inference
from monai.networks.nets import DynUNet

def build_vesselfm_dyunet():
    model = DynUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        kernel_size=[[3, 3, 3]] * 6,
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        upsample_kernel_size=[[2, 2, 2]] * 5,
        filters=[32, 64, 128, 256, 320, 320],
        res_block=True,
    )
    return model

infer_transforms = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    ScaleIntensityRange(
            a_min=-1000,
            a_max=400,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
    NormalizeIntensity(nonzero=True, channel_wise=True),
])


def run_inference(model_path, input_nifti, output_nifti, roi_size=(96, 96, 96)):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_vesselfm_dyunet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    img_np = infer_transforms(input_nifti)
    img_torch = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_logits = sliding_window_inference(
            img_torch,
            roi_size=roi_size,
            sw_batch_size=1,
            predictor=model,
            overlap=0.25,
        )

    prob = torch.sigmoid(pred_logits)
    pred_mask = (prob > 0.5).float()
    pred_mask_np = pred_mask.cpu().numpy()[0, 0]
    orig = nib.load(input_nifti)
    pred_nifti = nib.Nifti1Image(pred_mask_np.astype(np.uint8), orig.affine, orig.header)
    nib.save(pred_nifti, output_nifti)
    print(f"Saved prediction mask to: {output_nifti}")

if __name__ == "__main__":
    model_path = "vesselfm_finetuned_monai.pt"
    input_image_path = "/projectnb/ec500kb/projects/Project_4/Graph_Creations/Images/"
    completed = []
    for i in os.listdir(input_image_path):
        output_mask_path = "./Binary_Masks_Predictions/predicted_binary_mask" + i[5:8] + ".nii.gz"
        if os.path.exists(output_mask_path):
            print(f"Skipping {output_mask_path} → Output already exists.")
            continue
        in_path = input_image_path + i
        run_inference(model_path, in_path, output_mask_path)
