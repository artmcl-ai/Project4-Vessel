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
    prob_mask_np = prob.cpu().numpy()[0, 0]
    #pred_mask = (prob > 0.5).float()
    orig = nib.load(input_nifti)
    pred_nifti = nib.Nifti1Image(prob_mask_np.astype(np.float32), orig.affine, orig.header)
    nib.save(pred_nifti, output_nifti) #THIS GIVES THE PROBABILITY MATRIX 

    print(f"Saved prediction mask to: {output_nifti}")

if __name__ == "__main__":
    import sys
    import argparse
    from pathlib import Path
    import shutil

    parser = argparse.ArgumentParser(description="VesselFM Probability Map Generator")
    parser.add_argument("-i", "--input", required=True, type=Path, help="Input file or directory")
    parser.add_argument("-o", "--output-dir", required=True, type=Path, help="Output directory")
    parser.add_argument("-m", "--model", required=True, type=Path, help="Model weights path")

    args = parser.parse_args()

    # Collect input files
    if args.input.is_file():
        if not str(args.input).endswith('.nii.gz'):
            raise ValueError(f"Input must be .nii.gz format: {args.input}")
        input_files = [args.input]
    elif args.input.is_dir():
        input_files = sorted(args.input.glob("*.nii.gz"))
        if not input_files:
            raise ValueError(f"No .nii.gz files found in {args.input}")
    else:
        raise ValueError(f"Input does not exist: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(input_files)} file(s) with VesselFM...")

    for input_file in input_files:
        case_name = input_file.stem.replace('.nii', '')
        for suffix in ['_0000', '_0001', '_0002', '_0003']:
            if case_name.endswith(suffix):
                case_name = case_name[:-5]
                break

        # Channel 0: CT (copy original)
        ct_output = args.output_dir / f"{case_name}_0000.nii.gz"
        shutil.copy(input_file, ct_output)

        # Channel 1: VesselFM probability
        prob_output = args.output_dir / f"{case_name}_0001.nii.gz"
        run_inference(str(args.model), str(input_file), str(prob_output))

    print(f"VesselFM processing complete. Output saved to: {args.output_dir}")