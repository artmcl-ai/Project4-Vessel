#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
    infer_transforms = Compose([
        LoadImaged(keys=["image", "recon"]),
        EnsureChannelFirstd(keys=["image", "recon"]),

        ScaleIntensityRangeD(keys=["image"], a_min=-1000, a_max=400,
                             b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),

        ScaleIntensityRangeD(keys=["recon"], a_min=0, a_max=2,
                             b_min=0.0, b_max=1.0, clip=True),

        ConcatItemsd(keys=["image", "recon"], name="image"),
    ])

    data = infer_transforms({"image": image_path, "recon": recon_path})
    image_tensor = data["image"].unsqueeze(0).to(device)
    print("Doing sliding window inference")
    with torch.no_grad():
        logits = sliding_window_inference(
            image_tensor,
            roi_size=roi_size,
            sw_batch_size=1,
            predictor=model)
        probs = torch.softmax(logits, dim=1)
        seg = torch.argmax(probs, dim=1).cpu().numpy().astype(np.uint8)[0]
    print(f"Saving segmentation to: {output_path}")
    img_obj = nib.load(image_path)
    nib.save(nib.Nifti1Image(seg, img_obj.affine), output_path)
    
def main():
    checkpoint = "vesselfm_finetuned_multiclass_weighted_dice.pt"
    image_path = "/projectnb/ec500kb/projects/Project_4/Graph_Creations/Images/case_001_0000.nii.gz"
    recon_path = "output/recon1.nii.gz"
    output_path = "output/prediction1.nii.gz"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint, device)
    run_segmentation(model, image_path, recon_path, output_path, device)
    
if __name__ == "__main__":
    main()

