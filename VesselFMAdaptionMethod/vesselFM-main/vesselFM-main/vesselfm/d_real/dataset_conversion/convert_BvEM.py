import numpy as np
import SimpleITK as sitk
import os
import h5py
from .utils import (
    save_array,
    save_metadata,
    calculate_metadata,
)


def resample_BvEM_label(input_folder):
    label = "/mouse_microns-phase2_256-320nm_crop_bv_v4.h5" # only labels for mouse are publicly available
    with h5py.File(input_folder + label, "r") as f:
        label = f["main"][:]

    slabel = sitk.GetImageFromArray(label)
    slabel.SetSpacing([1, 1, 4])

    # Resample image to new size [2495, 3571, 5145]
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputSpacing([1, 1, 1])
    resampler.SetOutputOrigin(slabel.GetOrigin())
    resampler.SetOutputDirection(slabel.GetDirection())
    resampler.SetDefaultPixelValue(0)
    resampler.SetSize([5145, 3571, 2495])
    resampled_label = resampler.Execute(slabel)

    # turn cropping off as we crop used volumes in patch extraction
    # note: this should be turned on when one uses the full volume for anything
    # resampled_label = resampled_label[0:3874, 618:3058, :]

    sitk.WriteImage(
        resampled_label,
        input_folder + "/mouse_microns-phase2_256-320nm_crop_bv_v4_resampled.nii.gz",
    )


def convert_BvEM(input_folder, output_path):
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "imagesTr", "mouse_microns-phase2_256-320nm_crop"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "labelsTr", "mouse_microns-phase2_256-320nm_crop"), exist_ok=True)

    resample_BvEM_label(input_folder)
    label = "mouse_microns-phase2_256-320nm_crop_bv_v4_resampled.nii.gz"
    image = "mouse_microns-phase2_256-320nm_crop.h5"

    with h5py.File(os.path.join(input_folder, image), "r") as f:
        image_arr = f["main"][:]

    # turn cropping off as we crop used volumes in patch extraction
    # note: this should be turned on when one uses the full volume for anything
    # image_arr = image_arr[0:3874, 618:3058]
    
    save_array(image_arr, os.path.join(output_path, "imagesTr", "mouse_microns-phase2_256-320nm_crop"))
    metadata = []
    for slice_ in range(0, image_arr.shape[0], 100):
        metadata.append(calculate_metadata(image_arr[slice_:slice_+100]))
    metadata = {"max": np.max([m["max"] for m in metadata]), "min": np.min([m["min"] for m in metadata]), "mean": np.mean([m["mean"] for m in metadata]), "std": np.median([m["std"] for m in metadata]), "shape": [len(metadata), image_arr.shape[0], image_arr.shape[1]], "p95": np.median([m["p95"] for m in metadata]), "p5": np.median([m["p5"] for m in metadata])}
    save_metadata(metadata, os.path.join(output_path, "imagesTr", "mouse_microns-phase2_256-320nm_crop"))

    label = sitk.ReadImage(os.path.join(input_folder, "mouse_microns-phase2_256-320nm_crop_bv_v4_resampled.nii.gz"))
    label = sitk.GetArrayFromImage(label)
    assert label.shape == image_arr.shape, f"label shape {label.shape} does not match image shape {image_arr.shape}"
    save_array(label, os.path.join(output_path, "labelsTr", "mouse_microns-phase2_256-320nm_crop"))
