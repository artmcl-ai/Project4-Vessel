import SimpleITK as sitk
import os
from .utils import (
    save_array,
    save_metadata,
    calculate_metadata,
    convert_sitk_image
)


def resample_image(skimage, is_mask: bool = False):
    skimage.SetSpacing([1, 1, 2])
    size = list(skimage.GetSize())
    size[2] = int(size[2] * 2)

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(size)
    resampler.SetOutputSpacing([1, 1, 1])
    resampler.SetTransform(sitk.Transform())
    if is_mask:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)

    image = resampler.Execute(skimage)
    return image


def convert_tof(tof, output_dir_tof, type_="train"):
    print(os.path.join(tof, type_, "raw"))
    for sample in os.listdir(os.path.join(tof, type_, "seg")):
        if not sample.endswith(".nii.gz"):
            continue
        print(sample)
        sample_num = sample.split(".")[0]
        print(f"Convert {sample_num}...")
        annotation = sitk.ReadImage(
            os.path.join(tof, type_, "seg", f"{sample_num}.nii.gz")
        )
        annotation = resample_image(annotation, is_mask=True)
        annotation, metadata = convert_sitk_image(annotation)
        annotation = annotation == 1
        metadata = metadata | calculate_metadata(annotation)
        sample_idx = int(sample_num)
        save_array(
            annotation, os.path.join(output_dir_tof, "labelsTr", f"{type_}_{sample_idx}")
        )
        save_metadata(
            metadata, os.path.join(output_dir_tof, "labelsTr", f"{type_}_{sample_idx}")
        )

        image = sitk.ReadImage(os.path.join(tof, type_, "raw", f"{sample_num}.nii.gz"))
        image = resample_image(image)
        image, metadata = convert_sitk_image(image)
        metadata = metadata | calculate_metadata(image)
        save_array(
            image, os.path.join(output_dir_tof, "imagesTr", f"{type_}_{sample_idx}")
        )
        save_metadata(
            metadata, os.path.join(output_dir_tof, "imagesTr", f"{type_}_{sample_idx}")
        )


def convert_DeepVesselNet(input_folder, output_dir):
    tof = "tof_mra"
    sync = "syncrotone"

    tof = os.path.join(input_folder, tof)
    sync = os.path.join(input_folder, sync)
    output_dir_sync = output_dir + "_syncrotone"
    output_dir_tof = output_dir + "_tof_mra"

    os.makedirs(output_dir_sync, exist_ok=True)
    os.makedirs(os.path.join(output_dir_sync, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(output_dir_sync, "labelsTr"), exist_ok=True)
    os.makedirs(output_dir_tof, exist_ok=True)
    os.makedirs(os.path.join(output_dir_tof, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(output_dir_tof, "labelsTr"), exist_ok=True)

    print(f"Output dir: {output_dir_sync}")
    print(f"Output dir: {output_dir_tof}")

    # syncrotone
    for sample in os.listdir(os.path.join(sync, "label")):
        sample_name = "_".join(sample.split(".")[0].split("_")[1:])
        print(f"Convert {sample_name}...")
        annotation = sitk.ReadImage(os.path.join(sync, "label", sample))
        annotation, metadata = convert_sitk_image(annotation)
        annotation = annotation > 0
        metadata = metadata | calculate_metadata(annotation)
        save_array(
            annotation,
            os.path.join(output_dir_sync, "labelsTr", f"block_{sample_name}"),
        )
        save_metadata(
            metadata, os.path.join(output_dir_sync, "labelsTr", f"block_{sample_name}")
        )

        img_name = f"block_{sample_name}"
        image = sitk.ReadImage(os.path.join(sync, "raw", img_name, "Volume.mhd"))
        image, metadata = convert_sitk_image(image)
        metadata = metadata | calculate_metadata(image)
        save_array(
            image, os.path.join(output_dir_sync, "imagesTr", f"block_{sample_name}")
        )
        save_metadata(
            metadata, os.path.join(output_dir_sync, "imagesTr", f"block_{sample_name}")
        )

    # TOF MRA
    convert_tof(tof, output_dir_tof, type_="train")
    convert_tof(tof, output_dir_tof, type_="test")