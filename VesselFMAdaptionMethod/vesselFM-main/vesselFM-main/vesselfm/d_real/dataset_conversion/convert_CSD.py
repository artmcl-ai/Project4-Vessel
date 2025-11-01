import SimpleITK as sitk
import os
import numpy as np
from .utils import save_array, save_metadata, calculate_metadata, convert_sitk_image


def convert_CSD(input_folder: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labelsTr"), exist_ok=True)

    image_dir = os.path.join(input_folder, "raw")
    mask_dir = os.path.join(input_folder, "gt")

    print(f"Converting Images...")
    for sample in os.listdir(image_dir):
        print(f"Converting {sample}...")
        image = sitk.ReadImage(os.path.join(image_dir, sample))
        array, metadata = convert_sitk_image(image)
        array = array.astype(np.float32)
        metadata = metadata | calculate_metadata(array)
        sample_name = sample.split(".")[0]
        save_array(array, os.path.join(output_dir, "imagesTr", sample_name))
        save_metadata(metadata, os.path.join(output_dir, "imagesTr", sample_name))

    print(f"Converting Masks...")
    for sample in os.listdir(mask_dir):
        print(f"Converting {sample}...")
        mask = sitk.ReadImage(os.path.join(mask_dir, sample))
        array, metadata = convert_sitk_image(mask)
        array = array > 0
        sample_name = sample.replace("_GT", "").split(".")[0]
        save_array(array, os.path.join(output_dir, "labelsTr", sample_name))
        save_metadata(metadata, os.path.join(output_dir, "labelsTr", sample_name))
