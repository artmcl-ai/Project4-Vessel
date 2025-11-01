import numpy as np
import SimpleITK as sitk
import os
from monai.transforms import Resize
from .utils import save_array, save_metadata, calculate_metadata, convert_sitk_image


def upsample(image, is_mask: bool = False):
    min_size = min(list(image.shape))
    if min_size > 128:
        return image
    else:
        factor = 128 / min_size
        size = [int(x * factor) for x in list(image.shape)]
        print(f"Upsampling {image.shape} to {size}")
        transform = Resize(spatial_size=size, mode="nearest" if is_mask else "trilinear")
        return transform(image[None, ...]).squeeze().numpy()

def convert_TopCoW(input_folder: str, output_dir: str, extract_full_images: bool = False):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labelsTr"), exist_ok=True)

    image_dir_name = "imagesTr"
    labels_dir_name = "bin_labelsTr"

    type_ = "whole" if extract_full_images else "roi"

    for folder in os.listdir(input_folder):
        if "val" in folder or not os.path.isdir(os.path.join(input_folder, folder)):
            continue
        print(f"Converting {folder}...")
        
        image_dir = os.path.join(input_folder, folder, image_dir_name)
        mask_dir = os.path.join(input_folder, folder, labels_dir_name)

        print(f"Converting Images...")
        for sample in os.listdir(image_dir):
            if type_ not in sample:
                print(f"Skipping {sample}...")
                continue
            print(f"Converting {sample}...")
            image = sitk.ReadImage(os.path.join(image_dir, sample))
            
            array, metadata = convert_sitk_image(image)
            print(array.shape) 
            array = upsample(array)
            print(array.shape) 
            array = array.astype(np.float32)
            metadata = metadata | calculate_metadata(array)
            sample_name = sample.replace(type_ + "_", "").replace("_0000", "").split(".")[0]

            save_array(array, os.path.join(output_dir, "imagesTr", sample_name))
            save_metadata(metadata, os.path.join(output_dir, "imagesTr", sample_name))

        print(f"Converting Masks...")
        for sample in os.listdir(mask_dir):
            if type_ not in sample:
                print(f"Skipping {sample}...")
                continue
            print(f"Converting {sample}...")
            mask = sitk.ReadImage(os.path.join(mask_dir, sample))
            array, metadata = convert_sitk_image(mask)
            array = array > 0
            array = upsample(array, is_mask=True)
            sample_name = sample.replace(type_ + "_", "").replace("_0000", "").split(".")[0]
            save_array(array, os.path.join(output_dir, "labelsTr", sample_name))
            save_metadata(metadata, os.path.join(output_dir, "labelsTr", sample_name))
