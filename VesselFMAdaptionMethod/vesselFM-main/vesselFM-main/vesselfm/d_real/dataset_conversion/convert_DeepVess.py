import numpy as np
import os
from libtiff import TIFFfile
from .utils import (
    save_array,
    save_metadata,
    calculate_metadata,
)


def convert_DeepVess(folder: str, output_folder: str):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "labelsTr"), exist_ok=True)

    image = "HaftJavaherian_DeepVess2018_GroundTruthImage.tif"
    mask = "HaftJavaherian_DeepVess2018_GroundTruthLabel.tif"

    image_arr = np.array(TIFFfile(os.path.join(folder, image)).get_tiff_array())
    image_arr = np.squeeze(image_arr)
    print(f"Image shape: {image_arr.shape}")

    metadata = calculate_metadata(image_arr)
    save_array(
        image_arr,
        os.path.join(
            output_folder, "imagesTr", "HaftJavaherian_DeepVess2018_GroundTruthImage"
        ),
    )
    save_metadata(
        metadata,
        os.path.join(
            output_folder, "imagesTr", "HaftJavaherian_DeepVess2018_GroundTruthImage"
        ),
    )

    mask_arr = np.array(TIFFfile(os.path.join(folder, mask)).get_tiff_array())
    mask_arr = mask_arr > 0
    print(f"Mask shape: {mask_arr.shape}")
    mask_arr = np.squeeze(mask_arr)
    mask_arr = mask_arr > 0

    metadata = calculate_metadata(mask_arr)
    save_array(
        mask_arr,
        os.path.join(output_folder, "labelsTr", "HaftJavaherian_DeepVess2018_GroundTruthLabel")
    )
    save_metadata(
        metadata,
        os.path.join(output_folder, "labelsTr", "HaftJavaherian_DeepVess2018_GroundTruthLabel")
    )
