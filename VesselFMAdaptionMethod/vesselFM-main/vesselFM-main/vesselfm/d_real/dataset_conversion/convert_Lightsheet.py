import os
import SimpleITK as sitk
from .utils import (
    save_array,
    save_metadata,
    convert_sitk_image,
)


def convert_Lightsheet(folder: str, output_folder: str):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "labelsTr"), exist_ok=True)

    image = "imagesTr/lightsheet.nii"   # crop2_down2x.nii
    mask = "labelsTr/lightsheet.nii"    # crop2_down2x_seg_dilated_smoothed.nii
    image = sitk.ReadImage(os.path.join(folder, image))
    mask = sitk.ReadImage(os.path.join(folder, mask))
    print(image.GetSize())
    image, metadata = convert_sitk_image(image)

    save_array(image, os.path.join(output_folder, "imagesTr", "0"))
    save_metadata(metadata, os.path.join(output_folder, "imagesTr", "0"))

    mask, metadata = convert_sitk_image(mask)
    save_array(mask, os.path.join(output_folder, "labelsTr", "0"))
    save_metadata(metadata, os.path.join(output_folder, "labelsTr", "0"))
