import SimpleITK as sitk
import os
from .utils import save_array, save_metadata, calculate_metadata, convert_sitk_image


def convert_OCTA(input_folder: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labelsTr"), exist_ok=True)

    for file in os.listdir(input_folder):
        print(f"Converting {file}...")
        file_name = file.split(".")[0]

        if 'label' in file_name:
            file_name = file_name[:-6]
            label = sitk.ReadImage(os.path.join(input_folder, file))
            label, sitk_metadata = convert_sitk_image(label)
            label = label > 0
            metadata = sitk_metadata | calculate_metadata(label)
            save_array(label, os.path.join(output_dir, "labelsTr", file_name))
            save_metadata(metadata, os.path.join(output_dir, "labelsTr", file_name))
        else:
            image = sitk.ReadImage(os.path.join(input_folder, file))
            image, sitk_metadata = convert_sitk_image(image)
            metadata = sitk_metadata | calculate_metadata(image)
            save_array(image, os.path.join(output_dir, "imagesTr", file_name))
            save_metadata(metadata, os.path.join(output_dir, "imagesTr", file_name))