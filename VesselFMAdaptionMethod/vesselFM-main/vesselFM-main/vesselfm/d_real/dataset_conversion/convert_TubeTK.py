import SimpleITK as sitk
import os
from .utils import save_array, save_metadata, calculate_metadata, convert_sitk_image, resample_sample, smooth_label


def convert_TubeTK(input_folder: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labelsTr"), exist_ok=True)

    for sample in os.listdir(input_folder):
        if "Normal" not in sample:
            continue
        sample_path = os.path.join(input_folder, sample)
        if "AuxillaryData" not in os.listdir(sample_path):
            continue
        
        print(f"Converting {sample}...")
        sample_num = sample.split("-")[1]

        image = sitk.ReadImage(os.path.join(sample_path, f"MRA/{sample.replace('-', '')}-MRA.mha"))
        mask = sitk.ReadImage(os.path.join(sample_path, f"AuxillaryData/VascularNetwork.mha"))

        image, mask = resample_sample(image, mask, 2)
        mask = smooth_label(mask)
        array, metadata = convert_sitk_image(image)
        print(f"Array type: {array.dtype}")
        metadata = metadata | calculate_metadata(array)
        save_array(array, os.path.join(output_dir, "imagesTr", sample_num))
        save_metadata(metadata, os.path.join(output_dir, "imagesTr", sample_num))

        array, metadata = convert_sitk_image(mask)
        array = array > 0
        save_array(array, os.path.join(output_dir, "labelsTr", sample_num))
        save_metadata(metadata, os.path.join(output_dir, "labelsTr", sample_num))
