import SimpleITK as sitk
import os
from .utils import save_array, save_metadata, calculate_metadata, convert_sitk_image, resample_label, smooth_label


def convert_SMILE(input_folder: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labelsTr"), exist_ok=True)

    image_dir = os.path.join(input_folder, "imagesTr")  # we put TrainingSet/train and ValidationSet/validate in imagesTr
    mask_dir = os.path.join(input_folder, "labelsTr")   # we put TrainingSet/train_label and ValidationSet/validate_label in labelsTr

    print(f"Converting Images...")
    for sample in os.listdir(image_dir):
        if not sample.endswith(".nii.gz"):
            continue
        print(f"Converting {sample}...")
        image = sitk.ReadImage(os.path.join(image_dir, sample))
        # image = resample_image(image, 2)
        array, metadata = convert_sitk_image(image)
        metadata = metadata | calculate_metadata(array)
        sample_name = sample.split(".")[0]
        save_array(array, os.path.join(output_dir, "imagesTr", sample_name))
        save_metadata(metadata, os.path.join(output_dir, "imagesTr", sample_name))

    print(f"Converting Masks...")
    for sample in os.listdir(mask_dir):
        if not sample.endswith(".nii"):
            continue
        print(f"Converting {sample}...")
        mask = sitk.ReadImage(os.path.join(mask_dir, sample))
        # mask = resample_label(mask, 2)
        # mask = smooth_label(mask)
        array, metadata = convert_sitk_image(mask)
        array = array > 0
        sample_name = sample.split(".")[0]
        save_array(array, os.path.join(output_dir, "labelsTr", sample_name))
        save_metadata(metadata, os.path.join(output_dir, "labelsTr", sample_name))
