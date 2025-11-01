import os
import SimpleITK as sitk
from .utils import (
    save_array,
    save_metadata,
    calculate_metadata,
    convert_sitk_image,
)


def convert_VesSAP_anno(folder, out_dir):
    os.makedirs(out_dir + "_C00", exist_ok=True)
    os.makedirs(os.path.join(out_dir + "_C00", "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(out_dir + "_C00", "labelsTr"), exist_ok=True)
    os.makedirs(out_dir + "_C01", exist_ok=True)
    os.makedirs(os.path.join(out_dir + "_C01", "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(out_dir + "_C01", "labelsTr"), exist_ok=True)

    image = "prep"
    image_dir = os.path.join(folder, image) # contains prep_<experiment_name>_cube.nii.gz
    mask = "gt_all"
    mask_dir = os.path.join(folder, mask)   # contains <experiment_name>_cube_DoG026.nii.gz & <experiment_name>_cube_DoG026_classResult.nii.gz
    
    image_set = set()
    skiped = []
    for file in os.listdir(image_dir):
        channel = "_C00"
        if "_C01_" in file:
            channel = "_C01"

        image = sitk.ReadImage(os.path.join(image_dir, file))
        array, metadata = convert_sitk_image(image)
        metadata = metadata | calculate_metadata(array)
        sample_name = file.split(".")[0].replace("prep_", "")
        save_array(array, os.path.join(out_dir + channel, "imagesTr", sample_name))
        save_metadata(metadata, os.path.join(out_dir + channel, "imagesTr", sample_name))
        
        print(f"Converting {sample_name}...")
        image_set.add(sample_name)

    mask_set = set()
    skiped_mask = []
    for file in os.listdir(mask_dir):
        mask = sitk.ReadImage(os.path.join(mask_dir, file))
        array, metadata = convert_sitk_image(mask)
        array = array > 0

        if "Result" in file:
            skiped_mask.append(file)
            continue
        else:
            sample_name = "_".join(file.split(".")[0].split("_")[:-1])

        save_array(array, os.path.join(out_dir + "_C00", "labelsTr", sample_name))
        save_array(array, os.path.join(out_dir + "_C01", "labelsTr", sample_name.replace("C00", "C01")))
        save_metadata(metadata, os.path.join(out_dir + "_C00", "labelsTr", sample_name))
        save_metadata(metadata, os.path.join(out_dir + "_C01", "labelsTr", sample_name.replace("C00", "C01")))

        print(f"Converting {sample_name}...")
        mask_set.add(sample_name)

    print(len(skiped), len(skiped_mask))
    print(mask_set - image_set)
    print(image_set - mask_set)
