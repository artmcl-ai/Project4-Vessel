import numpy as np
import SimpleITK as sitk
import os
from .utils import save_array, save_metadata, calculate_metadata


def convert_HiP_CT(input_folder, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labelsTr"), exist_ok=True)
    
    input_folder = os.path.join(input_folder, "train")  # we solely processed kidney_1_dense, kidney_2, kidney_3_sparse
    print(f"converting {input_folder}")
    for sample in os.listdir(input_folder):
        sample_path = os.path.join(input_folder, sample)

        if len(os.listdir(sample_path)) != 2:
            continue
        print(f"Convert {sample}...")

        img = []
        gt = []
        img_metadata = []
        gt_metadata = []
        slices = os.listdir(os.path.join(sample_path, "images"))
        slices.sort()
        for slice in slices:
            img.append(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(os.path.join(sample_path, "images", slice)))))
            gt.append(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(os.path.join(sample_path, "labels", slice)))) > 0)

        img = np.stack(img, axis=0)
        gt = np.stack(gt, axis=0)
        img_metadata = calculate_metadata(img)
        gt_metadata = calculate_metadata(gt)

        assert img.shape == gt.shape, f"shape assertion failed {img.shape},  {gt.shape}"
        save_array(img.astype(np.float32), os.path.join(output_dir, "imagesTr", sample.split(".")[0]))
        save_array(gt.astype(np.uint8), os.path.join(output_dir, "labelsTr", sample.split(".")[0]))
        save_metadata(img_metadata, os.path.join(output_dir, "imagesTr", sample.split(".")[0]))
        save_metadata(gt_metadata, os.path.join(output_dir, "labelsTr", sample.split(".")[0]))
