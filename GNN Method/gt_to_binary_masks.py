# -*- coding: utf-8 -*-

#CONVERT GROUND TRUTH INTO BINARY MASKS
import os

in_path = "/projectnb/ec500kb/projects/Project_4/Graph_Creations/Masks_Truth"

for filename in os.listdir(in_path):
    out_path = "/projectnb/ec500kb/projects/Project_4/Graph_Creations/Binary_Masks_Truth/"
    full_path = os.path.join(in_path, filename)
    name = os.path.basename(filename).replace(".nii.gz", "")
    number = name[-3:]
    labels_to_merge = [2,3]
    target_label = 1
    img = sitk.ReadImage(full_path)
    arr = sitk.GetArrayFromImage(img)
    mask = np.isin(arr, labels_to_merge)
    arr[mask] = target_label
    out_img = sitk.GetImageFromArray(arr)
    out_img.CopyInformation(img)
    out_path = out_path + 'binary_case_' + number + '.nii.gz'
    print(out_path)
    sitk.WriteImage(out_img, out_path, useCompression=True)
