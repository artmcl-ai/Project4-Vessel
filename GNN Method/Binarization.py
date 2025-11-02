#Make the multi-class segmentation into binary
import SimpleITK as sitk
import numpy as np
in_path = "/content/case_001.nii.gz"
out_path = "/content/binary_case_001.nii.gz"
labels_to_merge = [2,3]
target_label = 1
img = sitk.ReadImage(in_path)
arr = sitk.GetArrayFromImage(img)
mask = np.isin(arr, labels_to_merge)
arr[mask] = target_label
out_img = sitk.GetImageFromArray(arr)
out_img.CopyInformation(img)
sitk.WriteImage(out_img, out_path, useCompression=True)
print("Saved", out_path)