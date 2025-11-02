#Skeletonize segmentation
import SimpleITK as sitk
in_nii = "/content/case_001.nii.gz"
out_nii = "/content/skeleton_case_001.nii.gz"
img = sitk.ReadImage(in_nii)
thinner = sitk.BinaryThinningImageFilter()
skeleton = thinner.Execute(img)
sitk.WriteImage(skeleton, out_nii)
print("Saved:", out_nii)