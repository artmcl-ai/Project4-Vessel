import os
import nibabel as nib
import numpy as np

label_dir = "data/nnUNet_raw/Dataset001_nnunet/labelsTr"

for fname in os.listdir(label_dir):
    if fname.endswith(".nii.gz"):
        fpath = os.path.join(label_dir, fname)
        img = nib.load(fpath)
        data = img.get_fdata()

        data_int = data.astype(np.uint8)

        new_img = nib.Nifti1Image(data_int, img.affine, img.header)
        nib.save(new_img, fpath)

        print(f"Converted {fname} to integer labels")
