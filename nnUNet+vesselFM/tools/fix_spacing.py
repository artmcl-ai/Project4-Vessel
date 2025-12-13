import os
from pathlib import Path
import nibabel as nib

ROOT = Path("/projectnb/ec500kb/projects/Fall_2025_Projects/Project_4_VesselFM/data/nnUNet_raw/Dataset003_vesselfm_good/imagesTs")

def get_spacing(img):
    return tuple([round(z, 5) for z in img.header.get_zooms()[:3]])

def fix_case(ct_path):
    prob_path = Path(str(ct_path).replace("_0000", "_0001"))
    if not prob_path.exists():
        print(f"SKIP, prob file not found: {prob_path.name}")
        return
    
    ct_img = nib.load(str(ct_path))
    ct_data = ct_img.get_fdata()
    ct_affine = ct_img.affine
    ct_header = ct_img.header.copy()
    
    prob_img = nib.load(str(prob_path))
    prob_data = prob_img.get_fdata()

    if ct_data.shape != prob_data.shape:
        print(f"SHAPE MISMATCH: {ct_path.name} vs {prob_path.name} {ct_data.shape} != {prob_data.shape}")
        return

    fixed_img = nib.Nifti1Image(prob_data.astype("float32"), ct_affine, ct_header)

    nib.save(fixed_img, str(prob_path))

    print(f"FIXED {prob_path.name} | spacing now = {get_spacing(fixed_img)}")


def main():
    print("=== Fixing spacing for vesselFM prob maps ===")
    ct_files = sorted(ROOT.glob("*_0000.nii.gz"))

    for ct_path in ct_files:
        fix_case(ct_path)

    print("\n=== DONE. Please re-run nnUNet preprocess. ===")


if __name__ == "__main__":
    main()
