import os
import random
import shutil
from pathlib import Path

dataset_root = Path("/projectnb/ec500kb/projects/Fall_2025_Projects/Project_4_VesselFM/data/nnUNet_raw/Dataset001_nnunet")

imagesTr_dir = dataset_root / "imagesTr"
labelsTr_dir = dataset_root / "labelsTr"
imagesTs_dir = dataset_root / "imagesTs"
labels_backup_dir = dataset_root / "test_labels_backup"

imagesTs_dir.mkdir(parents=True, exist_ok=True)
labels_backup_dir.mkdir(parents=True, exist_ok=True)

image_cases = sorted([
    f.name.replace("_0000.nii.gz", "")
    for f in imagesTr_dir.glob("*.nii.gz")
    if f.name.endswith("_0000.nii.gz")
])

num_total = len(image_cases)
num_test = max(1, int(num_total * 0.1))
test_cases = random.sample(image_cases, num_test)

print(f"total: {num_total}")
print(f"extract{num_test} to test:")
print(test_cases)

for case in test_cases:
    img_file = imagesTr_dir / f"{case}_0000.nii.gz"
    lbl_file = labelsTr_dir / f"{case}.nii.gz"
    img_target = imagesTs_dir / img_file.name
    lbl_target = labels_backup_dir / lbl_file.name

    if img_file.exists():
        shutil.move(str(img_file), str(img_target))
        print(f": {img_file.name}")
    else:
        print(f" {img_file}")

    if lbl_file.exists():
        shutil.move(str(lbl_file), str(lbl_target))
        print(f"{lbl_file.name}")
    else:
        print(f"{lbl_file}")

print("complete")
