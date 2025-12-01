# tools/preprocess_av_with_nnunet.py

import multiprocessing as mp
import pathlib
import re
import os
import sys

# Add repo root (one level up from tools/) to sys.path so 'vesselfm' is importable
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from vesselfm.seg.preprocessing.av_preprocessor import AVPreprocessor

# Paths relative to repo root
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"

NNUNET_RAW = DATA_ROOT / "nnUNet_raw"
NNUNET_PREPROCESSED = DATA_ROOT / "nnUNet_preprocessed"
NNUNET_RESULTS = DATA_ROOT / "nnUNet_results"
NNUNET_RESULTS.mkdir(parents=True, exist_ok=True)

def build_dataset_from_image_label_pattern(images_dir, labels_dir):
    """
    Build a nnUNet-style dataset dict from files named:
      images: image_###.nii.gz
      labels: label_###.nii.gz
    """
    images_dir = pathlib.Path(images_dir)
    labels_dir = pathlib.Path(labels_dir)
    dataset = {}

    for img_path in sorted(images_dir.glob("image_*.nii*")):
        img_name = img_path.name  # e.g. image_001.nii.gz
        stem = img_name.split(".")[0]  # image_001
        idx = stem.replace("image_", "")  # '001'

        lab_name = f"label_{idx}.nii.gz"
        lab_path = labels_dir / lab_name
        if not lab_path.exists():
            print(f"WARNING: missing label for {img_name}, expected {lab_path}")
            continue

        # case_id is arbitrary; just needs to be unique & consistent
        case_id = f"case_{idx}"

        dataset[case_id] = {
            "images": [str(img_path)],   # list of channels (1 channel here)
            "label": str(lab_path),
        }

    if not dataset:
        raise RuntimeError(
            f"No image/label pairs found in {images_dir} and {labels_dir}. "
            f"Check that filenames follow image_### / label_### pattern."
        )

    return dataset


def process_one_case(args):
    """
    Worker for multiprocessing. Each process gets its own AVPreprocessor and
    PlansManager so there are no issues.
    """
    (case_id,
     files,
     out_dir,
     plans,              # dict from nnUNetPlans.json
     configuration_name, # e.g. "3d_fullres"
     dataset_json) = args

    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from vesselfm.seg.preprocessing.av_preprocessor import AVPreprocessor

    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration(configuration_name)

    pp = AVPreprocessor(verbose=False)

    img_files = files["images"]   # list of image paths (1 channel)
    seg_file = files["label"]     # label path

    os.makedirs(out_dir, exist_ok=True)
    output_truncated = os.path.join(out_dir, case_id)

    pp.run_case_save(
        output_truncated,
        img_files,
        seg_file,
        plans_manager,
        configuration_manager,
        dataset_json,
    )


def main(
    dataset_name="Dataset001_nnunet",
    configuration_name="3d_fullres",
    plans_identifier="nnUNetPlans",
    num_processes=8,
    out_dir=str(DATA_ROOT / "nnunet_npz_all"),
):
    # Load plans + dataset.json
    plans_file = join(str(NNUNET_PREPROCESSED), dataset_name, plans_identifier + ".json")
    plans = load_json(plans_file)

    dataset_json_file = join(str(NNUNET_PREPROCESSED), dataset_name, "dataset.json")
    dataset_json = load_json(dataset_json_file)

    # Build dataset dict using your image_### / label_### pattern
    images_dir = join(str(NNUNET_RAW), dataset_name, "imagesTr")
    labels_dir = join(str(NNUNET_RAW), dataset_name, "labelsTr")
    dataset = build_dataset_from_image_label_pattern(images_dir, labels_dir)

    # No longer need a single shared AVPreprocessor or PlansManager here
    os.makedirs(out_dir, exist_ok=True)

    # Build argument list for multiprocessing
    args_list = []
    for case_id, files in dataset.items():
        args_list.append(
            (
                case_id,
                files,
                out_dir,
                plans,              # dict, picklable
                configuration_name, # string, picklable
                dataset_json,       # dict, picklable
            )
        )

    # Use a pool of workers
    with mp.Pool(processes=num_processes) as pool:
        list(pool.map(process_one_case, args_list))



if __name__ == "__main__":
    main()