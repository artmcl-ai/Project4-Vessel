# tools/preprocess_av_with_nnunet.py
import os
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from utilities.plans_handling.plans_handler import PlansManager
from paths import nnUNet_preprocessed, nnUNet_raw
from utilities.utils import get_filenames_of_train_images_and_targets
from vesselfm.preprocessing.av_preprocessor import AVPreprocessor


def main(
    dataset_name="Dataset001_nnunet",
    configuration_name="3d_fullres",
    plans_identifier="nnUNetPlans",
    num_processes=8,
    out_dir="/projectnb/ec500kb/projects/Fall_2025_Projects/Project_4_VesselFM/VesselFMAdaptationMethod/vesselFM-main/vesselFM-main/data/nnunet_npz_all",
):
    # Load plans + dataset.json
    plans_file = join(nnUNet_preprocessed, dataset_name, plans_identifier + ".json")
    plans = load_json(plans_file)
    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration(configuration_name)

    dataset_json_file = join(nnUNet_preprocessed, dataset_name, "dataset.json")
    dataset_json = load_json(dataset_json_file)

    # Get list of image/label paths from nnUNet_raw
    dataset = get_filenames_of_train_images_and_targets(
        join(nnUNet_raw, dataset_name), dataset_json
    )

    pp = AVPreprocessor(verbose=True)

    os.makedirs(out_dir, exist_ok=True)

    # Single-process (simple) version – you can parallelize like DefaultPreprocessor.run if you want
    for case_id, files in dataset.items():
        img_files = files["images"]     # list of channel files
        seg_file = files["label"]       # label .nii.gz
        output_truncated = os.path.join(out_dir, case_id)
        pp.run_case_save(
            output_truncated,
            img_files,
            seg_file,
            plans_manager,
            configuration_manager,
            dataset_json,
        )


if __name__ == "__main__":
    main()