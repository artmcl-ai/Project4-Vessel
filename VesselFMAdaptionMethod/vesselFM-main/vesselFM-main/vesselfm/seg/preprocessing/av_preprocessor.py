# vesselfm/nnunet_preprocessing/av_preprocessor.py
import os
import numpy as np
from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.paths import nnUNet_preprocessed
from .default_preprocessor import DefaultPreprocessor


class AVPreprocessor(DefaultPreprocessor):
    """
    Uses nnUNetv2 DefaultPreprocessor but saves cases as .npz so VesselFM
    can just load them as arrays.
    """

    def run_case_save(
        self,
        output_filename_truncated: str,
        image_files,
        seg_file,
        plans_manager,
        configuration_manager,
        dataset_json,
    ):
        # data: (C, X, Y, Z), seg: (1, X, Y, Z)
        data, seg, properties = self.run_case(
            image_files, seg_file, plans_manager, configuration_manager, dataset_json
        )

        out_dir = os.path.dirname(output_filename_truncated)
        os.makedirs(out_dir, exist_ok=True)

        # Example filename: out_dir/av_0000.npz
        case_id = Path(seg_file).stem if seg_file is not None else Path(image_files[0]).stem.split("_0000")[0]
        out_file = os.path.join(out_dir, f"{case_id}.npz")

        np.savez_compressed(
            out_file,
            image=data.astype(np.float32),
            label=seg.astype(np.int16),
            properties=properties,
        )
