#!/usr/bin/env python3
"""
nnUNet 10-Seed Ensemble Prediction with Non-Standard Filename Support
"""

import os
import sys
import re
import shutil
import hashlib
import argparse
import subprocess
import tempfile
from pathlib import Path


# Configuration
WORKSPACE = Path("/workspace")
NNUNET_RAW = WORKSPACE / "data/nnUNet_raw"
NNUNET_PREPROCESSED = WORKSPACE / "data/nnUNet_preprocessed"
NNUNET_RESULTS = WORKSPACE / "nnUNet_results"
DATASET_NAME = "Dataset001_nnunet"
NUM_SEEDS = 10


def setup_environment():
    """Set up nnUNet environment variables."""
    os.environ["nnUNet_raw"] = str(NNUNET_RAW)
    os.environ["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
    os.environ["nnUNet_results"] = str(NNUNET_RESULTS)


def prepare_input_files(input_path: Path, temp_dir: Path):
    """
    Prepare input files for nnUNet prediction.
    Handles non-standard filenames by renaming to nnUNet convention and creating a mapping.
    """
    input_temp = temp_dir / "input"
    input_temp.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        if not str(input_path).endswith('.nii.gz'):
            raise ValueError(f"Input must be .nii.gz format: {input_path}")
        input_files = [input_path]
    elif input_path.is_dir():
        input_files = sorted(input_path.glob("*.nii.gz"))
        if not input_files:
            raise ValueError(f"No .nii.gz files found in {input_path}")
    else:
        raise ValueError(f"Input does not exist: {input_path}")

    print(f"Processing {len(input_files)} file(s)...")

    filename_mapping = {}

    for input_file in input_files:
        original_name = input_file.name
        case_name = input_file.stem.replace('.nii', '')

        if not case_name.endswith('_0000'):
            match = re.search(r'(\d+)', case_name)
            if match:
                numeric_index = int(match.group(1))
                standardized_name = f"case_{numeric_index:04d}_0000.nii.gz"
            else:
                hash_val = int(hashlib.md5(case_name.encode()).hexdigest()[:4], 16) % 10000
                standardized_name = f"case_{hash_val:04d}_0000.nii.gz"
            print(f"  Renaming: {original_name} -> {standardized_name}")
        else:
            standardized_name = f"{case_name}.nii.gz"

        shutil.copy(input_file, input_temp / standardized_name)

        standardized_base = standardized_name.replace('_0000.nii.gz', '')
        original_base = original_name.replace('.nii.gz', '')
        filename_mapping[standardized_base] = original_base

    return input_temp, filename_mapping


def run_single_seed_prediction(seed_id: int, input_dir: Path, output_dir: Path):
    """Run nnUNetv2_predict for a single seed."""
    seed_dir = NNUNET_RESULTS / DATASET_NAME / f"seed_{seed_id}"
    checkpoint = seed_dir / "fold_0" / "checkpoint_best.pth"

    if not checkpoint.exists():
        raise RuntimeError(f"Checkpoint not found: {checkpoint}")

    # Create temporary results directory
    temp_results = Path(tempfile.mkdtemp(prefix=f"nnunet_seed{seed_id}_"))
    expected_path = temp_results / DATASET_NAME / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    expected_path.mkdir(parents=True, exist_ok=True)

    symlink_target = expected_path / "fold_0"
    actual_fold = seed_dir / "fold_0"
    symlink_target.symlink_to(actual_fold, target_is_directory=True)

    # Copy JSON files
    dataset_json_src = NNUNET_PREPROCESSED / DATASET_NAME / "dataset.json"
    plans_json_src = NNUNET_PREPROCESSED / DATASET_NAME / "nnUNetPlans.json"

    if dataset_json_src.exists():
        shutil.copy(dataset_json_src, expected_path / "dataset.json")
    if plans_json_src.exists():
        shutil.copy(plans_json_src, expected_path / "plans.json")

    env = os.environ.copy()
    env["nnUNet_results"] = str(temp_results)

    dataset_id = DATASET_NAME.split("_")[0].replace("Dataset", "").lstrip("0") or "0"

    cmd = [
        "nnUNetv2_predict",
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-d", dataset_id,
        "-c", "3d_fullres",
        "-p", "nnUNetPlans",
        "-tr", "nnUNetTrainer",
        "-f", "0",
        "--save_probabilities",
    ]

    try:
        subprocess.run(cmd, check=True, env=env)
    finally:
        if temp_results.exists():
            shutil.rmtree(temp_results)


def run_ensemble_prediction(input_dir: Path, temp_dir: Path, output_dir: Path, filename_mapping: dict):
    """Run 10-seed ensemble prediction."""
    seed_output_dirs = []
    for seed_id in range(1, NUM_SEEDS + 1):
        seed_output = temp_dir / f"seed_{seed_id}_pred"
        seed_output.mkdir(parents=True, exist_ok=True)
        seed_output_dirs.append(seed_output)

    print(f"Running {NUM_SEEDS} seed predictions...")
    for seed_id in range(1, NUM_SEEDS + 1):
        run_single_seed_prediction(seed_id, input_dir, seed_output_dirs[seed_id - 1])

    print("Running ensemble...")
    ensemble_output = temp_dir / "ensemble"
    ensemble_output.mkdir(parents=True, exist_ok=True)

    ensemble_cmd = ["nnUNetv2_ensemble", "-i"]
    ensemble_cmd.extend([str(d) for d in seed_output_dirs])
    ensemble_cmd.extend(["-o", str(ensemble_output)])

    subprocess.run(ensemble_cmd, check=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    for result_file in ensemble_output.glob("*.nii.gz"):
        standardized_name = result_file.stem.replace('.nii', '')

        if standardized_name in filename_mapping:
            original_name = filename_mapping[standardized_name]
            output_name = f"{original_name}.nii.gz"
            print(f"  Mapping result: {result_file.name} -> {output_name}")
        else:
            output_name = result_file.name

        shutil.copy(result_file, output_dir / output_name)

    # Copy metric.json if it exists
    metric_json = ensemble_output / "metric.json"
    if metric_json.exists():
        shutil.copy(metric_json, output_dir / "metric.json")
        print(f"  Copied metric.json to output directory")


def main():
    parser = argparse.ArgumentParser(description="nnUNet 10-Seed Ensemble Prediction")
    parser.add_argument("-i", "--input", required=True, type=Path, help="Input file or directory")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Output directory")
    parser.add_argument("--temp-dir", type=str, default=None,
                        help="Temporary directory for intermediate files (default: /tmp)")

    args = parser.parse_args()

    setup_environment()

    if not args.input.exists():
        raise ValueError(f"Input does not exist: {args.input}")

    if args.temp_dir is None:
        temp_base = Path("/tmp")
    else:
        temp_base = Path(args.temp_dir)
        temp_base.mkdir(parents=True, exist_ok=True)

    print(f"Using temporary directory: {temp_base}")

    with tempfile.TemporaryDirectory(dir=str(temp_base)) as temp_dir:
        temp_path = Path(temp_dir)
        input_temp, filename_mapping = prepare_input_files(args.input, temp_path)
        run_ensemble_prediction(input_temp, temp_path, args.output, filename_mapping)

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
