#!/usr/bin/env python3
"""
nnUNet 5-Seed Ensemble Prediction with VesselFM
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
DATASET_NAME = "Dataset003_vesselfm_good"
NUM_SEEDS = 5
VESSELFM_MODEL_PATH = WORKSPACE / "vesselfm_finetuned_monai.pt"


def setup_environment():
    """Set up nnUNet environment variables."""
    os.environ["nnUNet_raw"] = str(NNUNET_RAW)
    os.environ["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
    os.environ["nnUNet_results"] = str(NNUNET_RESULTS)


def standardize_filename(original_name: str) -> tuple[str, str]:
    """Convert non-standard filename to nnUNet format."""
    case_name = original_name.replace('.nii.gz', '').replace('.nii', '')

    if case_name.endswith('_0000'):
        base_name = case_name.replace('_0000', '')
        return original_name, base_name

    match = re.search(r'(\d+)', case_name)
    if match:
        numeric_index = int(match.group(1))
        base_name = f"case_{numeric_index:04d}"
        standardized_name = f"{base_name}_0000.nii.gz"
    else:
        hash_val = int(hashlib.md5(case_name.encode()).hexdigest()[:4], 16) % 10000
        base_name = f"case_{hash_val:04d}"
        standardized_name = f"{base_name}_0000.nii.gz"

    return standardized_name, base_name


def prepare_dual_channel_input(input_path: Path, temp_dir: Path):
    """Prepare dual-channel input (CT + VesselFM) with filename standardization."""
    input_standardized = temp_dir / "input_standardized"
    input_standardized.mkdir(parents=True, exist_ok=True)

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
        standardized_name, base_name = standardize_filename(original_name)

        if original_name != standardized_name:
            print(f"  Renaming: {original_name} -> {standardized_name}")

        shutil.copy(input_file, input_standardized / standardized_name)
        original_base = original_name.replace('.nii.gz', '').replace('.nii', '')
        filename_mapping[base_name] = original_base

    input_temp = temp_dir / "input_dual_channel"
    input_temp.mkdir(parents=True, exist_ok=True)

    probability_maker_script = WORKSPACE.parent / "nnunet_vesselfm" / "tools" / "probablity_maker.py"
    if not probability_maker_script.exists():
        probability_maker_script = Path(__file__).parent / "tools" / "probablity_maker.py"
    if not probability_maker_script.exists():
        raise FileNotFoundError(f"Could not find probablity_maker.py")

    print("Running VesselFM preprocessing...")
    subprocess.run([
        sys.executable,
        str(probability_maker_script),
        "-i", str(input_standardized),
        "-o", str(input_temp),
        "-m", str(VESSELFM_MODEL_PATH),
    ], check=True)

    print("VesselFM preprocessing complete.\n")
    return input_temp, filename_mapping


def run_single_seed_prediction(seed_id: int, input_dir: Path, output_dir: Path):
    """Run nnUNetv2_predict for a single seed."""
    seed_dir = NNUNET_RESULTS / DATASET_NAME / f"seed_{seed_id}"
    checkpoint = seed_dir / "fold_0" / "checkpoint_best.pth"

    if not checkpoint.exists():
        raise RuntimeError(f"Checkpoint not found: {checkpoint}")

    temp_results = Path(tempfile.mkdtemp(prefix=f"nnunet_seed{seed_id}_"))
    expected_path = temp_results / DATASET_NAME / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    expected_path.mkdir(parents=True, exist_ok=True)

    symlink_target = expected_path / "fold_0"
    actual_fold = seed_dir / "fold_0"
    symlink_target.symlink_to(actual_fold, target_is_directory=True)

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
    """Run 5-seed ensemble prediction with filename mapping."""
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

        for suffix in ['_0000', '_0001']:
            if standardized_name.endswith(suffix):
                standardized_name = standardized_name.replace(suffix, '')

        if standardized_name in filename_mapping:
            original_base = filename_mapping[standardized_name]
            output_name = f"{original_base}.nii.gz"
            print(f"  Restoring filename: {result_file.name} -> {output_name}")
        else:
            output_name = result_file.name
            for suffix in ['_0000', '_0001']:
                output_name = output_name.replace(suffix, '')

        shutil.copy(result_file, output_dir / output_name)


def main():
    parser = argparse.ArgumentParser(description="nnUNet 5-Seed Ensemble with VesselFM")
    parser.add_argument("-i", "--input", required=True, type=Path, help="Input file or directory")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Output directory")
    parser.add_argument("--temp-dir", type=str, nargs='?', const='auto', default=None,
                        help="Temporary directory (default: /tmp, 'auto': /scratch/nnunet_pred)")

    args = parser.parse_args()
    setup_environment()

    if not args.input.exists():
        raise ValueError(f"Input does not exist: {args.input}")

    if args.temp_dir is None:
        temp_base = Path("/tmp")
    elif args.temp_dir == 'auto':
        temp_base = Path("/scratch/nnunet_pred")
        temp_base.mkdir(parents=True, exist_ok=True)
    else:
        temp_base = Path(args.temp_dir)
        temp_base.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=str(temp_base)) as temp_dir:
        temp_path = Path(temp_dir)
        input_dual_channel, filename_mapping = prepare_dual_channel_input(args.input, temp_path)
        run_ensemble_prediction(input_dual_channel, temp_path, args.output, filename_mapping)

    print(f"\n✓ Results saved to: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
