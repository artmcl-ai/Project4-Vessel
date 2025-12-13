# nnU-Net Baseline (3D Full-Res) — Lung Vessel Segmentation

## Overview
This repo provides a **VesselFM-independent baseline** for multi-class lung vessel segmentation using **3D nnU-Net (full-resolution)**.

Pipeline:
1. **Input:** CT volume (`.nii.gz`)
2. **nnU-Net preprocessing:** auto plans + resampling
3. **Inference:** 3D nnU-Net predicts voxel-wise labels
4. **Output:** **3-class mask** (`0=background, 1=artery, 2=vein`)
5. **Ensemble + output:** multi-seed ensemble

## Structure
- `tools/` — Utilities for preprocessing/postprocessing (renaming to nnU-Net format, shape/spacing fixes, Dice/clDice evaluation, etc.)
- `data/` — Minimal metadata only (e.g., dataset.json). Full dataset is not included due to size.
- `configs/` - Randomly generate 10 seeds.
- `scripts/` - SCC submission scripts for preprocessing, training, inference, and evaluation.
- `docker/` - Docker assets.

## Data Format (nnU-Net)
- Images: `case_xxxx_0000.nii.gz`
- Labels: `case_xxxx.nii.gz` with class IDs above

## Required Environment Variables
Set nnU-Net paths(Already set in docker):
- `nnUNet_raw`
- `nnUNet_preprocessed`
- `nnUNet_results`

## Input
- CT images: `.nii.gz`

## Outputs
- Predicted masks: `.nii.gz` (same geometry as input)
- Logs/checkpoints: under `nnUNet_results/`

## Docker Workflow

this repo provides a containerized inference entrypoint (`process.py`) for prediction.

### Interface

The container exposes a simple CLI:

- `-i / --input`: a single `.nii.gz` file **or** a directory containing `.nii.gz` files
- `-o / --output`: output directory for final ensemble masks

### Option A: Run with Docker (local GPU)
Example (single case):

```bash
docker run --rm --gpus all \
  -v /path/to/case_005_0000.nii.gz:/input/case_005_0000.nii.gz:ro \
  -v /path/to/output_dir:/output \
  <IMAGE_NAME> \
  -i /input/case_005_0000.nii.gz \
  -o /output
```

### (SCC Note: avoid filling your home directory)
If you run the container on SCC via Singularity/Apptainer, **bind a scratch directory to `/tmp`**.
Otherwise, intermediate files may go to the default temp location and can easily exceed your home quota.

Example:
```bash
SCRATCH_DIR="/scratch/nnunet_pred_${JOB_ID}"
mkdir -p "${SCRATCH_DIR}"
export TMPDIR="${SCRATCH_DIR}"
singularity run --nv \
  -B /projectnb:/projectnb \
  -B "${SCRATCH_DIR}":/tmp \
  <IMAGE>.sif \
  -i <INPUT_NII_OR_DIR> \
  -o <OUTPUT_DIR>
```
One used example in nnUNet Baseline/scripts/docker_test_1.sh

## Note

You can find images from https://hub.docker.com/r/mzou2000/nnunet-10seed-predict

Also, you can find sif images in /projectnb/ec500kb/projects/Fall_2025_Projects/Project_4_VesselFM/result/nnunet_baseline.sif
