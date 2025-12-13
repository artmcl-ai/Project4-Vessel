# nnU-Net + VesselFM Probability-Map (3D Full-Res) — Lung Vessel Segmentation

## Overview
This repo implements a **dual-channel nnU-Net** approach that augments the CT input with a **VesselFM-generated probability map**.

Pipeline:
1. **Input:** CT volume (`.nii.gz`)
2. **VesselFM preprocessing:** generate a vessel probability map and build **2-channel** input
   - Channel 0: CT (`*_0000.nii.gz`)
   - Channel 1: VesselFM probability map (`*_0001.nii.gz`)
3. **Inference:** nnU-Net 3D full-res predicts voxel-wise labels (fold 0)
4. **Ensemble + output:** 5-seed ensemble; export final `.nii.gz` masks

## Structure
- `tools/` — Utilities for probability-map generation and spacing fixes (shared with the baseline).
- `data/` — Minimal metadata only (e.g., `dataset.json`). Full dataset is not included due to size.
- `scripts/` — SCC submission scripts.
- `docker/` — Docker assets (including `docker/process.py` as the inference entrypoint).

## Data Format (nnU-Net)
- Raw CT input (single channel): `case_xxxx_0000.nii.gz`
- During inference, `process.py` creates a temporary dual-channel folder containing:
  - `case_xxxx_0000.nii.gz` (CT)
  - `case_xxxx_0001.nii.gz` (VesselFM probability map)

## Expected Model Layout (inside container)
- VesselFM model:
  - `/workspace/vesselfm_finetuned_monai.pt`
- nnU-Net checkpoints (fold 0, best checkpoint):
  - `/workspace/nnUNet_results/Dataset003_vesselfm_good/seed_{1..5}/fold_0/checkpoint_best.pth`

## Required Environment Variables
Set nnU-Net paths:
- `nnUNet_raw`
- `nnUNet_preprocessed`
- `nnUNet_results`

## Inputs / Outputs
- **Input:** CT images (`.nii.gz`)
- **Output:** predicted masks (`.nii.gz`, same geometry as input)

## Docker Workflow
This repo provides a containerized inference entrypoint (`docker/process.py`) for prediction.

### Interface
- `-i / --input`: a single `.nii.gz` file **or** a directory containing `.nii.gz` files
- `-o / --output`: output directory for final ensemble masks

### Run with Docker (local GPU)
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
One used example in nnUNet Baseline/scripts/docker_test_3.sh

## Note

You can find images from https://hub.docker.com/r/mzou2000/nnunet-5seed-predict

Also, you can find sif images in /projectnb/ec500kb/projects/Fall_2025_Projects/Project_4_VesselFM/result/nnunet_vesselfm.sif
