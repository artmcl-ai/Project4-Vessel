#!/bin/bash
export PROJECT_ROOT="/projectnb/ec500kb/projects/Fall_2025_Projects/Project_4_VesselFM"
export SCRATCH_ROOT="/scratch/$USER/nnunet_run_${JOB_ID}"

mkdir -p "$SCRATCH_ROOT"
df -h /scratch | sed -n '1,2p'

rsync -a "$PROJECT_ROOT/data/nnUNet_preprocessed/" "$SCRATCH_ROOT/nnUNet_preprocessed/"
mkdir -p "$SCRATCH_ROOT/nnUNet_results"

export nnUNet_raw="${SCRATCH_ROOT}/nnUNet_raw"
export nnUNet_preprocessed="${SCRATCH_ROOT}/nnUNet_preprocessed"
export nnUNet_results="${SCRATCH_ROOT}/nnUNet_results"

echo "Running on node-local scratch: $SCRATCH_ROOT"
echo "nnUNet_raw=$nnUNet_raw"
echo "nnUNet_preprocessed=$nnUNet_preprocessed"
echo "nnUNet_results=$nnUNet_results"

