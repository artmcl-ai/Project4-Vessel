#!/bin/bash -l
#$ -P ec500kb
#$ -N nnunet_docker_pred_10seed
#$ -l h_rt=08:00:00
#$ -pe omp 14
#$ -l gpus=1
#$ -l gpu_type=L40S
#$ -l mem_per_core=4G
#$ -j y
#$ -m ea

PROJECT_ROOT="/projectnb/ec500kb/projects/Fall_2025_Projects/Project_4_VesselFM"
SIF="${PROJECT_ROOT}/result/nnunet_vesselfm.sif"

INPUT="/projectnb/ec500kb/projects/Fall_2025_Projects/Project_4_VesselFM/data/nnUNet_raw/Dataset001_nnunet/imagesTs"
OUTPUT="${PROJECT_ROOT}/test/preds_dir"

SCRATCH_DIR="/scratch/nnunet_pred_${JOB_ID}"
mkdir -p "${SCRATCH_DIR}"
mkdir -p "${OUTPUT}"

export TMPDIR="${SCRATCH_DIR}"

singularity run --nv \
  -B /projectnb:/projectnb \
  -B "${SCRATCH_DIR}":/tmp \
  "${SIF}" \
  -i "${INPUT}" \
  -o "${OUTPUT}"

rm -rf "${SCRATCH_DIR}"
echo "Done. Scratch cleaned: ${SCRATCH_DIR}"
