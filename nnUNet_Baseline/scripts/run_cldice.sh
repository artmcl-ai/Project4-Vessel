#!/bin/bash -l
#$ -P ec500kb
#$ -N cldice_eval
#$ -l h_rt=06:00:00
#$ -pe omp 8
#$ -l mem_per_core=4G
#$ -j y
#$ -m ea

set -euo pipefail

PROJECT_ROOT="/projectnb/ec500kb/projects/Fall_2025_Projects/Project_4_VesselFM"
SCRIPT_PATH="${PROJECT_ROOT}/tools/cldice.py"
CONDA_ENV="conda_envs/nnunet"

echo "[INFO] Starting clDice evaluation job"
echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Python script: ${SCRIPT_PATH}"

cd "${PROJECT_ROOT}"
conda activate "${CONDA_ENV}"

echo "[INFO] Running clDice evaluation..."
python "${SCRIPT_PATH}"

echo "[DONE] clDice evaluation completed successfully."
