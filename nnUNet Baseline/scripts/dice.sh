#!/bin/bash -l
#$ -P ec500kb
#$ -N nnunet_dice
#$ -l h_rt=06:00:00
#$ -pe omp 8
#$ -l mem_per_core=8G
#$ -j y
#$ -m ea

set -euo pipefail

PROJECT_ROOT="/projectnb/ec500kb/projects/Fall_2025_Projects/Project_4_VesselFM"

PRED_ROOT="${PROJECT_ROOT}/data/nnUNet_results/Dataset001_nnunet/preds"

OUTPUT_ENSEMBLE="${PRED_ROOT}/ensemble_seed1-9"

GT_DIR="${PROJECT_ROOT}/data/nnUNet_raw/Dataset001_nnunet/labelsTs"

CONDA_ENV="conda_envs/nnunet"

cd "${PROJECT_ROOT}"
echo "[INFO] Working dir: $(pwd)"
echo "[INFO] Activating conda env: ${CONDA_ENV}"
conda activate "${CONDA_ENV}"

echo "[INFO] Checking seed prediction folders..."
for s in {1..9}; do
  d="${PRED_ROOT}/seed_${s}"
  if [ ! -d "$d" ]; then
    echo "[ERROR] Missing prediction folder: $d"
    exit 1
  fi
done
echo "[OK] All seed_1..seed_9 folders found."

if [ ! -d "${GT_DIR}" ]; then
  echo "[ERROR] Ground-truth folder not found: ${GT_DIR}"
  exit 1
fi

mkdir -p "${OUTPUT_ENSEMBLE}"
echo "[INFO] Ensemble output -> ${OUTPUT_ENSEMBLE}"

echo "[INFO] Running nnUNetv2_ensemble..."
nnUNetv2_ensemble -i \
  "${PRED_ROOT}/seed_1" \
  "${PRED_ROOT}/seed_2" \
  "${PRED_ROOT}/seed_3" \
  "${PRED_ROOT}/seed_4" \
  "${PRED_ROOT}/seed_5" \
  "${PRED_ROOT}/seed_6" \
  "${PRED_ROOT}/seed_7" \
  "${PRED_ROOT}/seed_8" \
  "${PRED_ROOT}/seed_9" \
  -o "${OUTPUT_ENSEMBLE}"

echo "[OK] Ensemble done."

METRICS_JSON="${OUTPUT_ENSEMBLE}/metrics.json"
echo "[INFO] Evaluating Dice -> ${METRICS_JSON}"

nnUNetv2_evaluate_folder \
  -pred "${OUTPUT_ENSEMBLE}" \
  -gt   "${GT_DIR}" \
  -json "${METRICS_JSON}"

echo "[OK] Evaluation finished."
echo "[INFO] Results saved to: ${METRICS_JSON}"

echo "[DONE] All tasks completed."
