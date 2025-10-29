#!/bin/bash -l

#$ -P ec500kb
#$ -N nnunet_train_833017
#$ -l h_rt=47:00:00
#$ -pe omp 16
#$ -l gpus=1
#$ -l gpu_type=L40S
#$ -l mem_per_core=4G
#$ -j y
#$ -m ea

cd /projectnb/ec500kb/projects/Fall_2025_Projects/Project_4_VesselFM

source scripts/setup_node_env.sh
conda activate conda_envs/nnunet

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
export TORCH_NUM_THREADS=1

export nnUNet_random_seed=833017
nnUNetv2_train Dataset001_nnunet 3d_fullres 0 --npz

rsync -a $SCRATCH_ROOT/nnUNet_results/ $PROJECT_ROOT/data/nnUNet_results/
if [[ -n "$SCRATCH_ROOT" && -d "$SCRATCH_ROOT" && "$SCRATCH_ROOT" == /scratch/$USER/* ]]; then
    echo "Cleaning up $SCRATCH_ROOT"
    rm -rf "$SCRATCH_ROOT"
else
    echo "Skip cleanup: unsafe path -> $SCRATCH_ROOT"
fi