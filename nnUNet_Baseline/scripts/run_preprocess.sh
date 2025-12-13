#!/bin/bash -l
#$ -P ec500kb
#$ -N nnunet_preprocess
#$ -l h_rt=06:00:00
#$ -pe omp 8
#$ -l mem_per_core=8G
#$ -j y
#$ -m ea

cd /projectnb/ec500kb/projects/Fall_2025_Projects/Project_4_VesselFM
source scripts/setup_nnunet_env.sh
conda activate conda_envs/nnunet

export OMP_NUM_THREADS=$NSLOTS
export MKL_NUM_THREADS=$NSLOTS
export OPENBLAS_NUM_THREADS=$NSLOTS
export NUMEXPR_NUM_THREADS=$NSLOTS
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$NSLOTS
export TORCH_NUM_THREADS=$NSLOTS

nnUNetv2_plan_and_preprocess -d 1 -c 3d_fullres --verify_dataset_integrity -np 8