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

source scripts/setup_nnunet_env.sh
conda activate conda_envs/nnunet

export nnUNet_random_seed=702785
nnUNetv2_train Dataset001_nnunet 3d_fullres 0 --npz