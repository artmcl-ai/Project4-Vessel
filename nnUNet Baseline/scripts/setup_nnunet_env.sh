#!/bin/bash


PROJECT_ROOT="/projectnb/ec500kb/projects/Fall_2025_Projects/Project_4_VesselFM"

export nnUNet_raw="$PROJECT_ROOT/data/nnUNet_raw"
export nnUNet_preprocessed="$PROJECT_ROOT/data/nnUNet_preprocessed"
export nnUNet_results="$PROJECT_ROOT/data/nnUNet_results"

echo "nnU-Net environment variables have been set."
echo "nnUNet_raw=$nnUNet_raw"
echo "nnUNet_preprocessed=$nnUNet_preprocessed"
echo "nnUNet_results=$nnUNet_results"