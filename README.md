# Project Overview

This repository focuses on developing a model capable of accurately segmenting and classifying veins and arteries from non-contrast CT images. To support this goal, we leverage **VesselFM** — a pretrained model designed for vessel segmentation across diverse imaging domains. You can find VesselFM on GitHub: https://github.com/bwittmann/vesselFM

## Repository Structure

The project is divided into four primary folders each representing a different appraoch:


- **UNet Approach (Baseline)**  
  A convolutional neural network architecture optimized for CT image segmentation.
  
- **VesselFM + GNN Approach**  
  A graph neural network designed to leverage vessel topology and connectivity.

- **VesselFM 3-Class Multihead Adaptation Approach**  
  A customized version of VesselFM with significant architectural alterations to enable multiclass output.

- **VesselFM + nnUNET approach**  
  A customized version of VesselFM with significant architectural alterations to enable multiclass output.

Each folder contains the code required for model development, training, and evaluation; however, the data and models themselves are too large, so they are kept in BU SCC.

- **Metrics_Notebooks**  
  Contains Jupyter notebooks used to generate metrics, tables, figures, etc. seen in the report

---
##  Docker for nnUNet Baseline

The Docker Image for the nnUNet Baseline can be accessed from https://hub.docker.com/r/mzou2000/nnunet-10seed-predict

For the more detail can be found [here](nnUNet_Baseline/README.md)

##  Docker for nnUNet+VesselFM

The Docker Image for the nnUNet Baseline can be accessed from https://hub.docker.com/r/mzou2000/nnunet-5seed-predict

For the more detail can be found [here](nnUNet+vesselFM/README.md) 

##  Docker for GNN Approach

The Docker Image for the GNN approach can be accessed from: https://hub.docker.com/r/laith123/vessel-seg

Run this command: ```docker run --gpus all -v %cd%\input2:/input -v %cd%\output2:/output vessel-seg```

where '%cd%\input2' represents the directory with your inputs & %cd%\output2 represents the directory where you want the prediction masks to be stored

ENSURE THAT THESE DIRECTORIES EXIST if you do not create the directories and store your inputs in the input folder then the docker will not run! The inputs should be .nii.gz images of non-contrast CT images.

The models are not stored in this repository, but they are contained in the image.

---

##  Docker for VesselFM 3-Class Adaptation Approach

The Docker Image for the GNN approach can be accessed from: [https://hub.docker.com/r/artmcl/vesselfm-av-eval](https://hub.docker.com/r/artmcl/vesselfm-av-eval)

- Run this command (Just Input Images Directory, No Metrics Output): ```docker run --rm --gpus all -v "$PWD/input:/input:ro" -v "$PWD/output:/output" vesselfm-av-eval```
  
- Run this command (Input Images Directory + GT Labels Directory, Metrics Output): ```docker run --rm --gpus all \
                                                                                      -v "$PWD/input:/input:ro" \
                                                                                      -v "$PWD/labels:/labels:ro" \
                                                                                      -v "$PWD/output:/output" \
                                                                                      vesselfm-av-eval \
                                                                                      python -m vesselfm.seg.eval_inference /input /output --labels_dir /labels```

where '$PWD/input:/input' represents the directory with your inputs & $PWD/output:/output represents the directory where you want the prediction masks to be stored

ENSURE THAT THESE DIRECTORIES EXIST if you do not create the directories and store your inputs in the input folder then the docker will not run! The inputs should be .nii.gz images of non-contrast CT images.

The models are not stored in this repository, but they are contained in the image.

---
