# VesselFM 3-Class CT Artery/Vein Adaptation

## Overview

This folder contains the adaptation of vesselFM for 3-class segmentation of non-contrast CT volumes into artery, vein, and background. It wraps the original vesselFM foundation model with additional training and evaluation code to support supervised fine-tuning on the EC500 artery/vein dataset and to export Dockerized inference compatible with the course evaluation interface.

<img width="839" height="465" alt="demo" src="https://github.com/user-attachments/assets/5263dd4b-77dc-4480-9796-960a76854a9e" />

## Upstream dependency

All model architecture and most utility code are copied from the public vesselFM repository. This folder only adds and modifies a small number of files needed for the 3-class CT adaptation; the rest of the package behaves as in the original implementation.

## Layout and key files

### Top-level in `VesselFMAdaptionMethod/`

- `vesselfm/`  
  Local copy of the vesselFM Python package, including the UNet-style encoder–decoder backbone, inference helpers, and data utilities. Custom training and evaluation code for this project lives in `vesselfm/seg/` (see below).

- `av_ct.yaml`  
  Hydra/YAML configuration used for the CT artery/vein experiment. Defines:
  - Paths to preprocessed training images and labels.
  - Fold splits and dataset sampling options.
  - Patch size, batch size, and number of samples per volume.
  - Loss weights (Dice, clDice) and optimizer/scheduler hyperparameters.
  - Checkpoint paths for warm-starting from a vessel-only model.

- `Dockerfile`  
  Docker build recipe for the `artmcl/vesselfm-av-eval` image. Installs the vesselFM package, copies this adaptation code, and sets up the entrypoint for running end-to-end inference on a folder of CT NIfTI volumes mounted at `/input` with predictions written to `/output`.

### Custom training and evaluation scripts (`vesselfm/seg/`)

- `train_vessel.py`  
  Training script for adapting vesselFM to CT with a binary vessel-vs-background objective. This script:
  - Builds the CT dataset/dataloaders using `dataio.py`.
  - Freezes the bulk of the encoder and trains a new decoder head on CT data.
  - Uses Dice + clDice losses and logs vessel-level Dice/clDice over a validation split.
  - Saves checkpoints that can optionally be reused as initialization for 3-class training.

- `train_av.py`  
  Main training script for 3-class artery/vein/background segmentation. Compared with `train_vessel.py`, it:
  - Replaces the binary head with a 3-channel output for background, artery, and vein.
  - Uses a staged training schedule (e.g., starting with the encoder mostly frozen and then progressively unfreezing deeper blocks) to retain vesselFM’s pretrained features while adapting to CT.
  - Applies class-balanced losses to compensate for artery/vein class imbalance.
  - Tracks per-class Dice and clDice and saves the best 3-class checkpoint.

- `preprocess_av_ct_nii.py`
  Preprocessing script for images and labels for both validation and training sets:
  - Performs intensity normalization by either z score or percentile method (defined at runtime call).
  - Sets spacing to 1 mm, 1 mm, 1 mm.
  - Crops by factor of 10 for training set, no crop for validation.
  - Performs HU clip by specifying range at runtime call.

- `eval_preprocessing_av_ct_nii.py`
  Preprocessing script for images and labels for both validation and training sets (same as preprocess_av_ct_nii.py, just configured to work automatically with eval_inference.py):
  - Performs intensity normalization by either z score or percentile method (defined at runtime call).
  - Sets spacing to 1 mm, 1 mm, 1 mm.
  - Crops by factor of 10 for training set, no crop for validation.
  - Performs HU clip by specifying range at runtime call.

- `finetune_vesselfm.py`  
  Baseline fine-tuning script that follows the original vesselFM training procedure as closely as possible, but targets the CT dataset. Useful for ablation versus the custom `train_vessel.py` / `train_av.py` pipeline.

- `eval_inference.py`  
  Standalone evaluation script for a trained 3-class A/V model. Responsibilities:
  - For Docker Image.
  - Loads a trained checkpoint (usually produced by `train_av.py`).
  - Runs full-volume inference using sliding-window tiling but without any random cropping of evaluation images.
  - Writes out predicted multi-class NIfTI masks to an output directory.
  - Computes and logs Dice and clDice metrics for vessel (union of artery+vein), artery, and vein classes.

- `inference.py`  
  Core inference utilities, adapted from the original vesselFM implementation. Handles:
  - Construction and loading of the vesselFM backbone and segmentation heads.
  - Sliding-window inference over 3D volumes with configurable patch size and overlap.
  - Post-processing to convert raw logits into class labels for artery, vein, and background.  
  This module is used by both the training scripts (for validation) and by `eval_inference.py`.

### Shared utilities used by the adaptation

- `dataio.py`  
  I/O and dataset utilities for 3D NIfTI volumes. Provides:
  - The `NiftiVolume` wrapper and dataset classes for CT images and labels.
  - Train/validation splitting logic based on subject IDs / folds.
  - Data augmentation and normalization transforms appropriate for CT.

- `cldice_utils.py`  
  Implements the differentiable Soft clDice loss and corresponding metric utilities:
  - `SoftCLDiceLoss` for topology-aware training that emphasizes continuity of thin vessels.
  - `hard_cldice` and related helpers for computing clDice metrics on binarized predictions.  
  These functions are imported by the training and evaluation scripts to complement standard Dice loss.

> All other modules under `vesselfm/` are either unchanged from the upstream project or are slightly modified, and are not documented here individually. They include the core network definitions, configuration helpers, and support utilities provided by the original vesselFM authors.

## VesselFM Credit (https://github.com/bwittmann/vesselFM/tree/main)

```
@InProceedings{Wittmann_2025_CVPR,
    author    = {Wittmann, Bastian and Wattenberg, Yannick and Amiranashvili, Tamaz and Shit, Suprosanna and Menze, Bjoern},
    title     = {vesselFM: A Foundation Model for Universal 3D Blood Vessel Segmentation},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {20874-20884}
}
```

## Top Workflow

<img width="1008" height="533" alt="ModelLayout_TopFlow" src="https://github.com/user-attachments/assets/bf3ef9bb-5ac1-44a5-a931-cea457f81693" />

### Stage 1 Training - train_vessel.py

- Performing fine tuning of VesselFM base checkpoint through the use of the Vessel Head (binary segmentation).

### Stage 2 Training - train_av.py

#### S1 Training

<img width="1015" height="710" alt="ModelLayout_TrainingS1" src="https://github.com/user-attachments/assets/0ef04ee1-f6e7-4b36-b4be-3e7a3bf5578d" />

#### S2 Training

<img width="1015" height="680" alt="ModelLayout_TrainingS2" src="https://github.com/user-attachments/assets/c58a071d-ec34-404f-a432-4abbe3bebcb7" />

#### S3 Training

<img width="1015" height="719" alt="ModelLayout_TrainingS3" src="https://github.com/user-attachments/assets/d9b8ce00-ba5c-4b3f-8562-4d4e43ac1fc7" />

## Typical workflows

### 1. Fine-tune vesselFM on CT vessels (binary warm-up)

```
python -m vesselfm.seg.train_vessel \
  experiment=av_ct \
  data.train_images=/path/to/imagesTr_pre \
  data.train_labels=/path/to/labelsTr
```

### 2. Train the 3-class artery/vein model

```
python -m vesselfm.seg.train_av \
  experiment=av_ct \
  data.train_images=/path/to/imagesTr_pre \
  data.train_labels=/path/to/labelsTr
```

### 3. Run evaluation / inference

```
python -m vesselfm.seg.eval_inference \
  /path/to/input_nifti_dir \
  /path/to/output_dir
```




##  Docker for VesselFM 3-Class Adaptation Approach

The Docker Image for the VesselFM 3-Class Adaptation approach can be accessed from: [https://hub.docker.com/r/artmcl/vesselfm-av-eval](https://hub.docker.com/r/artmcl/vesselfm-av-eval)

### Directory Mode (Recommended)

- Run this command (Just Input Images Directory, No Metrics Output): ```docker run --rm --gpus all \
                                                                        -v "$PWD/input:/input:ro" \
                                                                        -v "$PWD/output:/output" \
                                                                        vesselfm-av-eval```
  
- Run this command (Input Images Directory + GT Labels Directory, Metrics Output): ```docker run --rm --gpus all \
                                                                                      -v "$PWD/input:/input:ro" \
                                                                                      -v "$PWD/labels:/labels:ro" \
                                                                                      -v "$PWD/output:/output" \
                                                                                      vesselfm-av-eval \
                                                                                      python -m vesselfm.seg.eval_inference /input /output --labels_dir /labels```

where '$PWD/input:/input:ro' represents the directory with your inputs & '$PWD/output:/output' represents the directory where you want the prediction masks to be stored.

Additionally, '$PWD/labels:/labels:ro' can be added for an metric output that is saved with the predicted masks in '$PWD/output:/output'.

ENSURE THAT THESE DIRECTORIES EXIST if you do not create the directories and store your inputs in the input folder then the docker will not run! The inputs should be .nii.gz images of non-contrast CT images.

### Single-file Mode

- Run this command (Just Input Image Filepath, No Metric Output): ```docker run --rm --gpus all \
                                                                                      -v "$PWD/input/image_010.nii.gz:/in.nii.gz:ro" \
                                                                                      -v "$PWD/output:/output" \
                                                                                      vesselfm-av-eval \
                                                                                      python -m vesselfm.seg.eval_inference /in.nii.gz /output/pred_mask_010.nii.gz```

- Run this command (Input Image Filepath + GT Label Path, Metric Output): ```docker run --rm --gpus all \
                                                                                      -v "$PWD/input/image_010.nii.gz:/in.nii.gz:ro" \
                                                                                      -v "$PWD/labels/label_010.nii.gz:/gt.nii.gz:ro" \
                                                                                      -v "$PWD/output:/output" \
                                                                                      vesselfm-av-eval \
                                                                                      python -m vesselfm.seg.eval_inference /in.nii.gz /output/pred_mask_010.nii.gz --labels_dir /gt.nii.gz```

where '$PWD/input/image_010.nii.gz:/in.nii.gz:ro' represents the filepath with your input image file of .nii.gz type & '$PWD/output:/output' represents the directory where you want the prediction masks to be stored.

Additionally, '$PWD/labels/label_010.nii.gz:/gt.nii.gz:ro' can be added for an metric output that is saved with the predicted masks in '$PWD/output:/output'.

The models are not stored in this repository, but they are contained in the image.

':ro' is added onto the directories and files optionally to prevent accidental writes.
