# VesselFM 3-Class CT Artery/Vein Adaptation

## Overview

This folder contains my adaptation of vesselFM for 3-class segmentation of non-contrast CT volumes into artery, vein, and background. It wraps the original vesselFM foundation model with additional training and evaluation code to support supervised fine-tuning on the EC500 artery/vein dataset and to export Dockerized inference compatible with the course evaluation interface.

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

- `finetune_vesselfm.py`  
  Baseline fine-tuning script that follows the original vesselFM training procedure as closely as possible, but targets the CT dataset. Useful for ablation versus the custom `train_vessel.py` / `train_av.py` pipeline.

- `eval_inference.py`  
  Standalone evaluation script for a trained 3-class A/V model. Responsibilities:
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

> All other modules under `vesselfm/` are unchanged from the upstream project and are not documented here individually. They include the core network definitions, configuration helpers, and support utilities provided by the original vesselFM authors.

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
- From the Docker Image:

```
docker run --gpus all \
  -v $PWD/input:/input \
  -v $PWD/output:/output \
  artmcl/vesselfm-av-eval
```
