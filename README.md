# Project Overview

This repository focuses on developing a model capable of accurately segmenting and classifying veins and arteries from non-contrast CT images. To support this goal, we leverage **VesselFM** — a pretrained model designed for vessel segmentation across diverse imaging domains. You can find VesselFM on GitHub: https://github.com/bwittmann/vesselFM

## Repository Structure

The project is divided into three primary folders, each representing a distinct experimental method:

- **GNN Approach**  
  A graph neural network designed to leverage vessel topology and connectivity.

- **UNet Approach**  
  A convolutional neural network architecture optimized for CT image segmentation.

- **Modified VesselFM**  
  A customized version of VesselFM with significant architectural alterations to enable multiclass output.

Each folder contains the code required for model development, training, and evaluation; however, the data and models themselves are too large, so they are kept in BU SCC.

---
