## Overview

This repository contains two main folders that provide all necessary files for both model training and inference. Each folder includes its own README with detailed explanations of its role within the full pipeline.

At a high level, the architecture works as follows:

1. **Initial Binary Segmentation**
   A fine-tuned VesselFM model generates a binary vessel segmentation mask from the input CT images.

2. **Graph Construction**
   The segmentation mask is converted into a graph where:

   * **Edges** represent vessel segments
   * **Nodes** represent junctions and endpoints

3. **Graph Neural Network Classification**
   A Graph Attention Network (GAT) model classifies each edge as either artery or vein.

4. **Mask Reconstruction**
   The classified vessel graph is transformed back into a multiclass segmentation mask.

5. **Final Refinement with VesselFM**
   To mitigate GNN limitations and improve anatomical consistency, a secondary augmented VesselFM model takes:

   * **Two input channels** (CT scan + reconstructed mask)
  
   which is used to produce:
   
   * **Two output channels** (artery mask + vein mask)

<p align="center">
  <img width="896" height="470" alt="Model Architecture" src="https://github.com/user-attachments/assets/931baf4a-9842-4cc2-8510-3f5779cc83a3" />
</p>
