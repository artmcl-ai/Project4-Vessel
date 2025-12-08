# Inference Pipeline Documentation

## Overview

This repository contains a multi-stage inference pipeline for processing CT data into a final 3D reconstruction. Each stage is modular and represented by a dedicated Python script.

The workflow transforms raw CT volumes → binary segmentation → graph representation → labeled graph → 3D reconstruction → final polished output.

---

## Pipeline Structure

| Step                               | Script                       | Description                                                                               | Output                   |
| ---------------------------------- | ---------------------------- | ----------------------------------------------------------------------------------------- | ------------------------ |
| 1️⃣ CT → Binary Segmentation       | `CT_to_Binary_Seg.py`        | Generates a binary mask of vessels from CT images                         | Binary segmentation mask |
| 2️⃣ Binary Seg → Graph             | `Binary_Seg_to_Graph.py`     | Converts the mask to a node/edge graph    | Graph representation     |
| 3️⃣ Graph → Labeled Graph          | `Graph to Labelled Graph.py` | Use a GAT to labels edges in graph as either veins or arteries     | Labeled graph            |
| 4️⃣ Labeled Graph → Reconstruction | `Labelled_Graph_to_Recon.py` | Converts graph back into a geometric 3D model with labeled structure                      | 3D reconstruction        |
| 5️⃣ Reconstruction → Final Mask   | `Recon_to_Final.py`          | Uses the reconstruction from the GAT to support an augmented version of VesselFM a labelled mask | Final Mask output       |

