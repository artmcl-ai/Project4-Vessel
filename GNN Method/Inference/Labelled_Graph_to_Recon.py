#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle
from pathlib import Path
import nibabel as nib
import sys
import numpy as np
import nibabel as nib
from scipy.ndimage import convolve, label, distance_transform_edt
from Reconstruction_Tools import reconstruct_artery_vein_from_graph, propagate_nearest_label

graph_file   = "output/labelled1.gpickle"
mask_file     = "output/binary1.nii.gz"
out_path  = "output/recon1.nii.gz"


with open(graph_file, "rb") as f:
    G = pickle.load(f)

nii = nib.load(mask_file)
shape   = nii.shape
affine  = nii.affine
spacing = nii.header.get_zooms()[:3]

artery, vein, unknown, combined = reconstruct_artery_vein_from_graph(G, shape, spacing)

seg_img = nib.load(mask_file)
seg = seg_img.get_fdata().astype(np.int32)'
out = propagate_nearest_label(seg, combined)
nib.save(nib.Nifti1Image(out, seg_img.affine), out_path)
print(f"  ✓ Saved nearest label propagation → {out_path}")


# In[ ]:




