#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np

def reconstruct_artery_vein_from_graph(G, shape, spacing=(1,1,1)):
    Z, Y, X = shape

    artery_vol  = np.zeros(shape, dtype=np.uint8)
    vein_vol    = np.zeros(shape, dtype=np.uint8)
    unknown_vol = np.zeros(shape, dtype=np.uint8)

    sz, sy, sx = spacing

    for br in G.graph.get("branches", []):
        vox = br["voxels"]          # (N,3) array
        u, v = br["node_u"], br["node_v"]

        label  = G.edges[u, v]["label"]      # 1=artery, 0=vein, -1=unknown
        R_edge = float(G.edges[u, v]["radius"])

        for (z, y, x) in vox:
            z = int(z); y = int(y); x = int(x)

            # Compute voxel extent from radius
            rz = int(np.ceil(R_edge / sz))
            ry = int(np.ceil(R_edge / sy))
            rx = int(np.ceil(R_edge / sx))
            # Bounding box
            z0, z1 = max(0, z-rz), min(Z, z+rz+1)
            y0, y1 = max(0, y-ry), min(Y, y+ry+1)
            x0, x1 = max(0, x-rx), min(X, x+rx+1)

            # Create spherical mask in physical space
            zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
            sphere = (
                ((zz - z)*sz)**2 +
                ((yy - y)*sy)**2 +
                ((xx - x)*sx)**2
            ) <= R_edge**2

            # Paint sphere according to label
            if label == 1:
                artery_vol[z0:z1, y0:y1, x0:x1][sphere] = 1
            elif label == 0:
                vein_vol[z0:z1, y0:y1, x0:x1][sphere]   = 1
            else:
                unknown_vol[z0:z1, y0:y1, x0:x1][sphere] = 1

    # Combined result
    combined = np.zeros(shape, dtype=np.uint8)
    combined[artery_vol == 1]  = 1
    combined[vein_vol == 1]    = 2
    combined[unknown_vol == 1] = 3

    return artery_vol, vein_vol, unknown_vol, combined


# In[9]:


import os
import pickle
from pathlib import Path
import nibabel as nib

# Directories
graph_dir   = Path("/projectnb/ec500kb/projects/Project_4/Graph_Creations/GNN_predicted_graphs")
nii_dir     = Path("/projectnb/ec500kb/projects/Project_4/Graph_Creations/Binary_Masks_Predictions/")
output_dir  = Path("/projectnb/ec500kb/projects/Project_4/Graph_Creations/GNN_recon_step1/")
output_dir.mkdir(parents=True, exist_ok=True)
# Loop over all predicted graph files
for graph_file in sorted(graph_dir.glob("finetune_graph*_predicted.gpickle")):

    # Extract case number (e.g. 001)
    case_num = graph_file.stem.split("finetune_graph")[1].split("_")[0]

    print(f"\n→ Processing graph: {graph_file.name}  (Case {case_num})")

    # Find matching mask file
    nii_file = next(nii_dir.glob(f"*{case_num}.nii.gz"), None)
    if nii_file is None:
        print(f"  ⚠ No segmentation found for case {case_num}, skipping.")
        continue

    print(f"  ✓ Found matching mask: {nii_file.name}")

    # Load graph
    with open(graph_file, "rb") as f:
        G = pickle.load(f)

    # Load segmentation + meta info
    nii = nib.load(nii_file)
    shape   = nii.shape
    affine  = nii.affine
    spacing = nii.header.get_zooms()[:3]

    # Reconstruction
    artery, vein, unknown, combined = reconstruct_artery_vein_from_graph(G, shape, spacing)

    # Save result
    out_path = output_dir / f"case_{case_num}_combined_reconstructed.nii.gz"
    nib.save(nib.Nifti1Image(combined, affine), out_path)

    print(f"  ✓ Saved reconstruction → {out_path}")

print("\n🎉 Done reconstructing all available cases!")


# In[1]:


import pickle
import sys
import os
import numpy as np
import nibabel as nib
from pathlib import Path

def propagate_nearest_label(seg, recon):
    """
    Fill seg>=1 region using nearest NONZERO voxel from recon.
    recon is typically your artery/vein skeleton.
    """

    seg    = seg.astype(np.int32)
    recon  = recon.astype(np.int32)

    # Region to fill
    region_mask = seg > 0

    # Valid labeled seeds in recon
    seed_mask = recon > 0

    # Nearest labeled voxel in recon
    _, nearest_idx = distance_transform_edt(
        ~seed_mask,
        return_indices=True
    )

    nz, ny, nx = nearest_idx

    # Get the label at the nearest recon voxel
    nearest_values = recon[nz, ny, nx]

    # Output: only fill seg>=1 region
    out = np.zeros_like(seg, dtype=np.int32)
    out[region_mask] = nearest_values[region_mask]

    return out


# In[3]:


import pickle
import sys
import os
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.ndimage import convolve, label, distance_transform_edt

# Directories
recon_dir = Path("/projectnb/ec500kb/projects/Project_4/Graph_Creations/GNN_recon_step1/")
mask_dir  = Path("/projectnb/ec500kb/projects/Project_4/Graph_Creations/Binary_Masks_Predictions/")
output_dir = Path("/projectnb/ec500kb/projects/Project_4/Graph_Creations/GNN_recon_step2/")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Scanning reconstructed files in: {recon_dir}")

for recon_file in sorted(recon_dir.glob("case_*_combined_reconstructed.nii.gz")):
    
    # Extract case number
    case_num = recon_file.stem.split("_")[1]  # gets ###

    print(f"\n→ Processing Case {case_num}")

    # Find segmentation file
    seg_file = next(mask_dir.glob(f"*{case_num}.nii.gz"), None)

    if seg_file is None:
        print(f"  ⚠ No original segmentation for case {case_num}, skipping.")
        continue

    print(f"  ✓ Found seg file: {seg_file.name}")

    # Load original segmentation
    seg_img = nib.load(seg_file)
    seg = seg_img.get_fdata().astype(np.int32)

    # Load reconstructed segmentation
    recon_img = nib.load(recon_file)
    recon_seg = recon_img.get_fdata().astype(np.int32)
    
    out_path = output_dir / f"case_{case_num}_nearest_label_propagated.nii.gz"
    
    if out_path.exists():
        print(f"Output already exists, skipping.")
        continue

    # Propagate labels
    out = propagate_nearest_label(seg, recon_seg)

    # Save output
    nib.save(nib.Nifti1Image(out, seg_img.affine), out_path)

    print(f"  ✓ Saved nearest label propagation → {out_path}")

print("\n🎉 Completed nearest-label propagation for all available cases!")

