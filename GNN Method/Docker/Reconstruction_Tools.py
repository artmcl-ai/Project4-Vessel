#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
from scipy.ndimage import distance_transform_edt

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

