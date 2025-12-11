# -*- coding: utf-8 -*-
#!/usr/bin/env python
import pickle
import sys
import os
import numpy as np
import nibabel as nib
import networkx as nx
from scipy.ndimage import convolve, label, distance_transform_edt
import torch
#Needs Python 3.12.4

def skeletonize_3d_safe(vol):
    try:
        from skimage.morphology import skeletonize_3d
        return skeletonize_3d(vol.astype(bool)).astype(np.uint8) #This should work with correct version of python
    except ImportError:
        from skimage.morphology import skeletonize
        skel = np.zeros_like(vol, dtype=bool) #failsafe
        for z in range(vol.shape[0]):
            skel[z] = skeletonize(vol[z])
        return skel.astype(np.uint8)

def compute_degree_fast(skel):
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    neighbors = convolve(skel.astype(np.uint8), kernel)
    return neighbors * skel

def compute_curvature(branch_voxels):
    pts = branch_voxels.astype(float)
    if len(pts) < 3:
        return 0.0

    diffs = np.diff(pts, axis=0)
    norms = np.linalg.norm(diffs, axis=1, keepdims=True) + 1e-8
    tangents = diffs / norms

    dots = np.sum(tangents[:-1] * tangents[1:], axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)
    return float(np.mean(angles))

def compute_tortuosity(branch_voxels):
    K = len(branch_voxels)
    if K < 2:
        return 1.0
    path_len = float(K)
    d = np.linalg.norm(branch_voxels[-1] - branch_voxels[0]) + 1e-8
    return float(path_len / d)

def compute_branch_radius(branch_voxels, edt):
    Z, Y, X = edt.shape
    vals = []
    for (z, y, x) in branch_voxels:
        z, y, x = int(z), int(y), int(x)
        if 0 <= z < Z and 0 <= y < Y and 0 <= x < X:
            vals.append(edt[z, y, x])
    if len(vals) == 0:
        return 0.0
    return float(np.mean(vals))

def compute_branch_features(branches, seg_full):
    edt = distance_transform_edt(seg_full > 0)
    out = []
    for br in branches:
        vox = br["voxels"]
        rad = compute_branch_radius(vox, edt)
        curv = compute_curvature(vox)
        tort = compute_tortuosity(vox)
        out.append({
            "node_u": br["node_u"],
            "node_v": br["node_v"],
            "voxels": vox,
            "length_vox": float(len(vox)),
            "radius": rad,
            "curvature": curv,
            "tortuosity": tort
        })
    return out
    
def extract_branches_fast(skel, nodes_mask, links_mask):
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labeled_links, num = label(links_mask, structure)
    node_positions = np.column_stack(np.where(nodes_mask))
    node_tuples = [tuple(p) for p in node_positions]
    node_to_id = {p: i for i, p in enumerate(node_tuples)}
    Z, Y, X = skel.shape
    neighbors26 = np.array(
        [[dz, dy, dx]
         for dz in (-1, 0, 1)
         for dy in (-1, 0, 1)
         for dx in (-1, 0, 1)
         if not (dz == 0 and dy == 0 and dx == 0)],
        dtype=int
    )
    branches = []
    for comp in range(1, num + 1):
        coords = np.column_stack(np.where(labeled_links == comp))
        nbrs = coords[:, None, :] + neighbors26[None, :, :]
        valid = (
            (nbrs[..., 0] >= 0) & (nbrs[..., 0] < Z) &
            (nbrs[..., 1] >= 0) & (nbrs[..., 1] < Y) &
            (nbrs[..., 2] >= 0) & (nbrs[..., 2] < X)
        )
        nbr_valid = nbrs[valid]
        touching = nodes_mask[nbr_valid[:, 0],
                              nbr_valid[:, 1],
                              nbr_valid[:, 2]]
        node_vox = nbr_valid[touching]
        if len(node_vox) < 2:
            continue
        un = np.unique(node_vox, axis=0)
        if len(un) == 2:
            u, v = un
        else:
            dif = un[:, None, :] - un[None, :, :]
            dst = np.linalg.norm(dif, axis=2)
            i, j = np.unravel_index(np.argmax(dst), dst.shape)
            u, v = un[i], un[j]
        branches.append({
            "node_u": node_to_id[tuple(u)],
            "node_v": node_to_id[tuple(v)],
            "voxels": coords
        })
    return branches, node_positions

def skeleton_array_to_graph(skel, seg_full):
    degree = compute_degree_fast(skel)
    nodes_mask = (degree != 2) & (skel == 1)
    links_mask = (degree == 2) & (skel == 1)
    branches, node_positions = extract_branches_fast(skel, nodes_mask, links_mask)
    branches = compute_branch_features(branches, seg_full)
    edt = distance_transform_edt(seg_full > 0)
    G = nx.Graph()
    node_positions = np.asarray(node_positions, int)
    for i, (z, y, x) in enumerate(node_positions):
      G.add_node(
          i,
          coord_vox=np.array([z, y, x], float),
          degree=int(degree[int(z), int(y), int(x)])
      )
    for br in branches:
        u, v = br["node_u"], br["node_v"]
        vox = br["voxels"]
        vals = seg_full[vox[:, 0], vox[:, 1], vox[:, 2]]
        artery_votes = np.sum(vals == 1)
        vein_votes = np.sum(vals == 2)
        if artery_votes == 0 and vein_votes == 0:
            edge_label = -1  # background or unknwon?
        else:
            edge_label = 1 if artery_votes > vein_votes else 0  # 1 = artery and 0 = vein
        G.add_edge(
            u, v,
            length_vox=br["length_vox"],
            radius=br["radius"],
            curvature=br["curvature"],
            tortuosity=br["tortuosity"],
            label=edge_label
        )
    G.graph["branches"] = branches
    return G

def reconstruct_artery_vein_volumes(G, shape):
    artery_vol = np.zeros(shape, dtype=np.uint8)
    vein_vol = np.zeros(shape, dtype=np.uint8)
    unknown_vol = np.zeros(shape, dtype=np.uint8)
    branches = G.graph.get("branches", [])
    for br in branches:
        u = br["node_u"]
        v = br["node_v"]
        if not G.has_edge(u, v):
            continue
        label = G.edges[u, v]["label"]  # 1=artery,0=vein
        vox = br["voxels"] 
        z = vox[:, 0]
        y = vox[:, 1]
        x = vox[:, 2]
        if label == 1:
            artery_vol[z, y, x] = 1
        elif label == 0:
            vein_vol[z, y, x] = 1
        else:
            unknown_vol[z, y, x] = 1
    combined_vol = np.zeros(shape, dtype=np.uint8)
    combined_vol[artery_vol == 1] = 1
    combined_vol[vein_vol == 1] = 2
    combined_vol[unknown_vol == 1] = 3
    return artery_vol, vein_vol, unknown_vol, combined_vol

def main(input_seg_path, graph_out_path):
    seg_img = nib.load(input_seg_path)
    seg = seg_img.get_fdata()
    affine = seg_img.affine
    seg_bin = (seg > 0).astype(np.uint8)
    skel = skeletonize_3d_safe(seg_bin)
    #nib.save(nib.Nifti1Image(skel, affine), skeleton_out_path)
    #print("Skeleton saved to", skeleton_out_path)
    G = skeleton_array_to_graph(skel, seg)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    with open(graph_out_path, "wb") as f:
        pickle.dump(G, f)
    print("Graph saved to", graph_out_path)
    shape = seg_bin.shape
    #artery_vol, vein_vol, unknown_vol, combined_vol = reconstruct_artery_vein_volumes(G, shape)
    if input_seg_path.endswith(".nii.gz"):
        base = input_seg_path[:-7]
    else:
        base, _ = os.path.splitext(input_seg_path)

    # Save artery/vein/combined volumes
    #nib.save(nib.Nifti1Image(artery_vol, affine), base + "_arteries.nii.gz")
    #nib.save(nib.Nifti1Image(vein_vol, affine),   base + "_veins.nii.gz")
    #nib.save(nib.Nifti1Image(combined_vol, affine), base + "_artery_vein_combined.nii.gz")
    print("Artery/vein volumes saved with base", base)
    #nib.save(nib.Nifti1Image(seg_bin, affine), recon_out_path)
    #print("Reconstructed segmentation saved to", recon_out_path)

if __name__ == "__main__":
    input_path = "/projectnb/ec500kb/projects/Project_4/Graph_Creations/Binary_Masks_Predictions/"
    for i in os.listdir(input_path):
        output_mask_path = "/projectnb/ec500kb/projects/Project_4/Graph_Creations/Graphs_from_FineTuned_VesselFM/finetune_graph" + i[-10:-7] + ".gpickle"
        in_path = input_path + i
        print(i)
        if os.path.exists(output_mask_path):
            print(f"Skipping {output_mask_path} → Output already exists.")
            continue
        main(in_path, output_mask_path)
