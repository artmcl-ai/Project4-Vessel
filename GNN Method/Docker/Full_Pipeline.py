#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
import torch
import nibabel as nib
import numpy as np
from monai.transforms import (
    LoadImage, EnsureChannelFirst, NormalizeIntensity,
    RandSpatialCropSamples, RandFlipd, RandRotate90d,
    RandGaussianNoised, ScaleIntensityRange, Compose
)
from monai.inferers import sliding_window_inference
from monai.networks.nets import DynUNet
import pickle
import networkx as nx
from scipy.ndimage import convolve, label, distance_transform_edt
from skimage.morphology import skeletonize
from pathlib import Path
from GAT_MODEL import EdgeGATLayer, EdgeGATNet
from Reconstruction_Tools import reconstruct_artery_vein_from_graph, propagate_nearest_label
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd,
    ScaleIntensityRangeD, NormalizeIntensityd,
    ConcatItemsd
)
import argparse

def graph_to_tensors(G):
    node_ids = list(G.nodes())
    id_to_idx = {n: i for i, n in enumerate(node_ids)}
    N = len(node_ids)
    coords = np.array([G.nodes[n]["coord_vox"] for n in node_ids], dtype=float)  # (N, 3)
    degrees = np.array([G.nodes[n]["degree"] for n in node_ids], dtype=float)    # (N,)

    z = coords[:, 0]
    y = coords[:, 1]
    x = coords[:, 2]
    def norm1d(v):
        vmin = v.min()
        vmax = v.max()
        return (v - vmin) / ( (vmax - vmin) + 1e-6 )

    z_norm = norm1d(z)
    y_norm = norm1d(y)
    x_norm = norm1d(x)
    incident_radii = [[] for _ in range(N)]
    for u, v, data in G.edges(data=True):
        r = float(data["radius"])
        incident_radii[id_to_idx[u]].append(r)
        incident_radii[id_to_idx[v]].append(r)

    mean_radius = np.array([
        np.mean(rs) if len(rs) > 0 else 0.0
        for rs in incident_radii
    ], dtype=float)
    X_np = np.stack([degrees, z_norm, y_norm, x_norm, mean_radius], axis=1)
    X = torch.tensor(X_np, dtype=torch.float32)
    edges = list(G.edges())
    E = len(edges)
    edge_idx_np = np.array([[id_to_idx[u], id_to_idx[v]] for (u, v) in edges], dtype=np.int64)
    edge_index = torch.tensor(edge_idx_np.T, dtype=torch.long)
    edge_feats = []
    edge_labels = []
    for (u, v) in edges:
        data = G.edges[u, v]
        radius     = float(data["radius"])
        curvature  = float(data["curvature"])
        tortuosity = float(data["tortuosity"])
        length_vox = float(data["length_vox"])
        cu = G.nodes[u]["coord_vox"].astype(float)
        cv = G.nodes[v]["coord_vox"].astype(float)
        dz, dy, dx = (cv - cu)
        dist = float(np.linalg.norm(cv - cu))

        edge_feats.append([
            radius,
            curvature,
            tortuosity,
            length_vox,
            dz,
            dy,
            dx,
            dist
        ])

        edge_labels.append(int(data.get("label", -1)))

    edge_attr = torch.tensor(np.array(edge_feats, dtype=float), dtype=torch.float32)   
    y = torch.tensor(np.array(edge_labels, dtype=int), dtype=torch.long)            

    return X, edge_index, edge_attr, y


def predict_graph_labels(model, G,device):
    X, edge_index, edge_attr, _ = graph_to_tensors(G)
    X = X.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(X, edge_index, edge_attr)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
    for (edge, label) in zip(G.edges(), pred):
        u, v = edge
        G.edges[u, v]["label"] = int(label)
    return G

def build_vesselfm_dyunet():
    model = DynUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        kernel_size=[[3, 3, 3]] * 6,
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        upsample_kernel_size=[[2, 2, 2]] * 5,
        filters=[32, 64, 128, 256, 320, 320],
        res_block=True,
    )
    return model

infer_transforms = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    ScaleIntensityRange(
            a_min=-1000,
            a_max=400,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
    NormalizeIntensity(nonzero=True, channel_wise=True),
])


def run_inference(model_path, input_nifti, device, roi_size=(96, 96, 96)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_vesselfm_dyunet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    img_np = infer_transforms(input_nifti)
    img_torch = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_logits = sliding_window_inference(
            img_torch,
            roi_size=roi_size,
            sw_batch_size=1,
            predictor=model,
            overlap=0.25,
        )

    prob = torch.sigmoid(pred_logits)
    prob_mask_np = prob.cpu().numpy()[0, 0]
    pred_mask = (prob > 0.5).float()
    #orig = nib.load(input_nifti)
    #pred_nifti = nib.Nifti1Image(prob_mask_np.astype(np.float32), orig.affine, orig.header)
    #nib.save(pred_nifti, output_nifti) #THIS GIVES THE PROBABILITY MATRIX 
    pred_mask_np = pred_mask.cpu().numpy()[0, 0]

    #orig = nib.load(input_nifti)
    pred_mask_np = pred_mask_np.astype(np.uint8)
    #pred_nifti = nib.Nifti1Image(pred_mask_np, orig.affine, orig.header)
    #nib.save(pred_nifti, output_nifti)
    #print(f"Saved prediction mask to: {output_nifti}")
    return pred_mask_np

def skeletonize_3d_safe(vol):
    vol = np.ascontiguousarray(vol.astype(bool))
    skel = np.zeros_like(vol, dtype=bool)
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
            edge_label = -1  # unknown / unlabeled
        else:
            edge_label = 1 if artery_votes > vein_votes else 0  # 1=artery,0=vein

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

def load_model(checkpoint_path, device):
    print(f"Loading model weights from {checkpoint_path}")
    model = DynUNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=3,
        kernel_size=[[3, 3, 3]] * 6,
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        upsample_kernel_size=[[2, 2, 2]] * 5,
        filters=[32, 64, 128, 256, 320, 320],
        res_block=True,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def run_segmentation(model, img_np, recon_np, affine, output_path, device,
                     roi_size=(96, 96, 96)):
    infer_transforms = Compose([
        #EnsureChannelFirstd(keys=["image", "recon"]),

        ScaleIntensityRangeD(keys=["image"], a_min=-1000, a_max=400,
                             b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),

        ScaleIntensityRangeD(keys=["recon"], a_min=0, a_max=2,
                             b_min=0.0, b_max=1.0, clip=True),

        ConcatItemsd(keys=["image", "recon"], name="image"),
    ])
    if img_np.ndim == 3:
        img_np = img_np[np.newaxis, ...]
    if recon_np.ndim == 3:
        recon_np = recon_np[np.newaxis, ...]
    data = infer_transforms({"image": img_np, "recon": recon_np})
    image_tensor = data["image"].unsqueeze(0).to(device)
    print("Doing sliding window inference")
    with torch.no_grad():
        logits = sliding_window_inference(
            image_tensor,
            roi_size=roi_size,
            sw_batch_size=1,
            predictor=model)
        probs = torch.softmax(logits, dim=1)
        seg = torch.argmax(probs, dim=1).cpu().numpy().astype(np.uint8)[0]
    print(f"Saving segmentation to: {output_path}")
    nib.save(nib.Nifti1Image(seg, affine), output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    in_path_dir = args.input_dir
    out_path_dir = args.output_dir
    
    for filename in os.listdir(in_path_dir):
        in_path = os.path.join(in_path_dir, filename)
        output_name = 'pred_' + filename
        output_path = os.path.join(out_path_dir, output_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = "vesselfm_finetuned_monai.pt"
        img = nib.load(in_path)
        affine = img.affine
        spacing = img.header.get_zooms()[:3]
        print(f"Starting segmentation for: {in_path}")
        print('Converting CT to Binary Masks')
        binary_mask = run_inference(model_path, in_path,device)
        print('Finished CT to Binary Masks')

        seg_bin = (binary_mask > 0).astype(np.uint8)
        skel = skeletonize_3d_safe(seg_bin)
        print('Making Graph (can take ~9min)')
        G = skeleton_array_to_graph(skel, binary_mask)
        print('Finished Graph')

        model = EdgeGATNet(
            node_in_dim=5,
            edge_in_dim=8,
            hidden_dims=[64, 128, 128, 64],
            num_classes=2,
            dropout=0.1,
        )

        # Load the converted state_dict (.pt)
        state_dict = torch.load("GAT_full_model.pt", map_location=device)
        model.load_state_dict(state_dict)

        # Send to GPU/CPU + eval
        model = model.to(device)
        model.eval()

        print("Loaded GAT model")
        G_pred = predict_graph_labels(model, G, device=device)
        print("Graph Labelled")

        print('Beginning Reconstruction')
        shape  = binary_mask.shape
        artery, vein, unknown, combined = reconstruct_artery_vein_from_graph(G_pred, shape, spacing)
        seg = binary_mask.astype(np.int32)
        recon_mask = propagate_nearest_label(seg, combined)
        print('Finished Reconstruction')

        print('Begin Final Segmentation Mask Creation')
        img_np = img.get_fdata().astype(np.float32)
        recon_np = recon_mask.astype(np.float32)
        checkpoint = "vesselfm_finetuned_multiclass_weighted_dice.pt"
        model = load_model(checkpoint, device)
        run_segmentation(model, img_np, recon_np, affine, output_path, device)
    print('Complete!')