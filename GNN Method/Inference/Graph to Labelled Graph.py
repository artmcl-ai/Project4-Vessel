#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import pickle
from pathlib import Path
from GAT_MODEL import EdgeGATLayer, EdgeGATNet

import torch
import numpy as np

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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.load("GAT_full_model.pth", map_location=DEVICE)
model = model.to(DEVICE) 
model.eval()

print("Loaded model.")

input_dir = 'output/test1.gpickle'
out_path = 'output/labelled1.gpickle'

print(f"\nProcessing graph")

with open(input_dir, "rb") as f:
    G = pickle.load(f)

print("Graph loaded")

G_pred = predict_graph_labels(model, G, device=DEVICE)

with open(out_path, "wb") as f:
    pickle.dump(G_pred, f)


# In[ ]:




