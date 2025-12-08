#!/usr/bin/env python
# coding: utf-8

# In[17]:


from scipy.ndimage import distance_transform_edt

def assign_true_edge_labels_from_mask(G_pred, gt_seg):
    my_list = []
    artery_counter = 0
    vein_counter = 0
    unknown_counter = 0

    # Precompute nearest-labeled voxel indices ONCE
    nonzero = (gt_seg != 0)
    _, nearest_idx = distance_transform_edt(~nonzero, return_indices=True)

    branches = G_pred.graph["branches"]

    for br in branches:
        u = br["node_u"]
        v = br["node_v"]

        if not G_pred.has_edge(u, v):
            continue

        vox = br["voxels"]
        z, y, x = vox[:,0], vox[:,1], vox[:,2]

        # Pull nearest nonzero replacement for zeros
        zz, yy, xx = nearest_idx[:, z, y, x]
        vals = gt_seg[zz, yy, xx]

        artery_votes = np.sum(vals == 1)
        vein_votes   = np.sum(vals == 2)

        if artery_votes == 0 and vein_votes == 0:
            label = -1
            unknown_counter += 1
        else:
            if artery_votes > vein_votes:
                label = 1  # artery
                artery_counter += 1
            else:
                label = 0  # vein
                vein_counter += 1

        G_pred.edges[u, v]["label"] = int(label)
    print(f"Arteries: {artery_counter}, Veins: {vein_counter}, Unknown: {unknown_counter}")

    return G_pred


# In[18]:


import torch
import numpy as np

def graph_to_tensors(G):
    node_ids = list(G.nodes())
    id_to_idx = {n: i for i, n in enumerate(node_ids)}
    N = len(node_ids)

    # coords and degree
    coords = np.array([G.nodes[n]["coord_vox"] for n in node_ids], dtype=float)  # (N, 3)
    degrees = np.array([G.nodes[n]["degree"] for n in node_ids], dtype=float)    # (N,)

    z = coords[:, 0]
    y = coords[:, 1]
    x = coords[:, 2]

    # normalize coordinates to [0,1] based on graph extents
    def norm1d(v):
        vmin = v.min()
        vmax = v.max()
        return (v - vmin) / ( (vmax - vmin) + 1e-6 )

    z_norm = norm1d(z)
    y_norm = norm1d(y)
    x_norm = norm1d(x)

    # mean incident edge radius per node
    incident_radii = [[] for _ in range(N)]
    for u, v, data in G.edges(data=True):
        r = float(data["radius"])
        incident_radii[id_to_idx[u]].append(r)
        incident_radii[id_to_idx[v]].append(r)

    mean_radius = np.array([
        np.mean(rs) if len(rs) > 0 else 0.0
        for rs in incident_radii
    ], dtype=float)

    # node feature matrix X: (N, 5)
    X_np = np.stack([degrees, z_norm, y_norm, x_norm, mean_radius], axis=1)
    X = torch.tensor(X_np, dtype=torch.float32)

    # ---- edges ----
    edges = list(G.edges())
    E = len(edges)

    # edge_index: (2, E)
    edge_idx_np = np.array([[id_to_idx[u], id_to_idx[v]] for (u, v) in edges], dtype=np.int64)
    edge_index = torch.tensor(edge_idx_np.T, dtype=torch.long)

    # edge features
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

        # ground truth label; 1=artery, 0=vein, -1 if missing
        edge_labels.append(int(data.get("label", -1)))

    edge_attr = torch.tensor(np.array(edge_feats, dtype=float), dtype=torch.float32)   # (E, 8)
    y = torch.tensor(np.array(edge_labels, dtype=int), dtype=torch.long)              # (E,)

    return X, edge_index, edge_attr, y


# In[19]:


import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeGATLayer(nn.Module):
    def __init__(self, in_dim, edge_dim, out_dim, dropout=0.0, negative_slope=0.2):
        super().__init__()

        # Linear projections for nodes and edges
        self.W_node = nn.Linear(in_dim, out_dim, bias=False)
        self.W_self = nn.Linear(in_dim, out_dim, bias=True)   # for self-term
        self.W_edge = nn.Linear(edge_dim, out_dim, bias=False)

        # Attention vector a^T [h_src || h_dst || e_ij]
        self.attn = nn.Linear(3 * out_dim, 1, bias=False)

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, edge_index, edge_attr):
        """
        X: (N, F)
        edge_index: (2, E)
        edge_attr: (E, Fe)  -- already raw or embedded edge features
        """
        N = X.size(0)
        src, dst = edge_index  # (E,), (E,)

        # Node & edge projections
        h = self.W_node(X)             # (N, out_dim)
        h_src = h[src]                 # (E, out_dim)
        h_dst = h[dst]                 # (E, out_dim)
        e_proj = self.W_edge(edge_attr)  # (E, out_dim)

        # Compute attention logits e_ij
        att_input = torch.cat([h_src, h_dst, e_proj], dim=1)  # (E, 3*out_dim)
        e_ij = self.leaky_relu(self.attn(att_input))          # (E, 1)

        # Softmax over incoming edges per dst node
        # (segment softmax using dst as segment index)
        exp_e = torch.exp(e_ij)
        denom = torch.zeros(N, 1, device=X.device)
        denom.index_add_(0, dst, exp_e)            # sum over edges going into each dst
        alpha = exp_e / (denom[dst] + 1e-16)       # (E, 1)

        alpha = self.dropout(alpha)

        # Messages: weighted source representations
        messages = h_src * alpha                    # (E, out_dim)

        # Aggregate messages into destination nodes
        agg = torch.zeros(N, messages.size(1), device=X.device)
        agg.index_add_(0, dst, messages)           # sum over neighbors

        # Combine self + neighbors
        out = self.W_self(X) + agg

        # Normalize + activation
        out = self.norm(out)
        out = F.relu(out)
        out = self.dropout(out)

        return out


# In[20]:


class EdgeGATNet(nn.Module):
    def __init__(self, node_in_dim=5, edge_in_dim=8,
                 hidden_dims=[64, 128, 128, 64], num_classes=2,
                 dropout=0.1):
        super().__init__()

        h1, h2, h3, h4 = hidden_dims

        # Project edge features once (same as before)
        self.edge_embed = nn.Sequential(
            nn.Linear(edge_in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        edge_dim = 32

        # 4 stacked GAT-style layers (edge-aware)
        self.gat1 = EdgeGATLayer(node_in_dim, edge_dim, h1, dropout=dropout)
        self.gat2 = EdgeGATLayer(h1, edge_dim, h2, dropout=dropout)
        self.gat3 = EdgeGATLayer(h2, edge_dim, h3, dropout=dropout)
        self.gat4 = EdgeGATLayer(h3, edge_dim, h4, dropout=dropout)

        # Final MLP for edge classification (same as yours)
        self.edge_mlp = nn.Sequential(
            nn.Linear(h4 * 2 + edge_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)   # 0 = vein, 1 = artery
        )

    def forward(self, X, edge_index, edge_attr):
        # Edge feature embedding
        edge_e = self.edge_embed(edge_attr)   # (E, 32)

        # GAT-style message passing
        h = self.gat1(X, edge_index, edge_e)
        h = self.gat2(h, edge_index, edge_e)
        h = self.gat3(h, edge_index, edge_e)
        h = self.gat4(h, edge_index, edge_e)

        # Edge-level classification
        src = edge_index[0]
        dst = edge_index[1]

        # concatenate node embeddings + edge embedding
        h_edge = torch.cat([h[src], h[dst], edge_e], dim=1)  # (E, 2*h4 + 32)

        logits = self.edge_mlp(h_edge)
        return logits


# In[21]:


def train_model(model, train_graphs, val_graphs, epochs, lr, save_dir, device):
    os.makedirs(save_dir, exist_ok=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_model_path = os.path.join(save_dir, "best_model.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for (X, edge_index, edge_attr, labels) in train_graphs:
            X = X.to(device)
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(X, edge_index, edge_attr)

            mask = labels >= 0
            if mask.sum() == 0:
                continue

            loss = criterion(logits[mask], labels[mask])
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            train_correct += (preds[mask] == labels[mask]).sum().item()
            train_total += mask.sum().item()

        train_acc = train_correct / max(1, train_total)

        # ----- VALIDATION -----
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for (X, edge_index, edge_attr, labels) in val_graphs:
                X = X.to(device)
                edge_index = edge_index.to(device)
                edge_attr = edge_attr.to(device)
                labels = labels.to(device)

                logits = model(X, edge_index, edge_attr)
                mask = labels >= 0
                if mask.sum() == 0:
                    continue

                loss = criterion(logits[mask], labels[mask])
                val_loss += loss.item()

                preds = logits.argmax(dim=1)
                val_correct += (preds[mask] == labels[mask]).sum().item()
                val_total += mask.sum().item()

        val_acc = val_correct / max(1, val_total)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
              f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")
    model.load_state_dict(torch.load(best_model_path))
    return model


# In[22]:


import torch.nn.functional as F

def run_epoch(model, graphs, optimizer, device, training=True):
    if training:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    total_correct = 0
    total_valid = 0
    for (X, edge_index, edge_attr, y) in graphs:
        X = X.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
        y = y.to(device)
        with torch.set_grad_enabled(training):
            logits = model(X, edge_index, edge_attr)     # (E, 2)
            mask = (y != -1)
            if mask.sum() == 0:
                continue
            logits_masked = logits[mask]
            y_masked = y[mask]
            loss = F.cross_entropy(logits_masked, y_masked)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # stats
        batch_size = mask.sum().item()
        total_loss += loss.item() * batch_size
        preds = logits_masked.argmax(dim=1)
        correct = (preds == y_masked).sum().item()
        total_correct += correct
        total_valid += batch_size
    if total_valid == 0:
        return 0.0, 0.0
    avg_loss = total_loss / total_valid
    avg_acc = total_correct / total_valid
    return avg_loss, avg_acc


# In[23]:


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


# In[24]:


#!/usr/bin/env python3
import os
import pickle
import random
import torch
from glob import glob
import nibabel as nib

GRAPH_DIR = "/projectnb/ec500kb/projects/Project_4/Graph_Creations/Graphs_from_FineTuned_VesselFM/"      # folder containing *.gpickle
GT_DIR    = "/projectnb/ec500kb/projects/Project_4/Graph_Creations/Masks_Truth/"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_graph(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_dataset(graph_dir, gt_dir):
    pred_paths = sorted(glob(os.path.join(graph_dir, "*.gpickle")))
    dataset = []
    for p in pred_paths:
        print(p)
        name = os.path.basename(p).replace("_graph.gpickle", "")
        gt_path = os.path.join(gt_dir, "case_" + name[-11:-8] + ".nii.gz")
        if not os.path.exists(gt_path):
            print(f"WARNING: Missing GT mask for {name}, skipping.")
            continue

        # Load pred graph
        with open(p, "rb") as f:
            G_pred = pickle.load(f)

        # Load GT mask
        gt_img = nib.load(gt_path)
        gt_seg = gt_img.get_fdata().astype(np.int32)
        # Assign labels from GT mask
        
        G_pred = assign_true_edge_labels_from_mask(G_pred, gt_seg)
        save_dir = "/projectnb/ec500kb/projects/Project_4/Graph_Creations/Labeled_Graphs_finetune_predictions"
        filename = os.path.basename(p)
        out_path = os.path.join(save_dir, filename.replace("_graph", "_graph_labeled"))
        with open(out_path, "wb") as f:
            pickle.dump(G_pred, f)
        print(f"Saved labeled graph → {out_path}")

        # Convert graph to tensors
        X, edge_index, edge_attr, y = graph_to_tensors(G_pred)
        dataset.append((X, edge_index, edge_attr, y))

    return dataset

# -------------------------------------------------------------
# TRAIN / VAL / TEST SPLIT
# -------------------------------------------------------------
def split_dataset(dataset, train_ratio, val_ratio, test_ratio):
    assert abs(train_ratio + val_ratio + test_ratio - 1) < 1e-6

    random.shuffle(dataset)
    N = len(dataset)

    N_train = int(N * train_ratio)
    N_val   = int(N * val_ratio)

    train = dataset[:N_train]
    val   = dataset[N_train:N_train+N_val]
    test  = dataset[N_train+N_val:]

    return train, val, test

# -------------------------------------------------------------
# MAIN TRAINING SCRIPT
# -------------------------------------------------------------
def main():

    print("\n=== Loading Graph Dataset ===")
    dataset = load_dataset(GRAPH_DIR, GT_DIR)
    print(f"Loaded {len(dataset)} graphs.")

    print("\n=== Splitting Dataset ===")
    train_set, val_set, test_set = split_dataset(
        dataset,
        TRAIN_RATIO,
        VAL_RATIO,
        TEST_RATIO
    )

    print(f"Train: {len(train_set)}  Val: {len(val_set)}  Test: {len(test_set)}")

    # ---------------------------------------------------------
    # Initialize Model
    # ---------------------------------------------------------
    model = EdgeGATNet(
        node_in_dim=5,
        edge_in_dim=8,
        hidden_dims=[64, 128, 128, 64],   # or customize if you want
        num_classes=2                     # 0=vein, 1=artery
    ).to(DEVICE)


    # ---------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------
    print("\n=== Training Model ===\n")
    trained_model = train_model(
        model,
        train_graphs=train_set,
        val_graphs=val_set,
        epochs=200,
        lr=1e-3,
        save_dir="checkpoints",
        device=DEVICE)
    torch.save(trained_model, "full_model.pth")
    # ---------------------------------------------------------
    # TEST
    # ---------------------------------------------------------
    print("\n=== Testing ===")
    criterion = torch.nn.CrossEntropyLoss()

    trained_model.eval()
    total = 0
    correct = 0
    test_loss = 0

    with torch.no_grad():
        for (X, edge_index, edge_attr, labels) in test_set:

            X = X.to(DEVICE)
            edge_index = edge_index.to(DEVICE)
            edge_attr = edge_attr.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = trained_model(X, edge_index, edge_attr)
            mask = labels >= 0
            if mask.sum() == 0:
                continue

            loss = criterion(logits[mask], labels[mask])
            test_loss += float(loss)

            preds = logits.argmax(dim=1)
            correct += int((preds[mask] == labels[mask]).sum())
            total += int(mask.sum())

    test_acc = correct / max(1, total)
    print(f"Test Loss: {test_loss:.4f}  |  Test Accuracy: {test_acc:.4f}")

    print("\n=== DONE ===\n")


if __name__ == "__main__":
    main()


# In[ ]:




