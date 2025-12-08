#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

