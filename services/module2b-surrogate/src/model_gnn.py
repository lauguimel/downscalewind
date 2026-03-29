"""
model_gnn.py — GNN (GATv2) for wind/T/q downscaling on unstructured mesh.

Architecture:
    - Graph from mesh topology (preferred) or k-NN fallback (k=16)
    - Edge features: [dx/d, dy/d, dz/d, d, log(d)] (5D)
    - Node features: (x, y, z_agl, elev, z0) + optional ERA5 values
    - FiLM conditioning from ProfileEncoder
    - 6 layers GATv2Conv (64 hidden, 4 heads)
    - Output: (u, v, w, T, q) per node — 5D

Requires torch_geometric.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import GATv2Conv
    from torch_geometric.nn import knn_graph

    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: scale and shift features by global conditioning."""

    def __init__(self, n_features: int, n_cond: int):
        super().__init__()
        self.gamma = nn.Linear(n_cond, n_features)
        self.beta = nn.Linear(n_cond, n_features)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: (N, F), cond: (1, F_cond) or (F_cond,)"""
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        gamma = self.gamma(cond)
        beta = self.beta(cond)
        return x * (1 + gamma) + beta


class EdgeEncoder(nn.Module):
    """Encode raw edge attributes into edge features for GATv2."""

    def __init__(self, in_dim: int = 5, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
        )

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.net(edge_attr)


def compute_edge_attr(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Compute edge attributes: [dx/d, dy/d, dz/d, d, log(d)].

    Parameters
    ----------
    pos : (N, 3) — node positions (x, y, z)
    edge_index : (2, E) — edge indices

    Returns
    -------
    (E, 5) — edge features
    """
    src, dst = edge_index
    diff = pos[dst] - pos[src]  # (E, 3)
    dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # (E, 1)
    direction = diff / dist  # (E, 3)
    log_dist = torch.log(dist)
    return torch.cat([direction, dist, log_dist], dim=-1)  # (E, 5)


class GNNSpeedup(nn.Module):
    """GATv2-based GNN with edge features and FiLM conditioning."""

    def __init__(
        self,
        n_node_features: int = 5,
        n_global: int = 128,
        hidden: int = 64,
        n_heads: int = 4,
        n_layers: int = 6,
        output_dim: int = 5,
        k_neighbours: int = 16,
        edge_dim: int = 32,
    ):
        super().__init__()
        if not HAS_PYG:
            raise ImportError(
                "torch_geometric is required for GNN model: "
                "pip install torch-geometric"
            )

        self.k = k_neighbours
        self.n_layers = n_layers

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_node_features, hidden),
            nn.ReLU(),
        )

        # FiLM conditioning
        self.film = FiLMLayer(hidden, n_global)

        # Edge encoder
        self.edge_encoder = EdgeEncoder(in_dim=5, hidden=edge_dim)

        # GATv2 layers with edge features
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(n_layers):
            in_dim = hidden if i == 0 else hidden * n_heads
            self.convs.append(
                GATv2Conv(
                    in_dim, hidden, heads=n_heads, concat=True,
                    edge_dim=edge_dim,
                )
            )
            self.norms.append(nn.LayerNorm(hidden * n_heads))

        # Residual projections for dimension mismatch
        self.res_projs = nn.ModuleList()
        for i in range(n_layers):
            in_dim = hidden if i == 0 else hidden * n_heads
            out_dim = hidden * n_heads
            if in_dim != out_dim:
                self.res_projs.append(nn.Linear(in_dim, out_dim))
            else:
                self.res_projs.append(nn.Identity())

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden * n_heads, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        node_features : (N, D_node)
        global_features : (D_global,) or (1, D_global)
        edge_index : (2, E) — from mesh topology or built via k-NN
        pos : (N, 3) — for k-NN graph construction and edge attr computation
        edge_attr : (E, 5) — precomputed edge attributes (optional)

        Returns
        -------
        (N, 5) — predicted u, v, w, T, q
        """
        # Build graph if not provided
        if edge_index is None:
            if pos is None:
                raise ValueError("Either edge_index or pos must be provided")
            edge_index = knn_graph(pos, k=self.k, loop=False)

        # Compute edge attributes if not provided
        if edge_attr is None:
            if pos is None:
                raise ValueError("pos required to compute edge attributes")
            edge_attr = compute_edge_attr(pos, edge_index)

        # Encode edge features
        edge_feat = self.edge_encoder(edge_attr)

        # Input projection + FiLM conditioning
        x = self.input_proj(node_features)
        x = self.film(x, global_features)

        # GATv2 layers with residual connections
        for conv, norm, res_proj in zip(self.convs, self.norms, self.res_projs):
            x_prev = res_proj(x)
            x = conv(x, edge_index, edge_attr=edge_feat)
            x = norm(x)
            x = torch.relu(x)
            x = x + x_prev

        return self.output_proj(x)
