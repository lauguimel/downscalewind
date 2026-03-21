"""
model_gnn.py — GNN (GATv2) for wind downscaling on unstructured mesh.

Architecture:
    - k-NN graph built from cell centres (k=16)
    - Node features: (x, y, z_agl, elev, slope) + FiLM conditioning from global
    - 4 layers GATv2Conv (64 hidden, 4 heads)
    - Output: (u, v, w) per node

Requires torch_geometric.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import GATv2Conv
    from torch_geometric.data import Data
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

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: (N, F), cond: (F_cond,) or (1, F_cond)"""
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        gamma = self.gamma(cond)  # (1, F)
        beta = self.beta(cond)    # (1, F)
        return x * (1 + gamma) + beta


class GNNSpeedup(nn.Module):
    """GATv2-based GNN for wind field prediction."""

    def __init__(
        self,
        n_node_features: int = 5,
        n_global: int = 3,
        hidden: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        output_dim: int = 3,
        k_neighbours: int = 16,
    ):
        super().__init__()
        if not HAS_PYG:
            raise ImportError(
                "torch_geometric is required for GNN model: "
                "pip install torch-geometric"
            )

        self.k = k_neighbours

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_node_features, hidden),
            nn.ReLU(),
        )

        # FiLM conditioning
        self.film = FiLMLayer(hidden, n_global)

        # GATv2 layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(n_layers):
            in_dim = hidden if i == 0 else hidden * n_heads
            self.convs.append(
                GATv2Conv(in_dim, hidden, heads=n_heads, concat=True)
            )
            self.norms.append(nn.LayerNorm(hidden * n_heads))

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
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        node_features : (N, D_node)
        global_features : (D_global,)
        edge_index : (2, E) — optional, built from pos if not provided
        pos : (N, 3) — xyz positions for k-NN graph construction

        Returns
        -------
        (N, 3) — predicted u, v, w
        """
        # Build graph if not provided
        if edge_index is None:
            if pos is None:
                raise ValueError("Either edge_index or pos must be provided")
            edge_index = knn_graph(pos, k=self.k, loop=False)

        # Input projection + FiLM conditioning
        x = self.input_proj(node_features)
        x = self.film(x, global_features)

        # GATv2 layers with residual connections
        for conv, norm in zip(self.convs, self.norms):
            x_prev = x
            x = conv(x, edge_index)
            x = norm(x)
            x = torch.relu(x)
            # Residual (project x_prev if dimensions differ)
            if x_prev.shape[-1] == x.shape[-1]:
                x = x + x_prev

        return self.output_proj(x)
