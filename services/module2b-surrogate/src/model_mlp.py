"""
model_mlp.py — MLP per-node baseline for wind downscaling.

Architecture:
    node_features (5) + global_features (3) → concat → MLP → (u, v, w)

Global features (wind speed, direction sin/cos) are broadcast to every node.
This is the simplest possible architecture — no spatial structure.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPSpeedup(nn.Module):
    """Per-node MLP conditioned on global wind features."""

    def __init__(
        self,
        n_node_features: int = 5,
        n_global: int = 3,
        hidden: int = 64,
        n_layers: int = 4,
        output_dim: int = 3,
    ):
        super().__init__()
        layers = [nn.Linear(n_node_features + n_global, hidden), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
        layers.append(nn.Linear(hidden, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        node_features : (B, N, D_node) or (N, D_node)
        global_features : (B, D_global) or (D_global,)

        Returns
        -------
        (B, N, 3) or (N, 3) — predicted u, v, w
        """
        if node_features.dim() == 2:
            # Unbatched: (N, D_node) + (D_global,)
            g = global_features.unsqueeze(0).expand(node_features.shape[0], -1)
            x = torch.cat([node_features, g], dim=-1)
            return self.net(x)
        else:
            # Batched: (B, N, D_node) + (B, D_global)
            B, N, _ = node_features.shape
            g = global_features.unsqueeze(1).expand(B, N, -1)
            x = torch.cat([node_features, g], dim=-1)
            return self.net(x)
