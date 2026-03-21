"""
dataset_sf.py — PyTorch datasets for SF PoC (unstructured + regular grid).

Two dataset classes:
  - SFUnstructuredDataset: for MLP and GNN (cell-level features)
  - SFGridDataset: for U-Net 3D (regular grid tensors)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset


class SFUnstructuredDataset(Dataset):
    """Dataset for MLP and GNN: per-cell features + global conditioning.

    Each sample contains all cells from one CFD case (one timestamp).

    Features (per cell, 5D):
        x_norm, y_norm, z_agl_norm, elev_norm, slope_proxy

    Global conditioning (3D):
        speed_norm, dir_sin, dir_cos

    Target (per cell, 3D):
        u, v, w velocity components
    """

    def __init__(
        self,
        data_dir: str | Path,
        dataset_yaml: str | Path,
        split: str = "train",
        normalize: bool = True,
    ):
        self.data_dir = Path(data_dir)
        with open(dataset_yaml) as f:
            self.meta = yaml.safe_load(f)

        self.norm_stats = self.meta.get("norm_stats", {})
        self.normalize = normalize

        # Filter cases for this split
        self.cases = [
            c for c in self.meta["cases"]
            if c["split"] == split
        ]

    def __len__(self) -> int:
        return len(self.cases)

    def _norm(self, data: np.ndarray, var: str) -> np.ndarray:
        """Normalize using precomputed mean/std."""
        if not self.normalize or var not in self.norm_stats:
            return data
        s = self.norm_stats[var]
        return (data - s["mean"]) / s["std"]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        import zarr

        case = self.cases[idx]
        case_id = case["case_id"]
        store = zarr.open_group(
            str(self.data_dir / case_id / "unstructured.zarr"), mode="r",
        )

        # Cell features
        x = self._norm(np.array(store["x"][:], dtype=np.float32), "x")
        y = self._norm(np.array(store["y"][:], dtype=np.float32), "y")
        z_agl = self._norm(np.array(store["z_agl"][:], dtype=np.float32), "z_agl")
        elev = self._norm(np.array(store["elev"][:], dtype=np.float32), "elev")

        # Slope proxy: |∇elev| ≈ difference from neighbours (approximate with z_agl variance)
        slope = np.zeros_like(elev)  # placeholder — will be computed from terrain grid

        node_features = np.stack([x, y, z_agl, elev, slope], axis=-1)  # (N, 5)

        # Velocity target
        U = np.array(store["U"][:], dtype=np.float32)  # (N, 3)

        # Global conditioning from inflow
        u_hub = float(case.get("u_hub", 5.0))
        wind_dir = float(case.get("wind_dir", 270.0))
        global_features = np.array([
            u_hub / 20.0,
            np.sin(np.radians(wind_dir)),
            np.cos(np.radians(wind_dir)),
        ], dtype=np.float32)

        return {
            "node_features": torch.from_numpy(node_features),      # (N, 5)
            "global_features": torch.from_numpy(global_features),   # (3,)
            "target": torch.from_numpy(U),                          # (N, 3)
            "case_id": case_id,
        }


class SFGridDataset(Dataset):
    """Dataset for U-Net 3D: regular grid tensors.

    Each sample is a 3D volume with terrain-following levels.

    Input channels (C_in):
        terrain (Nx, Ny)  — replicated across Nz
        inflow_u, inflow_v, inflow_k  — profiles broadcast to (Nx, Ny, Nz)

    Target channels (C_out = 3):
        u, v, w  — velocity on the regular grid (Nx, Ny, Nz, 3)
    """

    def __init__(
        self,
        data_dir: str | Path,
        dataset_yaml: str | Path,
        split: str = "train",
    ):
        self.data_dir = Path(data_dir)
        with open(dataset_yaml) as f:
            self.meta = yaml.safe_load(f)

        self.cases = [
            c for c in self.meta["cases"]
            if c["split"] == split
        ]

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        import zarr

        case = self.cases[idx]
        case_id = case["case_id"]
        store = zarr.open_group(
            str(self.data_dir / case_id / "grid.zarr"), mode="r",
        )

        # Terrain (Ny, Nx) → (1, Ny, Nx)
        terrain = np.array(store["terrain"][:], dtype=np.float32)

        # Velocity field (Ny, Nx, Nz, 3)
        U = np.array(store["U"][:], dtype=np.float32)
        ny, nx, nz, _ = U.shape

        # k field (Ny, Nx, Nz)
        k = np.array(store["k"][:], dtype=np.float32)

        # Build input tensor: terrain + inflow profile
        # Terrain: broadcast across z levels → (1, Ny, Nx, Nz)
        terrain_3d = np.broadcast_to(terrain[:, :, None], (ny, nx, nz))

        # Inflow profile from metadata
        u_hub = float(case.get("u_hub", 5.0))
        wind_dir = float(case.get("wind_dir", 270.0))
        z_levels = np.array(store["z_levels_agl"][:], dtype=np.float32)

        # Simple log-law approximation for inflow profile shape
        z_ref = 100.0
        u_profile = u_hub * np.log(np.maximum(z_levels, 1.0) / 0.05) / np.log(z_ref / 0.05)

        # Inflow u, v components based on wind direction
        dir_rad = np.radians(270.0 - wind_dir)  # met convention to math
        inflow_u = u_profile * np.cos(dir_rad)
        inflow_v = u_profile * np.sin(dir_rad)

        # Broadcast inflow to 3D (Ny, Nx, Nz)
        inflow_u_3d = np.broadcast_to(inflow_u[None, None, :], (ny, nx, nz))
        inflow_v_3d = np.broadcast_to(inflow_v[None, None, :], (ny, nx, nz))

        # Input: (C_in=3, Ny, Nx, Nz) — terrain + inflow_u + inflow_v
        input_tensor = np.stack([terrain_3d, inflow_u_3d, inflow_v_3d], axis=0)

        # Target: (3, Ny, Nx, Nz) — u, v, w
        target_tensor = np.moveaxis(U, -1, 0)  # (3, Ny, Nx, Nz)

        return {
            "input": torch.from_numpy(input_tensor.copy()),    # (C_in, Ny, Nx, Nz)
            "target": torch.from_numpy(target_tensor.copy()),  # (3, Ny, Nx, Nz)
            "case_id": case_id,
        }
