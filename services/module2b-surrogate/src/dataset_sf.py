"""
dataset_sf.py — PyTorch datasets for surrogate training (unstructured + regular grid).

Two dataset classes:
  - SFUnstructuredDataset: for MLP and GNN (cell-level features)
  - SFGridDataset: for U-Net 3D (regular grid tensors, 128×128×32)

Grid Zarr schema (from export_sf_dataset.py):
    input/terrain        (ny, nx)        — elevation [m]
    input/z0             (ny, nx)        — roughness [m]
    input/era5/{u,v,T,q,k}  (nz,)       — ERA5 profiles
    target/U             (ny, nx, nz, 3) — CFD velocity
    target/T             (ny, nx, nz)    — CFD temperature
    target/q             (ny, nx, nz)    — CFD humidity
    residual/U           (ny, nx, nz, 3) — velocity residual
    residual/T           (ny, nx, nz)    — T residual
    residual/q           (ny, nx, nz)    — q residual
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

        self.cases = [c for c in self.meta["cases"] if c["split"] == split]

    def __len__(self) -> int:
        return len(self.cases)

    def _norm(self, data: np.ndarray, var: str) -> np.ndarray:
        if not self.normalize or var not in self.norm_stats:
            return data
        s = self.norm_stats[var]
        return (data - s["mean"]) / s["std"]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        import zarr

        case = self.cases[idx]
        case_id = case["case_id"]
        store = zarr.open_group(
            str(self.data_dir / case_id / "unstructured.zarr"), mode="r"
        )

        x = self._norm(np.array(store["x"][:], dtype=np.float32), "x")
        y = self._norm(np.array(store["y"][:], dtype=np.float32), "y")
        z_agl = self._norm(np.array(store["z_agl"][:], dtype=np.float32), "z_agl")
        elev = self._norm(np.array(store["elev"][:], dtype=np.float32), "elev")
        slope = np.zeros_like(elev)  # placeholder

        node_features = np.stack([x, y, z_agl, elev, slope], axis=-1)
        U = np.array(store["U"][:], dtype=np.float32)

        u_hub = float(case.get("u_hub", 5.0))
        wind_dir = float(case.get("wind_dir", 270.0))
        global_features = np.array(
            [u_hub / 20.0, np.sin(np.radians(wind_dir)), np.cos(np.radians(wind_dir))],
            dtype=np.float32,
        )

        return {
            "node_features": torch.from_numpy(node_features),
            "global_features": torch.from_numpy(global_features),
            "target": torch.from_numpy(U),
            "case_id": case_id,
        }


class SFGridDataset(Dataset):
    """Dataset for U-Net 3D: regular grid tensors with ERA5 residual learning.

    Each sample assembles a 3D input volume from compact Zarr data:
    - 2D terrain fields (broadcast across z)
    - 1D ERA5 profiles (broadcast across x, y)

    Supports two input variants:
    - variant="volume": 7 channels — terrain(x,y) + z0(x,y) + ERA5 profiles(z)
    - variant="terrain_only": 2 channels — terrain(x,y) + z0(x,y) (for FiLM)

    Target: residual (CFD - ERA5) if available, else absolute CFD fields.
    """

    # Normalization constants (updated after training set statistics)
    TERRAIN_SCALE = 500.0  # elevation [m] — typical range 0-1000m
    Z0_SCALE = 1.0  # roughness [m] — max 1.0
    WIND_SCALE = 20.0  # velocity [m/s]
    T_SCALE = 30.0  # temperature deviation [K]
    Q_SCALE = 0.01  # humidity [kg/kg]

    def __init__(
        self,
        data_dir: str | Path,
        dataset_yaml: str | Path,
        split: str = "train",
        variant: str = "volume",
        use_residual: bool = True,
    ):
        """
        Parameters
        ----------
        data_dir : path to exported cases (each subdir has grid.zarr/)
        dataset_yaml : path to dataset.yaml with case metadata
        split : "train", "val", or "test"
        variant : "volume" (7ch: terrain+z0+ERA5) or "terrain_only" (2ch)
        use_residual : if True, target = residual (CFD-ERA5); else absolute CFD
        """
        self.data_dir = Path(data_dir)
        with open(dataset_yaml) as f:
            self.meta = yaml.safe_load(f)

        self.cases = [c for c in self.meta["cases"] if c["split"] == split]
        self.variant = variant
        self.use_residual = use_residual

    def __len__(self) -> int:
        return len(self.cases)

    @property
    def n_input_channels(self) -> int:
        return 7 if self.variant == "volume" else 2

    @property
    def n_output_channels(self) -> int:
        return 5  # u, v, w, T, q

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        import zarr

        case = self.cases[idx]
        case_id = case["case_id"]
        store = zarr.open_group(str(self.data_dir / case_id / "grid.zarr"), mode="r")

        ny = int(store.attrs["ny"])
        nx = int(store.attrs["nx"])
        nz = int(store.attrs["nz"])

        # ── 2D terrain fields ──
        terrain = np.array(store["input/terrain"][:], dtype=np.float32)  # (ny, nx)
        z0 = np.array(store["input/z0"][:], dtype=np.float32)  # (ny, nx)

        # Normalize
        terrain_norm = terrain / self.TERRAIN_SCALE
        z0_norm = z0 / self.Z0_SCALE

        # ── Build input channels ──
        if self.variant == "volume":
            # 7 channels: terrain, z0, era5_u, era5_v, era5_T, era5_q, era5_k
            channels = []

            # Terrain (2D → 3D)
            channels.append(np.broadcast_to(terrain_norm[:, :, None], (ny, nx, nz)))
            channels.append(np.broadcast_to(z0_norm[:, :, None], (ny, nx, nz)))

            # ERA5 profiles (1D → 3D)
            era5_grp = store["input/era5"]
            for var, scale in [
                ("u", self.WIND_SCALE),
                ("v", self.WIND_SCALE),
                ("T", self.T_SCALE),
                ("q", self.Q_SCALE),
                ("k", 1.0),
            ]:
                if var in era5_grp:
                    profile = np.array(era5_grp[var][:], dtype=np.float32) / scale
                else:
                    profile = np.zeros(nz, dtype=np.float32)
                channels.append(
                    np.broadcast_to(profile[None, None, :], (ny, nx, nz))
                )

            # Stack: (7, ny, nx, nz)
            input_tensor = np.stack(channels, axis=0).copy()

        else:
            # 2 channels: terrain + z0
            channels = [
                np.broadcast_to(terrain_norm[:, :, None], (ny, nx, nz)),
                np.broadcast_to(z0_norm[:, :, None], (ny, nx, nz)),
            ]
            input_tensor = np.stack(channels, axis=0).copy()

        # ── Target ──
        if self.use_residual and "residual" in store:
            U_target = np.array(store["residual/U"][:], dtype=np.float32)  # (ny,nx,nz,3)
            T_target = (
                np.array(store["residual/T"][:], dtype=np.float32)
                if "residual/T" in store
                else np.zeros((ny, nx, nz), dtype=np.float32)
            )
            q_target = (
                np.array(store["residual/q"][:], dtype=np.float32)
                if "residual/q" in store
                else np.zeros((ny, nx, nz), dtype=np.float32)
            )
        else:
            U_target = np.array(store["target/U"][:], dtype=np.float32)
            T_target = (
                np.array(store["target/T"][:], dtype=np.float32)
                if "target/T" in store
                else np.zeros((ny, nx, nz), dtype=np.float32)
            )
            q_target = (
                np.array(store["target/q"][:], dtype=np.float32)
                if "target/q" in store
                else np.zeros((ny, nx, nz), dtype=np.float32)
            )

        # Normalize target velocities
        U_target /= self.WIND_SCALE
        T_target /= self.T_SCALE
        q_target /= self.Q_SCALE

        # Stack target: (5, ny, nx, nz) — u, v, w, T, q
        target_tensor = np.stack(
            [
                U_target[:, :, :, 0],  # u
                U_target[:, :, :, 1],  # v
                U_target[:, :, :, 2],  # w
                T_target,
                q_target,
            ],
            axis=0,
        ).copy()

        # ── ERA5 profiles for FiLM (variant="terrain_only") ──
        era5_profile = None
        if self.variant == "terrain_only" and "input/era5" in store:
            era5_grp = store["input/era5"]
            profiles = []
            for var in ["u", "v", "T", "q", "k"]:
                if var in era5_grp:
                    profiles.append(np.array(era5_grp[var][:], dtype=np.float32))
                else:
                    profiles.append(np.zeros(nz, dtype=np.float32))
            era5_profile = np.stack(profiles, axis=0)  # (5, nz)

        result = {
            "input": torch.from_numpy(input_tensor),  # (C_in, ny, nx, nz)
            "target": torch.from_numpy(target_tensor),  # (5, ny, nx, nz)
            "case_id": case_id,
        }

        if era5_profile is not None:
            result["era5_profile"] = torch.from_numpy(era5_profile)  # (5, nz)

        return result
