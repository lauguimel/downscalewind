"""Build the 7-channel input volume from raw terrain + z0 + ERA5 profile.

Replicates exactly the normalization used in dataset_sf.py::SFGridDataset
(variant="volume"). Any drift here will silently corrupt predictions.
"""

from __future__ import annotations

import numpy as np

from ..config import settings


def build_input_volume(
    terrain: np.ndarray,   # (ny, nx) elevation in meters
    z0: np.ndarray,        # (ny, nx) roughness in meters
    era5_u: np.ndarray,    # (nz,) u at 32 AGL levels, m/s
    era5_v: np.ndarray,    # (nz,) v, m/s
    era5_T: np.ndarray,    # (nz,) T, K
    era5_q: np.ndarray,    # (nz,) q, kg/kg
    era5_k: np.ndarray,    # (nz,) TKE, m²/s²
) -> np.ndarray:
    """Return (7, ny, nx, nz) float32 normalized input volume."""
    ny = settings.grid_ny
    nx = settings.grid_nx
    nz = settings.grid_nz
    assert terrain.shape == (ny, nx), f"terrain shape {terrain.shape}"
    assert z0.shape == (ny, nx), f"z0 shape {z0.shape}"
    for name, arr in [("u", era5_u), ("v", era5_v), ("T", era5_T), ("q", era5_q), ("k", era5_k)]:
        if arr.shape != (nz,):
            raise ValueError(f"era5_{name} shape {arr.shape}, expected ({nz},)")

    terrain_n = (terrain / settings.terrain_scale).astype(np.float32)
    z0_n = (z0 / settings.z0_scale).astype(np.float32)

    profiles = [
        (era5_u / settings.wind_scale).astype(np.float32),
        (era5_v / settings.wind_scale).astype(np.float32),
        (era5_T / settings.t_scale).astype(np.float32),
        (era5_q / settings.q_scale).astype(np.float32),
        era5_k.astype(np.float32),  # k scale = 1
    ]

    channels = [
        np.broadcast_to(terrain_n[:, :, None], (ny, nx, nz)),
        np.broadcast_to(z0_n[:, :, None], (ny, nx, nz)),
    ]
    for p in profiles:
        channels.append(np.broadcast_to(p[None, None, :], (ny, nx, nz)))

    return np.stack(channels, axis=0).copy()  # (7, ny, nx, nz)
