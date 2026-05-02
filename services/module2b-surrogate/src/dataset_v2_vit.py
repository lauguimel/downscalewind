"""
dataset_v2_vit.py — ViT-format wrapper around v2 grid.zarr cases.

Returns per __getitem__:
    terrain  (2, 180, 180)         — [terrain_norm, z0_broadcast_norm]
    era5     (era5_input_dim,)     — flatten ERA5 3D (4×3×3×N_p) + surface (4×3×3)
                                      + pressure levels (N_p) + lat + z0_eff
    target   (5, 180, 180, 40)     — same as WindV2Dataset (u, v, w, T, q normalised)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import yaml
import zarr
from torch.utils.data import Dataset

from .dataset_v2 import DEFAULT_NORM, NI, NJ, NK

logger = logging.getLogger(__name__)


def compute_era5_dim(n_pressure: int = 10) -> int:
    """ERA5 flat dim: 4 vars × 3 × 3 × N_p + 4 vars × 3 × 3 + N_p + lat + z0."""
    return 4 * 3 * 3 * n_pressure + 4 * 3 * 3 + n_pressure + 2


class WindV2DatasetViT(Dataset):
    def __init__(self, data_dir, splits_yaml, split="train", *, norm=None,
                 n_pressure=10):
        self.data_dir = Path(data_dir)
        self.split = split
        self.norm = {**DEFAULT_NORM, **(norm or {})}
        self.n_pressure = n_pressure

        with open(splits_yaml) as f:
            splits = yaml.safe_load(f)
        site_ids = splits[split]

        cases, n_skipped = [], 0
        for sid in site_ids:
            for c in sorted(self.data_dir.glob(f"{sid}_case_ts*")):
                z = c / "grid.zarr"
                if not z.exists():
                    continue
                try:
                    g = zarr.open_group(str(z), mode="r")
                    if {"U", "T", "q"}.issubset(set(g["target"])):
                        cases.append(c)
                    else:
                        n_skipped += 1
                except Exception:
                    n_skipped += 1
        self.cases = cases
        logger.info("WindV2DatasetViT[%s]: %d cases (skipped %d)",
                    split, len(cases), n_skipped)

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        n = self.norm
        case_dir = self.cases[idx]
        g = zarr.open_group(str(case_dir / "grid.zarr"), mode="r")

        # --- terrain (2, 180, 180) ---
        terrain = np.asarray(g["input/terrain"][:], dtype=np.float32) / n["terrain_scale"]
        z0_eff = float(g["input"].attrs.get("z0_eff", 0.0)) / n["z0_scale"]
        z0_map = np.full((NI, NJ), z0_eff, dtype=np.float32)
        terrain_2d = np.stack([terrain, z0_map], axis=0)  # (2, 180, 180)

        # --- era5 (era5_dim,) ---
        plev = np.asarray(g["input/era5_pressure_levels"][:], dtype=np.float32)
        flat_parts = []
        for var, scale, offset in [
            ("u", n["era5_u_scale"], 0.0),
            ("v", n["era5_v_scale"], 0.0),
            ("T", n["era5_T_scale"], n["era5_T_offset"]),
            ("q", n["era5_q_scale"], 0.0),
        ]:
            arr = np.asarray(g[f"input/era5_3d/{var}"][:], dtype=np.float32)  # (3,3,N_p)
            flat_parts.append(((arr - offset) / scale).ravel())
        for var, scale, offset in [
            ("t2m", n["t2m_scale"], n["t2m_offset"]),
            ("d2m", n["d2m_scale"], n["d2m_offset"]),
            ("u10", n["u10_scale"], 0.0),
            ("v10", n["v10_scale"], 0.0),
        ]:
            arr = np.asarray(g[f"input/era5_surface/{var}"][:], dtype=np.float32)
            flat_parts.append(((arr - offset) / scale).ravel())
        flat_parts.append(((plev - n["pressure_offset"]) / n["pressure_scale"])
                          .astype(np.float32))
        lat = float(g["input"].attrs.get("lat", 0.0)) / n["lat_scale"]
        flat_parts.append(np.array([lat, z0_eff], dtype=np.float32))
        era5_flat = np.concatenate(flat_parts).astype(np.float32)

        # --- target (5, 180, 180, 40) ---
        U = np.asarray(g["target/U"][:], dtype=np.float32)
        T = np.asarray(g["target/T"][:], dtype=np.float32)
        q = np.asarray(g["target/q"][:], dtype=np.float32)
        target = np.stack([
            U[..., 0] / n["U_uv_scale"],
            U[..., 1] / n["U_uv_scale"],
            U[..., 2] / n["U_w_scale"],
            (T - n["T_offset"]) / n["T_scale"],
            q / n["q_scale"],
        ], axis=0).astype(np.float32)

        return (torch.from_numpy(terrain_2d),
                torch.from_numpy(era5_flat),
                torch.from_numpy(target),
                case_dir.name)
