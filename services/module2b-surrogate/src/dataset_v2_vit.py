"""
dataset_v2_vit.py — ViT-format wrapper around v2 grid.zarr cases.

Returns per __getitem__:
    terrain  (C, 180, 180)         — [terrain_norm, optional slopes, z0_broadcast_norm]
    era5     (era5_input_dim,)     — flatten ERA5 3D (4×3×3×N_p) + surface (4×3×3)
                                      + pressure levels (N_p) + lat + z0_eff
    geo      (2, 180, 180, 40)     — optional [z_norm, agl_norm]
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

from .dataset_v2 import (
    DEFAULT_NORM,
    NI,
    NJ,
    NK,
    build_era5_baseline_tensor,
    parse_agl_levels,
    resample_volume_to_agl_levels,
)

logger = logging.getLogger(__name__)


def compute_era5_dim(n_pressure: int = 10) -> int:
    """ERA5 flat dim: 4 vars × 3 × 3 × N_p + 4 vars × 3 × 3 + N_p + lat + z0."""
    return 4 * 3 * 3 * n_pressure + 4 * 3 * 3 + n_pressure + 2


class WindV2DatasetViT(Dataset):
    def __init__(self, data_dir, splits_yaml, split="train", *, norm=None,
                 n_pressure=10, include_slopes=False, return_geo=False,
                 use_residual=False, return_weight=False,
                 residual_baseline_mode="pressure_index",
                 agl_weight_alpha=0.0, agl_weight_height=300.0,
                 target_agl_levels=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.norm = {**DEFAULT_NORM, **(norm or {})}
        self.n_pressure = n_pressure
        self.include_slopes = include_slopes
        self.return_geo = return_geo
        self.use_residual = use_residual
        self.residual_baseline_mode = residual_baseline_mode
        self.return_weight = return_weight
        self.agl_weight_alpha = agl_weight_alpha
        self.agl_weight_height = agl_weight_height
        self.target_agl_levels = parse_agl_levels(target_agl_levels)

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

        # --- terrain (C, 180, 180) ---
        terrain_raw = np.asarray(g["input/terrain"][:], dtype=np.float32)
        terrain = terrain_raw / n["terrain_scale"]
        z0_eff = float(g["input"].attrs.get("z0_eff", 0.0)) / n["z0_scale"]
        z0_map = np.full((NI, NJ), z0_eff, dtype=np.float32)
        terrain_parts = [terrain.astype(np.float32)]
        if self.include_slopes:
            slope_y, slope_x = np.gradient(terrain_raw, 33.333, 33.333)
            terrain_parts.extend([slope_x.astype(np.float32),
                                  slope_y.astype(np.float32)])
        terrain_parts.append(z0_map)
        terrain_2d = np.stack(terrain_parts, axis=0)

        native_z = np.asarray(g["coords/z"][:], dtype=np.float32)
        native_agl = native_z - terrain_raw[:, :, None]
        if self.target_agl_levels is None:
            z = native_z
            agl = native_agl
        else:
            agl = np.broadcast_to(
                self.target_agl_levels[None, None, :],
                (NI, NJ, self.target_agl_levels.size),
            ).copy()
            z = terrain_raw[:, :, None] + agl
        geo = np.stack([
            z / n["z_scale"],
            agl / n["agl_scale"],
        ], axis=0).astype(np.float32)

        # --- era5 (era5_dim,) ---
        plev = np.asarray(g["input/era5_pressure_levels"][:], dtype=np.float32)
        flat_parts = []
        for var, scale, offset in [
            ("u", n["era5_u_scale"], n["era5_u_offset"]),
            ("v", n["era5_v_scale"], n["era5_v_offset"]),
            ("T", n["era5_T_scale"], n["era5_T_offset"]),
            ("q", n["era5_q_scale"], n["era5_q_offset"]),
        ]:
            arr = np.asarray(g[f"input/era5_3d/{var}"][:], dtype=np.float32)  # (3,3,N_p)
            flat_parts.append(((arr - offset) / scale).ravel())
        for var, scale, offset in [
            ("t2m", n["t2m_scale"], n["t2m_offset"]),
            ("d2m", n["d2m_scale"], n["d2m_offset"]),
            ("u10", n["u10_scale"], n["u10_offset"]),
            ("v10", n["v10_scale"], n["v10_offset"]),
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
            (U[..., 0] - n["U_x_offset"]) / n["U_uv_scale"],
            (U[..., 1] - n["U_y_offset"]) / n["U_uv_scale"],
            (U[..., 2] - n["U_z_offset"]) / n["U_w_scale"],
            (T - n["T_offset"]) / n["T_scale"],
            (q - n["q_offset"]) / n["q_scale"],
        ], axis=0).astype(np.float32)
        if self.target_agl_levels is not None:
            target = resample_volume_to_agl_levels(
                target, native_agl, self.target_agl_levels
            )
        if self.use_residual:
            target = target - self._build_era5_baseline_tensor(g)

        out = [torch.from_numpy(terrain_2d), torch.from_numpy(era5_flat)]
        if self.return_geo:
            out.append(torch.from_numpy(geo))
        out.append(torch.from_numpy(target))
        if self.return_weight:
            out.append(torch.from_numpy(self._build_loss_weight(agl)))
        out.append(case_dir.name)
        return tuple(out)

    def _build_era5_baseline_tensor(self, store) -> np.ndarray:
        nz = NK if self.target_agl_levels is None else int(self.target_agl_levels.size)
        return build_era5_baseline_tensor(
            store,
            self.norm,
            nz,
            mode=self.residual_baseline_mode,
        )

    def _build_loss_weight(self, agl: np.ndarray) -> np.ndarray:
        agl = np.maximum(agl, 0.0)
        alpha = float(self.agl_weight_alpha)
        height = max(float(self.agl_weight_height), 1.0)
        weight = 1.0 + alpha * np.exp(-agl / height)
        return weight[None, ...].astype(np.float32)
