"""
inference_batch.py — Batched helpers for surrogate v2 inference at OBS stations.

Reuses the per-case normalisation logic from
`services/validation/run_station_surrogate_inference.py` (single-sample path),
adapted to a batched pipeline driven by `infer_at_stations.py`.

Public API:
  build_features(store, norm, cfg)   → (terrain_2d, era5_flat, geo, levels)
  build_baseline(store, norm, nz, mode) → (5, NI, NJ, nz) ERA5-lifted baseline
  denorm_fields(volume, norm)         → dict[str, np.ndarray]
  k_index_at_height(levels, h_obs)    → fractional index for vertical interp
"""
from __future__ import annotations

from typing import Any

import numpy as np

# Native grid constants (must match M_G6 inference_input.NI/NJ)
NI = 180
NJ = 180


# ─── Per-case normalisation (replicates WindV2DatasetViT.__getitem__) ────────

def build_features(
    store: Any,
    norm: dict[str, float],
    cfg: dict[str, Any],
    target_agl_levels: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build (terrain_2d, era5_flat, geo, levels) for a single grid.zarr/input.

    Returns the same tensors WindV2DatasetViT.__getitem__ would produce, sans
    target. Shapes:
        terrain_2d  (C_t, 180, 180)         C_t = 4 if include_slopes else 2
        era5_flat   (era5_dim,)             era5_dim = 408 for N_p=10
        geo         (2, 180, 180, nz)
        levels      (nz,)                   AGL levels used by `geo`
    """
    terrain_raw = np.asarray(store["input/terrain"][:], dtype=np.float32)
    terrain = terrain_raw / norm["terrain_scale"]
    z0_eff = float(store["input"].attrs.get("z0_eff", 0.0)) / norm["z0_scale"]
    z0_map = np.full((NI, NJ), z0_eff, dtype=np.float32)

    parts: list[np.ndarray] = [terrain.astype(np.float32)]
    if bool(cfg.get("include_slopes", False)):
        slope_y, slope_x = np.gradient(terrain_raw, 33.333, 33.333)
        parts.extend([slope_x.astype(np.float32), slope_y.astype(np.float32)])
    parts.append(z0_map)
    terrain_2d = np.stack(parts, axis=0)

    # geo / AGL levels
    if target_agl_levels is None:
        z = np.asarray(store["coords/z"][:], dtype=np.float32)
        agl = z - terrain_raw[:, :, None]
        levels = agl[NI // 2, NJ // 2, :].astype(np.float32)
    else:
        levels = target_agl_levels.astype(np.float32)
        agl = np.broadcast_to(levels[None, None, :], (NI, NJ, levels.size)).copy()
        z = terrain_raw[:, :, None] + agl
    geo = np.stack(
        [z / norm["z_scale"], agl / norm["agl_scale"]],
        axis=0,
    ).astype(np.float32)

    # era5_flat
    plev = np.asarray(store["input/era5_pressure_levels"][:], dtype=np.float32)
    flat_parts: list[np.ndarray] = []
    for var, scale, offset in [
        ("u", norm["era5_u_scale"], norm["era5_u_offset"]),
        ("v", norm["era5_v_scale"], norm["era5_v_offset"]),
        ("T", norm["era5_T_scale"], norm["era5_T_offset"]),
        ("q", norm["era5_q_scale"], norm["era5_q_offset"]),
    ]:
        arr = np.asarray(store[f"input/era5_3d/{var}"][:], dtype=np.float32)
        flat_parts.append(((arr - offset) / scale).ravel())
    for var, scale, offset in [
        ("t2m", norm["t2m_scale"], norm["t2m_offset"]),
        ("d2m", norm["d2m_scale"], norm["d2m_offset"]),
        ("u10", norm["u10_scale"], norm["u10_offset"]),
        ("v10", norm["v10_scale"], norm["v10_offset"]),
    ]:
        arr = np.asarray(store[f"input/era5_surface/{var}"][:], dtype=np.float32)
        flat_parts.append(((arr - offset) / scale).ravel())
    flat_parts.append(
        ((plev - norm["pressure_offset"]) / norm["pressure_scale"]).astype(np.float32)
    )
    lat = float(store["input"].attrs.get("lat", 0.0)) / norm["lat_scale"]
    flat_parts.append(np.array([lat, z0_eff], dtype=np.float32))
    era5_flat = np.concatenate(flat_parts).astype(np.float32)

    return terrain_2d, era5_flat, geo, levels


# ─── Denormalisation ────────────────────────────────────────────────────────

def denorm_fields(volume: np.ndarray, norm: dict[str, float]) -> dict[str, np.ndarray]:
    """volume: (5, NI, NJ, nz). Returns dict of physical fields.

    Order is (u, v, w, T, q), matching WindV2DatasetViT target stack.
    """
    return {
        "u": volume[0] * norm["U_uv_scale"] + norm["U_x_offset"],
        "v": volume[1] * norm["U_uv_scale"] + norm["U_y_offset"],
        "w": volume[2] * norm["U_w_scale"] + norm["U_z_offset"],
        "T": volume[3] * norm["T_scale"] + norm["T_offset"],
        "q": volume[4] * norm["q_scale"] + norm["q_offset"],
    }


# ─── Vertical interpolation at central column ───────────────────────────────

def value_at_height(profile: np.ndarray, levels: np.ndarray, target_agl: float) -> float:
    """Linear interp in AGL of a 1D profile at `target_agl` metres.

    `levels` must be 1D AGL (m), monotonic increasing or freely ordered.
    """
    order = np.argsort(levels)
    return float(np.interp(float(target_agl), levels[order], profile[order]))
