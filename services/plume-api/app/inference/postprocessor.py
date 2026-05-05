"""Denormalize model output and extract derived quantities (60m wind, FWI)."""

from __future__ import annotations

import numpy as np

from ..config import settings


def denormalize_residual(
    pred: np.ndarray,      # (5, ny, nx, nz) normalized residual
    era5_u: np.ndarray,    # (nz,) absolute ERA5 u
    era5_v: np.ndarray,
    era5_T: np.ndarray,
    era5_q: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return absolute fields: u, v, w (m/s), T (K), q (kg/kg), all (ny,nx,nz)."""
    du = pred[0] * settings.wind_scale
    dv = pred[1] * settings.wind_scale
    dw = pred[2] * settings.wind_scale   # ERA5 has no w → absolute
    dT = pred[3] * settings.t_scale
    dq = pred[4] * settings.q_scale

    u = du + era5_u[None, None, :]
    v = dv + era5_v[None, None, :]
    w = dw
    T = dT + era5_T[None, None, :]
    q = dq + era5_q[None, None, :]
    q = np.clip(q, 0.0, None)  # physical: humidity >= 0

    return {"u": u, "v": v, "w": w, "T": T, "q": q}


def extract_level(field: np.ndarray, z_levels: np.ndarray, target_z: float) -> np.ndarray:
    """Linearly interpolate (ny,nx,nz) field to a target AGL height.

    z_levels: (nz,) AGL heights in meters (log-spaced 5 → 5000 by default).
    """
    idx_hi = np.searchsorted(z_levels, target_z)
    if idx_hi == 0:
        return field[:, :, 0]
    if idx_hi >= len(z_levels):
        return field[:, :, -1]
    idx_lo = idx_hi - 1
    z_lo, z_hi = z_levels[idx_lo], z_levels[idx_hi]
    w = (target_z - z_lo) / (z_hi - z_lo)
    return (1 - w) * field[:, :, idx_lo] + w * field[:, :, idx_hi]
