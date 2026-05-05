"""Assemble the full input volume + run inference for a (lat, lon, time) request."""

from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass

import numpy as np

from ..config import settings
from ..inference.preprocessor import build_input_volume
from ..inference.postprocessor import denormalize_residual, extract_level
from . import era5, landcover, terrain


@dataclass
class DomainResult:
    terrain: np.ndarray          # (ny, nx)
    bounds: tuple[float, float, float, float]   # (south, west, north, east)
    u: np.ndarray                # (ny, nx, nz) absolute
    v: np.ndarray
    w: np.ndarray
    T: np.ndarray
    q: np.ndarray
    z_levels: np.ndarray         # (nz,) AGL
    era5_profile: dict[str, np.ndarray]


def build_and_infer(lat: float, lon: float, when: datetime, engine) -> DomainResult:
    """End-to-end: fetch data, normalize, infer, denormalize.

    `engine` is an FNOEngine instance (pre-loaded). If None, raises.
    """
    if engine is None:
        raise RuntimeError("inference engine is not loaded")

    # 1. Terrain
    terr, bounds = terrain.fetch_terrain(lat, lon, settings.domain_km,
                                         settings.grid_ny, settings.grid_nx)

    # 2. Roughness
    z0 = landcover.fetch_z0(lat, lon, settings.grid_ny, settings.grid_nx)

    # 3. ERA5 profile
    prof = era5.fetch_era5_profile(lat, lon, when)

    # 4. Build 7ch input
    x = build_input_volume(
        terrain=terr, z0=z0,
        era5_u=prof["u"], era5_v=prof["v"], era5_T=prof["T"],
        era5_q=prof["q"], era5_k=prof["k"],
    )

    # 5. Inference → normalized residual
    pred = engine.predict(x)

    # 6. Denormalize to absolute fields
    fields = denormalize_residual(
        pred, era5_u=prof["u"], era5_v=prof["v"], era5_T=prof["T"], era5_q=prof["q"],
    )

    return DomainResult(
        terrain=terr, bounds=bounds,
        u=fields["u"], v=fields["v"], w=fields["w"],
        T=fields["T"], q=fields["q"],
        z_levels=prof["z_agl"],
        era5_profile=prof,
    )
