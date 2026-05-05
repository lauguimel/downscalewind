"""Roughness length z0 for the domain.

v1: constant fallback (0.1 m, grassland). Proper WorldCover integration can
reuse services/data-ingestion/ingest_landcover.py later.
"""

from __future__ import annotations

import numpy as np


def fetch_z0(
    lat_center: float,
    lon_center: float,
    ny: int,
    nx: int,
    fallback: float = 0.1,
) -> np.ndarray:
    """Return (ny, nx) roughness length in meters. Currently returns a constant."""
    return np.full((ny, nx), fallback, dtype=np.float32)
