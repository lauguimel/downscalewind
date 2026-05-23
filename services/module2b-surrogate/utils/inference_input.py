"""
inference_input.py — Build a v2 grid.zarr/input from arbitrary (lat, lon, timestamp).

Used by `extract_v2_input_at_coords.py` to feed the trained v2 surrogate at any
location/time pairing. Mirrors the on-disk schema produced by
`services/module2a-cfd/export_to_grid_zarr_v2.py` so that the existing
`WindV2DatasetViT` loader can consume the result without modification.

Grid (TBM inner block):
    NI×NJ×NK = 180×180×40 voxels, 6 km × 6 km × ~2.5 km, DX = 33.333 m,
    z(i,j,k) = terrain(i,j) + agl_levels[k] (terrain-following).

Note: only `input/` and `coords/` are written. No `target/` (this is for
inference, not validation against OF).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ─── Grid constants (must match export_to_grid_zarr_v2.NI/NJ/NK) ────────────
NI, NJ, NK = 180, 180, 40
HALF_EXTENT_M = 3000.0
DX = 2 * HALF_EXTENT_M / NI  # 33.333 m
Z_TOP = 2500.0               # m, TBM inner block top
EXPANSION = 15.0             # vertical grading factor (last/first cell height ratio)

# ESA WorldCover 2021 → aerodynamic z0 [m] (matches generate_z0_field.py)
WC_Z0_LOOKUP: dict[int, float] = {
    10: 0.5,     20: 0.10,   30: 0.03,   40: 0.05,   50: 1.0,
    60: 0.005,   70: 0.001,  80: 0.0002, 90: 0.10,   95: 0.5,
    100: 0.01,
}
WC_Z0_DEFAULT = 0.05


# ─── AGL level construction ─────────────────────────────────────────────────

def build_agl_levels(
    z_top: float = Z_TOP, n_levels: int = NK, expansion: float = EXPANSION
) -> np.ndarray:
    """Geometric grading: 40 cells from 0 → z_top with last/first ratio = expansion.

    Returns the cell-CENTER AGL values (length n_levels).
    """
    r = expansion ** (1.0 / max(1, n_levels - 1))
    if abs(r - 1.0) < 1e-9:
        widths = np.full(n_levels, z_top / n_levels, dtype=np.float64)
    else:
        first = z_top * (r - 1.0) / (r ** n_levels - 1.0)
        widths = first * np.array([r ** k for k in range(n_levels)], dtype=np.float64)
    edges = np.concatenate([[0.0], np.cumsum(widths)])
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres.astype(np.float32)


# ─── DEM (Copernicus GLO-30) crop + resample ────────────────────────────────

def extract_terrain_from_dem(
    dem_tif: Path, lat: float, lon: float,
    half_extent_m: float = HALF_EXTENT_M, ni: int = NI, nj: int = NJ,
) -> np.ndarray:
    """Crop a 6 km×6 km terrain patch from a DEM raster and resample to (NI, NJ).

    Uses bilinear resampling. Output shape (NI, NJ) float32 [m].
    Convention: i = E-W (lon-aligned), j = N-S (lat-aligned). Matches
    export_to_grid_zarr_v2 (x increases with longitude eastward, y with latitude
    northward).
    """
    import rasterio
    from rasterio.warp import Resampling, calculate_default_transform, reproject
    from pyproj import Transformer

    # Local metric frame: pick UTM zone for the site, project box corners,
    # then sample the DEM via reproject to (NI, NJ) regular grid.
    utm_zone = int(math.floor((lon + 180) / 6) % 60 + 1)
    epsg_utm = 32600 + utm_zone if lat >= 0 else 32700 + utm_zone
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_utm}", always_xy=True)
    x0, y0 = transformer.transform(lon, lat)
    dst_x = np.linspace(x0 - half_extent_m, x0 + half_extent_m, ni + 1)
    dst_y = np.linspace(y0 - half_extent_m, y0 + half_extent_m, nj + 1)
    # Cell centres (length NI, NJ)
    cx = 0.5 * (dst_x[:-1] + dst_x[1:])
    cy = 0.5 * (dst_y[:-1] + dst_y[1:])

    # Build a destination affine grid: pixel (col, row) = (i, NJ-1-j) so that
    # row 0 is north, mirroring rasterio convention. We then flip back to (i,j)
    # convention used by OF.
    from rasterio.transform import Affine
    dst_transform = Affine.translation(dst_x[0], dst_y[-1]) * Affine.scale(
        (dst_x[-1] - dst_x[0]) / ni, -(dst_y[-1] - dst_y[0]) / nj
    )
    dst = np.empty((nj, ni), dtype=np.float32)

    with rasterio.open(dem_tif) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=f"EPSG:{epsg_utm}",
            resampling=Resampling.bilinear,
        )
    # `dst[row=0]` is northernmost, OF convention has y increasing northward
    # ⇒ flip the row axis so that `terrain[i, j]` with j=0 → south.
    terrain = np.flipud(dst).T.astype(np.float32)  # (ni, nj)
    return terrain


def build_native_z(terrain: np.ndarray) -> np.ndarray:
    """Build coords/z (NI, NJ, NK) = terrain[i,j] + agl_levels[k]."""
    agl = build_agl_levels()
    return (terrain[:, :, None] + agl[None, None, :]).astype(np.float32)


# ─── WorldCover → z0_eff ────────────────────────────────────────────────────

def compute_z0_eff_from_wc(
    wc_tif: Path, lat: float, lon: float,
    patch_radius_m: float = 3000.0,
) -> tuple[float, dict[int, int]]:
    """Geometric mean of z0 over a circular patch around the site.

    Returns (z0_eff [m], class_counts).
    """
    import rasterio
    from rasterio.transform import rowcol

    if not Path(wc_tif).exists():
        logger.warning("WC raster missing: %s — falling back to z0_eff=0.05", wc_tif)
        return 0.05, {}

    # Convert metric patch radius to degrees (≈ flat-earth)
    dlat = patch_radius_m / 111_000.0
    dlon = patch_radius_m / (111_000.0 * max(0.1, math.cos(math.radians(lat))))

    with rasterio.open(wc_tif) as src:
        row_c, col_c = rowcol(src.transform, lon, lat)
        # Half-window in pixels (raster res in degrees)
        half_row = max(1, int(round(dlat / abs(src.res[1]))))
        half_col = max(1, int(round(dlon / abs(src.res[0]))))
        row0, col0 = max(0, row_c - half_row), max(0, col_c - half_col)
        row1 = min(src.height, row_c + half_row + 1)
        col1 = min(src.width, col_c + half_col + 1)
        window = rasterio.windows.Window(col0, row0, col1 - col0, row1 - row0)
        patch = src.read(1, window=window).astype(np.int32).ravel()

    if patch.size == 0:
        return WC_Z0_DEFAULT, {}

    classes, counts = np.unique(patch, return_counts=True)
    counts_map = {int(c): int(n) for c, n in zip(classes, counts)}
    z0_values = np.array(
        [WC_Z0_LOOKUP.get(int(c), WC_Z0_DEFAULT) for c in patch],
        dtype=np.float64,
    )
    z0_values = z0_values[z0_values > 0]
    if z0_values.size == 0:
        return WC_Z0_DEFAULT, counts_map
    z0_eff = float(np.exp(np.mean(np.log(z0_values))))
    return z0_eff, counts_map


# ─── ERA5 3×3 extraction ────────────────────────────────────────────────────

@dataclass
class Era5Sample:
    pressure_levels: np.ndarray              # (N_p,)
    pressure_3d: dict[str, np.ndarray]       # var → (3, 3, N_p)
    surface: dict[str, np.ndarray]           # var → (3, 3)
    timestamp_iso: str
    actual_timestamp_iso: str
    delta_seconds: float


def _find_3x3_indices(lat: float, lon: float,
                      lats: np.ndarray, lons: np.ndarray) -> tuple[slice, slice]:
    i_c = int(np.argmin(np.abs(lats - lat)))
    j_c = int(np.argmin(np.abs(lons - lon)))
    i_lo = max(0, min(len(lats) - 3, i_c - 1))
    j_lo = max(0, min(len(lons) - 3, j_c - 1))
    return slice(i_lo, i_lo + 3), slice(j_lo, j_lo + 3)


def extract_era5_at_coords(
    era5_store: Path | str, lat: float, lon: float, timestamp_iso: str,
    *, max_delta_seconds: float = 3 * 3600 + 1,
) -> Era5Sample:
    """Read ERA5 3×3 pressure + surface at the nearest time to `timestamp_iso`.

    Surrogate v2 expects: pressure_3d[var] shaped (3, 3, N_p), with axis 0 = lat,
    axis 1 = lon (matches export_to_grid_zarr_v2.load_era5_at_timestamp:
    `np.transpose(pres[var][idx,:,:,:], (1,2,0))`).
    """
    import zarr
    g = zarr.open_group(str(era5_store), mode="r")
    times = np.asarray(g["coords/time"][:], dtype=np.int64)
    target_ns = np.datetime64(timestamp_iso).astype("datetime64[ns]").astype(np.int64)
    idx = int(np.argmin(np.abs(times - target_ns)))
    delta = abs(int(times[idx]) - int(target_ns)) / 1e9
    if delta > max_delta_seconds:
        raise ValueError(
            f"ERA5 store nearest time {delta/3600:.1f} h away from {timestamp_iso} "
            f"(max allowed {max_delta_seconds/3600:.1f} h)"
        )
    lats = np.asarray(g["coords/lat"][:], dtype=np.float32)
    lons = np.asarray(g["coords/lon"][:], dtype=np.float32)
    i_slc, j_slc = _find_3x3_indices(lat, lon, lats, lons)

    levels = np.asarray(g["coords/level"][:], dtype=np.float32)
    pres = g["pressure"]
    pressure_3d: dict[str, np.ndarray] = {}
    for src_var in ("u", "v", "t", "q"):
        if src_var not in pres:
            continue
        arr = np.asarray(pres[src_var][idx, :, i_slc, j_slc], dtype=np.float32)
        # arr shape: (level, 3, 3) → (3, 3, level)
        arr = np.transpose(arr, (1, 2, 0))
        out_name = "T" if src_var == "t" else src_var
        pressure_3d[out_name] = arr

    surf = g["surface"]
    surface: dict[str, np.ndarray] = {}
    for src_var in ("t2m", "d2m", "u10", "v10"):
        if src_var not in surf:
            continue
        arr = np.asarray(surf[src_var][idx, i_slc, j_slc], dtype=np.float32)
        surface[src_var] = arr

    actual_iso = str(np.array(int(times[idx])).astype("datetime64[ns]"))
    return Era5Sample(
        pressure_levels=levels,
        pressure_3d=pressure_3d,
        surface=surface,
        timestamp_iso=timestamp_iso,
        actual_timestamp_iso=actual_iso,
        delta_seconds=delta,
    )


# ─── Wind direction & inflow_meta ───────────────────────────────────────────

def estimate_wind_direction_deg(u: float, v: float) -> float:
    """Direction (meteorological convention: wind coming FROM) in degrees [0,360)."""
    return float((270.0 - math.degrees(math.atan2(v, u))) % 360.0)


# ─── Zarr writer (matches export_to_grid_zarr_v2 input schema) ─────────────

def write_input_grid_zarr(
    out_path: Path,
    *,
    site_id: str,
    lat: float, lon: float,
    terrain: np.ndarray,
    z_grid: np.ndarray,
    z0_eff: float,
    era5: Era5Sample,
    timestamp_iso: str,
    extra_meta: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a grid.zarr with only coords/ + input/ groups (no target/)."""
    import shutil
    import zarr

    def _write(grp: zarr.Group, name: str, arr: np.ndarray) -> None:
        """Create + populate an array, compatible with zarr 3.x."""
        out = grp.create_array(name, shape=arr.shape, dtype=arr.dtype)
        out[...] = arr

    out_path = Path(out_path)
    if out_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists (use --overwrite): {out_path}")
        shutil.rmtree(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g = zarr.open_group(str(out_path), mode="w")

    x_1d = (np.arange(NI) + 0.5) * DX - HALF_EXTENT_M
    y_1d = (np.arange(NJ) + 0.5) * DX - HALF_EXTENT_M
    coords = g.create_group("coords")
    _write(coords, "x", x_1d.astype(np.float32))
    _write(coords, "y", y_1d.astype(np.float32))
    _write(coords, "z", z_grid.astype(np.float32))

    inp = g.create_group("input")
    _write(inp, "terrain", terrain.astype(np.float32))
    inp.attrs["z0_eff"] = float(z0_eff)
    inp.attrs["lat"] = float(lat)
    inp.attrs["lon"] = float(lon)
    _write(inp, "era5_pressure_levels", era5.pressure_levels.astype(np.float32))

    e3d = inp.create_group("era5_3d")
    for var, arr in era5.pressure_3d.items():
        _write(e3d, var, arr.astype(np.float32))

    esrf = inp.create_group("era5_surface")
    for var, arr in era5.surface.items():
        _write(esrf, var, arr.astype(np.float32))

    meta = inp.create_group("inflow_meta")
    meta.attrs["timestamp"] = timestamp_iso
    meta.attrs["actual_era5_timestamp"] = era5.actual_timestamp_iso
    meta.attrs["era5_time_delta_s"] = era5.delta_seconds
    meta.attrs["site_id"] = site_id
    meta.attrs["site_lat"] = float(lat)
    meta.attrs["site_lon"] = float(lon)
    meta.attrs["z0_eff"] = float(z0_eff)
    if "u10" in era5.surface and "v10" in era5.surface:
        u10_c = float(era5.surface["u10"][1, 1])
        v10_c = float(era5.surface["v10"][1, 1])
        meta.attrs["u10_ms"] = u10_c
        meta.attrs["v10_ms"] = v10_c
        meta.attrs["wind_dir"] = estimate_wind_direction_deg(u10_c, v10_c)
    if "t2m" in era5.surface:
        meta.attrs["t2m_K"] = float(era5.surface["t2m"][1, 1])
    if "d2m" in era5.surface:
        meta.attrs["d2m_K"] = float(era5.surface["d2m"][1, 1])
    if extra_meta:
        for k, v in extra_meta.items():
            meta.attrs[k] = v

    g.attrs.update({
        "schema_version": "v2.0-inference",
        "site_id": site_id,
        "grid_shape": [NI, NJ, NK],
        "dx_m": DX,
        "half_extent_m": HALF_EXTENT_M,
        "source": "extract_v2_input_at_coords",
    })
    return out_path
