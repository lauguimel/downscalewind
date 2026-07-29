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

def _resolve_dem_path(dem: Path, lat: float, lon: float) -> Path:
    """Resolve DEM path: pass-through for single file, or pick the Copernicus
    DSM tile matching (lat, lon) when `dem` is a directory of tiles.

    Tile naming convention: `Copernicus_DSM_COG_10_N<NN>_00_<E|W><EEE>_00_DEM.tif`
    where N<NN> is floor(lat), and longitude uses E for >=0, W for <0 with
    abs(floor(lon)). Tiles can be the raw .tif files (the dir is searched).
    """
    p = Path(dem)
    if p.is_file():
        return p
    if not p.is_dir():
        raise FileNotFoundError(f"DEM not found: {dem}")
    # Copernicus DSM tiles are named by their LOWER-LEFT (SW) corner: the tile
    # with lon_ll=L covers [L, L+1). The correct index is therefore
    # abs(floor(lon)) for BOTH hemispheres (E/W prefix carries the sign):
    #   lon= 7.73 → floor= 7  → E007 ✓ ; lon=-7.73 → floor=-8 → abs=8 → W008 ✓ ;
    #   lon=-8.0  → floor=-8  → abs=8 → W008 ✓ .
    # The previous `floor(-lon)` form returned W007 for lon=-7.73 (data lives in
    # W008) → the box fell outside the tile → rasterio filled a flat-ZERO patch.
    lat_dir = "N" if lat >= 0 else "S"
    lat_idx = abs(int(math.floor(lat)))
    lon_dir = "E" if lon >= 0 else "W"
    lon_idx = abs(int(math.floor(lon)))
    name = f"Copernicus_DSM_COG_10_{lat_dir}{lat_idx:02d}_00_{lon_dir}{lon_idx:03d}_00_DEM.tif"
    candidate = p / name
    if candidate.is_file():
        return candidate
    candidate_nested = p / "srtm_tiles" / name
    if candidate_nested.is_file():
        return candidate_nested
    raise FileNotFoundError(
        f"No Copernicus DSM tile for (lat={lat:.2f}, lon={lon:.2f}); "
        f"expected {name} under {p}"
    )


def _cop_dsm_tile_name(lat_ll: int, lon_ll: int) -> str:
    """Copernicus DSM tile filename for the tile whose SW corner is (lat_ll, lon_ll)."""
    lat_dir = "N" if lat_ll >= 0 else "S"
    lon_dir = "E" if lon_ll >= 0 else "W"
    return (f"Copernicus_DSM_COG_10_{lat_dir}{abs(lat_ll):02d}_00_"
            f"{lon_dir}{abs(lon_ll):03d}_00_DEM.tif")


def _resolve_dem_for_window(
    dem: Path, lat: float, lon: float, half_extent_m: float,
) -> tuple[Path, bool]:
    """Resolve a DEM path covering the FULL metric window at (lat, lon).

    When the requested box straddles a 1° tile boundary (≈ a few % of stations),
    a single-tile resolver fills the off-tile pixels with 0 (flat-zero artefact).
    This builds a `rasterio.merge` mosaic of every covered tile into a temp
    GeoTIFF so the downstream reproject sees real data on every pixel. Returns
    (path, is_temp): the caller MUST unlink the temp file when is_temp is True.

    Falls back to the single-file `_resolve_dem_path` when `dem` is already a
    file, or when only one tile is needed (no temp file created).
    """
    p = Path(dem)
    if p.is_file():
        return p, False
    if not p.is_dir():
        raise FileNotFoundError(f"DEM not found: {dem}")
    # Window half-span in degrees (flat-earth, generous on lon via cos(lat)).
    dlat = half_extent_m / 111_000.0
    dlon = half_extent_m / (111_000.0 * max(0.1, math.cos(math.radians(lat))))
    lat_lls = sorted({int(math.floor(lat - dlat)), int(math.floor(lat + dlat))})
    lon_lls = sorted({int(math.floor(lon - dlon)), int(math.floor(lon + dlon))})

    def _find_tile(la: int, lo: int) -> Path | None:
        name = _cop_dsm_tile_name(la, lo)
        for cand in (p / name, p / "srtm_tiles" / name):
            if cand.is_file():
                return cand
        return None

    tiles: list[Path] = []
    for la in lat_lls:
        for lo in lon_lls:
            t = _find_tile(la, lo)
            if t is not None:
                tiles.append(t)
    if len(tiles) <= 1:
        # Single tile (or none found) → defer to the canonical resolver, which
        # raises a precise FileNotFoundError if the centre tile is missing.
        return _resolve_dem_path(p, lat, lon), False
    # Straddles a tile edge → mosaic the covered tiles (rasterio.merge, no extra
    # deps) into a temp GeoTIFF so every pixel of the box has real data.
    import tempfile
    import rasterio
    from rasterio.merge import merge as rio_merge
    srcs = [rasterio.open(t) for t in tiles]
    try:
        mosaic, out_transform = rio_merge(srcs)
        meta = srcs[0].meta.copy()
        meta.update(height=mosaic.shape[1], width=mosaic.shape[2],
                    transform=out_transform, count=mosaic.shape[0])
        tif_path = Path(tempfile.mkstemp(prefix="dsw_dem_", suffix=".tif")[1])
        with rasterio.open(tif_path, "w", **meta) as dst:
            dst.write(mosaic)
    finally:
        for s in srcs:
            s.close()
    return tif_path, True


def _resolve_wc_path(wc: Path, lat: float, lon: float) -> Path:
    """Resolve ESA WorldCover path: pass-through for single file, or pick the
    3°×3° tile matching (lat, lon) when `wc` is a directory of WC tiles.

    Tile naming convention (ESA WC v200):
        ESA_WorldCover_10m_2021_v200_<LAT><LON>_Map.tif
    where <LAT> = N<NN>|S<NN> and <LON> = E<EEE>|W<EEE> are the
    LOWER-LEFT corner snapped to multiples of 3°. Example: (lat=38.77,
    lon=-9.13) → N36W012 (covers lat ∈ [36, 39), lon ∈ [-12, -9)).

    Snapping rule (matches ingest_worldcover_esa.enumerate_tiles):
        lat_ll = floor(lat / 3) * 3
        lon_ll = floor(lon / 3) * 3
    Both formulas use Python's `math.floor` semantics so negative values
    round towards -∞ (e.g. lon=-9.13 → lon_ll = floor(-3.04) * 3 = -12).
    """
    p = Path(wc)
    if p.is_file():
        return p
    if not p.is_dir():
        raise FileNotFoundError(f"WorldCover path not found: {wc}")
    lat_ll = int(math.floor(lat / 3.0) * 3)
    lon_ll = int(math.floor(lon / 3.0) * 3)
    lat_dir = "N" if lat_ll >= 0 else "S"
    lat_idx = lat_ll if lat_ll >= 0 else -lat_ll
    lon_dir = "E" if lon_ll >= 0 else "W"
    lon_idx = lon_ll if lon_ll >= 0 else -lon_ll
    name = f"ESA_WorldCover_10m_2021_v200_{lat_dir}{lat_idx:02d}{lon_dir}{lon_idx:03d}_Map.tif"
    candidate = p / name
    if candidate.is_file():
        return candidate
    # Allow a nested `worldcover_esa/` subdirectory (mirrors srtm_tiles/ layout).
    candidate_nested = p / "worldcover_esa" / name
    if candidate_nested.is_file():
        return candidate_nested
    raise FileNotFoundError(
        f"No ESA WorldCover tile for (lat={lat:.2f}, lon={lon:.2f}); "
        f"expected {name} under {p}"
    )


def extract_terrain_from_dem(
    dem_tif: Path, lat: float, lon: float,
    half_extent_m: float = HALF_EXTENT_M, ni: int = NI, nj: int = NJ,
) -> np.ndarray:
    """Crop a 6 km×6 km terrain patch from a DEM raster and resample to (NI, NJ).

    Uses bilinear resampling. Output shape (NI, NJ) float32 [m].
    Convention: i = E-W (lon-aligned), j = N-S (lat-aligned). Matches
    export_to_grid_zarr_v2 (x increases with longitude eastward, y with latitude
    northward).

    `dem_tif` accepts a single GeoTIFF/VRT or a directory of Copernicus DSM
    tiles (the matching 1°×1° tile is auto-selected per (lat, lon)).
    """
    # Window-aware resolver: mosaics straddling 1° tiles so edge stations get
    # real data on every pixel (not a flat-zero off-tile fill).
    dem_tif, _is_temp = _resolve_dem_for_window(Path(dem_tif), lat, lon, half_extent_m)
    import rasterio
    from rasterio.warp import Resampling, calculate_default_transform, reproject
    from pyproj import Transformer

    try:
        # Local metric frame: pick UTM zone for the site, project box corners,
        # then sample the DEM via reproject to (NI, NJ) regular grid.
        utm_zone = int(math.floor((lon + 180) / 6) % 60 + 1)
        epsg_utm = 32600 + utm_zone if lat >= 0 else 32700 + utm_zone
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_utm}", always_xy=True)
        x0, y0 = transformer.transform(lon, lat)
        dst_x = np.linspace(x0 - half_extent_m, x0 + half_extent_m, ni + 1)
        dst_y = np.linspace(y0 - half_extent_m, y0 + half_extent_m, nj + 1)

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
    finally:
        if _is_temp:
            Path(dem_tif).unlink(missing_ok=True)
    # `dst[row=0]` is northernmost, OF convention has y increasing northward
    # ⇒ flip the row axis so that `terrain[i, j]` with j=0 → south.
    terrain = np.flipud(dst).T.astype(np.float32)  # (ni, nj)
    return terrain


def terrain_slope_deg(
    terrain: np.ndarray,
    dx_m: float = DX,
    window_m: float = 1000.0,
) -> float:
    """Mean terrain slope (degrees) over a central window of the DEM patch.

    `terrain` is the (NI, NJ) patch from `extract_terrain_from_dem` (cell size
    `dx_m` ≈ 33.3 m). Slope per cell = arctan(|∇z|) where the gradient magnitude
    is `sqrt((dz/dx)^2 + (dz/dy)^2)` from `np.gradient` at `dx_m` spacing.
    Aggregated to ONE scalar = mean slope over a central ~`window_m` window
    (default 1 km), which isolates the on-site steepness from domain edges.

    Returns NaN if the patch is empty or all-NaN.
    """
    terrain = np.asarray(terrain, dtype=np.float64)
    if terrain.ndim != 2 or terrain.size == 0 or not np.isfinite(terrain).any():
        return float("nan")
    gy, gx = np.gradient(terrain, dx_m)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))

    ni, nj = terrain.shape
    half = max(1, int(round(0.5 * window_m / dx_m)))
    ci, cj = ni // 2, nj // 2
    i0, i1 = max(0, ci - half), min(ni, ci + half)
    j0, j1 = max(0, cj - half), min(nj, cj + half)
    central = slope[i0:i1, j0:j1]
    central = central[np.isfinite(central)]
    if central.size == 0:
        return float("nan")
    return float(np.mean(central))


def slope_at_dem(
    dem: Path,
    lat: float,
    lon: float,
    *,
    window_m: float = 1000.0,
) -> float:
    """Convenience: read the DEM patch at (lat, lon) and return mean slope (deg).

    `dem` may be a single GeoTIFF or a directory of Copernicus DSM tiles
    (auto-resolved per coordinate by `extract_terrain_from_dem`).
    """
    terrain = extract_terrain_from_dem(Path(dem), lat, lon)
    return terrain_slope_deg(terrain, dx_m=DX, window_m=window_m)


def build_native_z(terrain: np.ndarray) -> np.ndarray:
    """Build coords/z (NI, NJ, NK) = terrain[i,j] + agl_levels[k]."""
    agl = build_agl_levels()
    return (terrain[:, :, None] + agl[None, None, :]).astype(np.float32)


# ─── WorldCover → z0_eff ────────────────────────────────────────────────────

def compute_z0_eff_from_wc(
    wc_tif: Path, lat: float, lon: float,
    patch_radius_m: float = 3000.0,
) -> tuple[float, dict[int, int]]:
    """Geometric mean of z0 over a square patch around the site.

    `wc_tif` accepts a single GeoTIFF or a directory of ESA WorldCover tiles
    (the matching 3°×3° tile is auto-selected per (lat, lon) by
    `_resolve_wc_path`).

    Returns (z0_eff [m], class_counts).
    """
    import rasterio
    from rasterio.transform import rowcol

    wc_path = Path(wc_tif)
    if not wc_path.exists():
        logger.warning("WC raster missing: %s — falling back to z0_eff=%.4f",
                       wc_tif, WC_Z0_DEFAULT)
        return WC_Z0_DEFAULT, {}
    try:
        wc_tif = _resolve_wc_path(wc_path, lat, lon)
    except FileNotFoundError as exc:
        logger.warning("%s — falling back to z0_eff=%.4f", exc, WC_Z0_DEFAULT)
        return WC_Z0_DEFAULT, {}

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


def radcloud_at(
    lat: float,
    lon: float,
    timestamp_iso: str,
    store: Path | str,
    *,
    max_delta_seconds: float = 3 * 3600 + 1,
) -> tuple[float, float]:
    """Return (ssrd_Jm2, tcc) at the nearest grid cell + nearest time.

    ssrd is J/m2 accumulated over the preceding hour; divide by 3600 for W/m2.
    tcc is dimensionless [0,1]. NaN source values are returned as 0.0.
    Raises ValueError if the nearest time is farther than max_delta_seconds.
    """
    import zarr

    g = zarr.open_group(str(store), mode="r")
    times = np.asarray(g["coords/time"][:], dtype=np.int64)
    target_ns = np.datetime64(timestamp_iso).astype("datetime64[ns]").astype(np.int64)
    idx = int(np.argmin(np.abs(times - target_ns)))
    delta = abs(int(times[idx]) - int(target_ns)) / 1e9
    if delta > max_delta_seconds:
        raise ValueError(
            f"radcloud store nearest time {delta/3600:.1f} h away from "
            f"{timestamp_iso} (max allowed {max_delta_seconds/3600:.1f} h)"
        )

    lats = np.asarray(g["coords/lat"][:], dtype=np.float32)
    lons = np.asarray(g["coords/lon"][:], dtype=np.float32)
    i = int(np.argmin(np.abs(lats - lat)))
    j = int(np.argmin(np.abs(lons - lon)))

    ssrd = float(g["ssrd"][idx, i, j])
    tcc = float(g["tcc"][idx, i, j])
    if not np.isfinite(ssrd):
        ssrd = 0.0
    if not np.isfinite(tcc):
        tcc = 0.0
    return ssrd, tcc
