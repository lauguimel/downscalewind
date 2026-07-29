"""
fuxicfd_infer_at_stations.py — Run the OFFICIAL FuXi-CFD ONNX model at OBS stations.

This is the head-to-head benchmark adapter: it feeds OUR station pairings
through the published FuXi-CFD model (Lin Chensen et al., Nat. Commun. 17:3713,
2026; HF `linchensen/FuXi-CFD-model`) and reads the 10 m wind speed at the
central pixel, so we can score FuXi against the SAME obs our surrogate v2
(M_I3/M_I5, val MAE 1.46 m/s) and ERA5 are scored against.

It does NOT touch our surrogate model/training code. It mirrors the CLI /
batching / checkpointing structure of `infer_at_stations.py`.

────────────────────────────────────────────────────────────────────────────
I/O contract (authoritative: scratch/fuxicfd_io_notes.md, reproduced 2026-06-11)
────────────────────────────────────────────────────────────────────────────
INPUT tensor (1, 4, 300, 300), channel order [u_100m, v_100m, dem, roughness]:
  - u_100m, v_100m : 9×9 100 m-AGL wind field (m/s) over a ~1 km-spaced grid.
  - dem            : 301×301 terrain elevation (m).
  - roughness      : 301×301 aerodynamic roughness length z0 (m).
  Preprocessing (utils/preprocessing.build_model_input, verbatim):
    1. standardize each group with scaler_input.npy (high=dem,rough; low=u,v).
    2. scipy.ndimage.zoom: dem_rough (1,300/301,300/301) order=1 -> (2,300,300);
       uv_100m (1,300/9,300/9) order=1 -> (2,300,300).
    3. concat([uv_100m, dem_rough]) -> (4,300,300), add batch dim.
OUTPUT tensor (1, 27, 4, 300, 300) = (level, var, y, x), vars [u,v,w,k]:
  de-norm pred*std+mean via scaler_output.npy (27,4).
  10 m wind = OUTPUT level index 0; centre pixel (row=150, col=150);
  speed = hypot(u[0,150,150], v[0,150,150]).

The 301×301 dem/roughness span the SAME 9 km footprint as the 300×300 model
field; we extract them at 30 m spacing centred on each station (9.0 km wide,
i.e. wider than our surrogate's 6 km / 180×180 window).

────────────────────────────────────────────────────────────────────────────
100 m WIND INPUT — the crux of an apples-to-apples benchmark
────────────────────────────────────────────────────────────────────────────
FuXi wants a 9×9 field of 100 m-AGL u/v. The faithful source is a NATIVE ERA5
100 m wind store (era5_100m_*), being ingested by a parallel mission. Until it
lands, `--uv100-source` selects a clearly-labelled PROXY for SMOKE ONLY:
  - `era5_uv10`   : tile ERA5 u10/v10 (surface) into a uniform 9×9 field.
  - `obs100`      : use the station's own 100 m obs wind (Perdigão has 100 m),
                    uniform 9×9. Best mechanics check; NOT a real forecast.
  - `era5_native` : read u100/v100 from an ERA5 store that actually has them
                    (the real scoring path; selected once the store lands).
The proxy is recorded in the `uv100_source` parquet column so no proxy result
is ever mistaken for the real benchmark.

────────────────────────────────────────────────────────────────────────────
SMOKE (Aqua, CPU, Perdigão, ~10 pairings, obs-100m proxy)
────────────────────────────────────────────────────────────────────────────
  python fuxicfd_infer_at_stations.py \\
    --obs-zarr ~/dsw/data/raw/perdigao_obs.zarr --obs-schema perdigao \\
    --onnx ~/dsw/data/models/fuxicfd_official/fuxicfd-model/model/fuxicfd_model.onnx \\
    --scaler-dir ~/dsw/data/models/fuxicfd_official/fuxicfd-model/inference_example/normalization \\
    --dem ~/dsw/data/raw/srtm_tiles --worldcover ~/dsw/data/raw/worldcover_esa \\
    --uv100-source obs100 \\
    --era5-store ~/dsw/data/raw/era5_europe_spring2017_v2.zarr \\
    --output ~/dsw/data/inference/fuxicfd_smoke_perdigao.parquet \\
    --smoke --max-pairings 10
"""
from __future__ import annotations

import logging
import math
import sys
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd

# Local utils import (reuse OUR terrain/z0 reading logic — do NOT reimplement).
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from utils.inference_input import (  # noqa: E402
    WC_Z0_DEFAULT,
    WC_Z0_LOOKUP,
    _resolve_dem_path,
    _resolve_wc_path,
    extract_terrain_from_dem,
)

logger = logging.getLogger("fuxicfd_infer_at_stations")

# ─── FuXi grid constants (verified contract) ────────────────────────────────
FUXI_HIGH_RES = 301          # dem/roughness side
FUXI_TARGET = 300            # model field side
FUXI_LOW_RES = 9             # 100 m wind field side
FUXI_DEM_SPACING_M = 30.0    # 30 m
# 301 cells × 30 m at cell-center spacing → 300 intervals → 9000 m footprint.
FUXI_HALF_EXTENT_M = 0.5 * (FUXI_HIGH_RES - 1) * FUXI_DEM_SPACING_M  # 4500.0 m
FUXI_CENTRE = FUXI_TARGET // 2  # 150
OUT_LEVEL_10M = 0               # output level index 0 == 10 m AGL


# ─── FuXi static-input extraction (301×301 dem + roughness) ─────────────────

def _window_deg_halfspan(lat: float) -> tuple[float, float]:
    """Half-window of the 9 km FuXi footprint in degrees (lat, lon)."""
    dlat = FUXI_HALF_EXTENT_M / 111_000.0
    dlon = FUXI_HALF_EXTENT_M / (111_000.0 * max(0.1, math.cos(math.radians(lat))))
    return dlat, dlon


def _cop_dsm_tile_name(lat_ll: int, lon_ll: int) -> str:
    lat_dir = "N" if lat_ll >= 0 else "S"
    lon_dir = "E" if lon_ll >= 0 else "W"
    return (f"Copernicus_DSM_COG_10_{lat_dir}{abs(lat_ll):02d}_00_"
            f"{lon_dir}{abs(lon_ll):03d}_00_DEM.tif")


def resolve_dem_for_window(dem: Path, lat: float, lon: float):
    """Return a single DEM path covering the FULL 9 km window at (lat, lon).

    Copernicus DSM tiles are named by their LOWER-LEFT corner: tile with
    lon_ll=L covers [L, L+1). Hence the correct tile for a coordinate is
    `floor(lon)` (NOT `abs(floor(-lon))`, which is the bug in the shared
    `_resolve_dem_path` for negative longitudes — it returned W007 for
    lon=-7.73 whose data lives in W008, yielding an all-zero patch).

    If the window straddles a 1° tile edge (≈19/106 val stations), build a
    VRT mosaic of all covered tiles so the reproject sees real data on every
    pixel. Returns (path, is_temp_vrt). Falls back to the shared single-file
    resolver if `dem` is already a file.
    """
    import math as _m
    p = Path(dem)
    if p.is_file():
        return p, False
    dlat, dlon = _window_deg_halfspan(lat)
    lat_lls = sorted({int(_m.floor(lat - dlat)), int(_m.floor(lat + dlat))})
    lon_lls = sorted({int(_m.floor(lon - dlon)), int(_m.floor(lon + dlon))})
    tiles: list[Path] = []
    for la in lat_lls:
        for lo in lon_lls:
            name = _cop_dsm_tile_name(la, lo)
            cand = p / name
            if not cand.is_file():
                cand2 = p / "srtm_tiles" / name
                cand = cand2 if cand2.is_file() else cand
            if cand.is_file():
                tiles.append(cand)
    if not tiles:
        raise FileNotFoundError(
            f"No Copernicus DSM tile covers 9 km window at "
            f"(lat={lat:.3f}, lon={lon:.3f}); lat_ll={lat_lls} lon_ll={lon_lls}")
    if len(tiles) == 1:
        return tiles[0], False
    # Mosaic the covered tiles (rasterio.merge, no extra deps) into a temp
    # GeoTIFF so the downstream reproject sees real data on every pixel.
    import tempfile
    import rasterio
    from rasterio.merge import merge as rio_merge
    srcs = [rasterio.open(t) for t in tiles]
    try:
        mosaic, out_transform = rio_merge(srcs)
        meta = srcs[0].meta.copy()
        meta.update(height=mosaic.shape[1], width=mosaic.shape[2],
                    transform=out_transform, count=mosaic.shape[0])
        tif_path = Path(tempfile.mkstemp(prefix="fuxi_dem_", suffix=".tif")[1])
        with rasterio.open(tif_path, "w", **meta) as dst:
            dst.write(mosaic)
    finally:
        for s in srcs:
            s.close()
    return tif_path, True


def extract_fuxi_dem(dem: Path, lat: float, lon: float) -> np.ndarray:
    """301×301 DEM (m) on a 9 km window centred at (lat, lon).

    Reuses OUR `extract_terrain_from_dem` (UTM reproject + bilinear) but with the
    FuXi window (half_extent=4500 m, 301×301) and a window-correct DEM path
    (see `resolve_dem_for_window`). Output indexed (row=y=lat, col=x=lon) to
    match the FuXi convention; `extract_terrain_from_dem` returns (i=lon, j=lat),
    so we transpose.
    """
    dem_path, is_temp = resolve_dem_for_window(dem, lat, lon)
    try:
        terr_ij = extract_terrain_from_dem(
            dem_path, lat, lon,
            half_extent_m=FUXI_HALF_EXTENT_M, ni=FUXI_HIGH_RES, nj=FUXI_HIGH_RES,
        )  # (NI=lon, NJ=lat)
    finally:
        if is_temp:
            Path(dem_path).unlink(missing_ok=True)
    return np.ascontiguousarray(terr_ij.T).astype(np.float32)


def extract_fuxi_roughness(wc: Path, lat: float, lon: float) -> np.ndarray:
    """301×301 roughness z0 (m) field, ESA WorldCover classes -> z0 lookup.

    Reads the WC raster on the SAME 9 km window as the DEM, reprojected to a
    301×301 metric grid via nearest-neighbour (categorical), then maps each class
    to z0 with `WC_Z0_LOOKUP` (geometric-mean fallback `WC_Z0_DEFAULT` on missing
    tile). Output indexed (row=y=lat, col=x=lon) to match `extract_fuxi_dem`.
    """
    import rasterio
    from rasterio.transform import Affine
    from rasterio.warp import Resampling, reproject
    from pyproj import Transformer

    wc_path = Path(wc)
    try:
        wc_tif = _resolve_wc_path(wc_path, lat, lon) if not wc_path.is_file() else wc_path
    except FileNotFoundError as exc:
        logger.warning("%s — uniform roughness z0=%.4f m", exc, WC_Z0_DEFAULT)
        return np.full((FUXI_HIGH_RES, FUXI_HIGH_RES), WC_Z0_DEFAULT, np.float32)

    utm_zone = int(math.floor((lon + 180) / 6) % 60 + 1)
    epsg_utm = 32600 + utm_zone if lat >= 0 else 32700 + utm_zone
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_utm}", always_xy=True)
    x0, y0 = transformer.transform(lon, lat)
    he = FUXI_HALF_EXTENT_M
    n = FUXI_HIGH_RES
    # Destination grid: row 0 = north (max y), col 0 = west (min x).
    dst_transform = Affine.translation(x0 - he, y0 + he) * Affine.scale(
        2 * he / n, -2 * he / n
    )
    dst = np.zeros((n, n), dtype=np.int32)
    with rasterio.open(wc_tif) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=f"EPSG:{epsg_utm}",
            resampling=Resampling.nearest,   # categorical land-cover
        )
    # Map classes -> z0 (m). dst row 0 = north; FuXi y axis (we keep row 0 = north,
    # consistent with extract_fuxi_dem which transposes the OF (lon, lat) patch:
    # extract_terrain_from_dem flips so j=0 is south, then .T makes row=lat with
    # row 0 = south. Make both consistent: flip roughness so row 0 = south too.)
    dst = np.flipud(dst)  # now row 0 = south, matching extract_fuxi_dem's (lat,lon)
    z0 = np.full(dst.shape, WC_Z0_DEFAULT, dtype=np.float32)
    for cls, val in WC_Z0_LOOKUP.items():
        z0[dst == cls] = val
    return z0.astype(np.float32)


# ─── 100 m wind field providers (9×9) ───────────────────────────────────────

def _uniform_9x9(u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
    u9 = np.full((FUXI_LOW_RES, FUXI_LOW_RES), float(u), np.float32)
    v9 = np.full((FUXI_LOW_RES, FUXI_LOW_RES), float(v), np.float32)
    return u9, v9


def uv100_from_era5_uv10(
    era5_store: Path, lat: float, lon: float, timestamp_ns: int,
    *, max_delta_h: float = 6.5,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    """PROXY: nearest-time ERA5 u10/v10 at (lat,lon), tiled to a uniform 9×9.

    Returns (u9, v9, delta_h, label). NOT a 100 m wind — surface 10 m proxy for
    SMOKE mechanics only.
    """
    import zarr
    g = zarr.open_group(str(era5_store), mode="r")
    times = np.asarray(g["coords/time"][:], dtype=np.int64)
    idx = int(np.argmin(np.abs(times - timestamp_ns)))
    delta_h = abs(int(times[idx]) - int(timestamp_ns)) / 3.6e12
    if delta_h > max_delta_h:
        raise ValueError(f"ERA5 nearest time {delta_h:.1f} h > {max_delta_h} h")
    lats = np.asarray(g["coords/lat"][:], dtype=np.float32)
    lons = np.asarray(g["coords/lon"][:], dtype=np.float32)
    i = int(np.argmin(np.abs(lats - lat)))
    j = int(np.argmin(np.abs(lons - lon)))
    u10 = float(g["surface/u10"][idx, i, j])
    v10 = float(g["surface/v10"][idx, i, j])
    u9, v9 = _uniform_9x9(u10, v10)
    return u9, v9, delta_h, "era5_uv10_PROXY"


def uv100_from_era5_native(
    era5_store: Path, lat: float, lon: float, timestamp_ns: int,
    *, max_delta_h: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    """REAL PATH (once store lands): ERA5 native u100/v100, 3×3 -> bilinear 9×9.

    Expects a store with `surface/u100` & `surface/v100` (or `pressure`-derived
    100 m). Reads a 3×3 footprint around (lat,lon) and bilinearly upsamples to 9×9
    so the model sees real spatial structure rather than a uniform field.
    """
    import zarr
    from scipy.ndimage import zoom
    g = zarr.open_group(str(era5_store), mode="r")
    surf = g["surface"]
    if "u100" not in surf or "v100" not in surf:
        raise KeyError(
            f"{era5_store} has no surface/u100,v100 — native 100 m store not ready"
        )
    times = np.asarray(g["coords/time"][:], dtype=np.int64)
    idx = int(np.argmin(np.abs(times - timestamp_ns)))
    delta_h = abs(int(times[idx]) - int(timestamp_ns)) / 3.6e12
    if delta_h > max_delta_h:
        raise ValueError(f"ERA5-100m nearest time {delta_h:.1f} h > {max_delta_h} h")
    lats = np.asarray(g["coords/lat"][:], dtype=np.float32)
    lons = np.asarray(g["coords/lon"][:], dtype=np.float32)
    ic = int(np.argmin(np.abs(lats - lat)))
    jc = int(np.argmin(np.abs(lons - lon)))
    i0 = max(0, min(len(lats) - 3, ic - 1))
    j0 = max(0, min(len(lons) - 3, jc - 1))
    u3 = np.asarray(surf["u100"][idx, i0:i0 + 3, j0:j0 + 3], dtype=np.float32)
    v3 = np.asarray(surf["v100"][idx, i0:i0 + 3, j0:j0 + 3], dtype=np.float32)
    u9 = zoom(u3, FUXI_LOW_RES / 3.0, order=1).astype(np.float32)
    v9 = zoom(v3, FUXI_LOW_RES / 3.0, order=1).astype(np.float32)
    return u9, v9, delta_h, "era5_native_100m"


# ─── FuXi ONNX session + pre/post (verified contract, inlined) ──────────────

class FuxiRunner:
    def __init__(self, onnx_path: Path, scaler_dir: Path, device: str = "cpu"):
        import os
        import onnxruntime as ort
        self.in_stats = np.load(scaler_dir / "scaler_input.npy", allow_pickle=True).item()
        self.out_stats = np.load(scaler_dir / "scaler_output.npy", allow_pickle=True).item()
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if device == "cuda" else ["CPUExecutionProvider"])
        # Cap ONNX threads on CPU to avoid pthread_setaffinity oversubscription/spam.
        _so = ort.SessionOptions()
        _so.intra_op_num_threads = int(os.environ.get("OMP_NUM_THREADS", "8"))
        _so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(onnx_path), sess_options=_so,
                                         providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        # Sanity: channel order must be wind-first.
        order = list(self.in_stats.get("input_channel_order", []))
        if order and order != ["u_100m", "v_100m", "dem", "roughness"]:
            raise RuntimeError(f"Unexpected input_channel_order: {order}")
        logger.info("FuXi ONNX loaded (%s) providers=%s input_channel_order=%s",
                    onnx_path.name, self.sess.get_providers(), order)

    def build_input(self, u9, v9, dem301, rough301) -> np.ndarray:
        """(1,4,300,300) per utils/preprocessing.build_model_input (verbatim)."""
        from scipy.ndimage import zoom
        s = self.in_stats
        dem_rough = np.stack([dem301, rough301], axis=0).astype(np.float64)  # (2,301,301)
        uv = np.stack([u9, v9], axis=0).astype(np.float64)                   # (2,9,9)
        dem_rough = (dem_rough - s["high_mean"][:, None, None]) / s["high_std"][:, None, None]
        uv = (uv - s["low_mean"][:, None, None]) / s["low_std"][:, None, None]
        dem_rough = zoom(dem_rough, (1, FUXI_TARGET / FUXI_HIGH_RES, FUXI_TARGET / FUXI_HIGH_RES), order=1)
        uv = zoom(uv, (1, FUXI_TARGET / FUXI_LOW_RES, FUXI_TARGET / FUXI_LOW_RES), order=1)
        x = np.concatenate([uv, dem_rough], axis=0).astype(np.float32)       # (4,300,300)
        return x[None, ...]

    def forward_speed10_centre(self, x_batch: np.ndarray) -> np.ndarray:
        """Run a (B,4,300,300) batch -> 10 m centre-pixel speed per sample (B,)."""
        pred = self.sess.run(None, {self.input_name: x_batch})[0]  # (B,27,4,300,300)
        mean = self.out_stats["mean"][:, :, None, None]            # (27,4,1,1)
        std = self.out_stats["std"][:, :, None, None]
        speeds = np.empty(pred.shape[0], dtype=np.float32)
        for b in range(pred.shape[0]):
            p = pred[b] * std + mean                               # (27,4,300,300)
            u = p[OUT_LEVEL_10M, 0, FUXI_CENTRE, FUXI_CENTRE]
            v = p[OUT_LEVEL_10M, 1, FUXI_CENTRE, FUXI_CENTRE]
            speeds[b] = math.hypot(float(u), float(v))
        return speeds


# ─── OBS pairings (two schemas: obs_unified `stations/`, perdigao `sites/`) ──

def _decode(values: np.ndarray) -> list[str]:
    out = []
    for v in values:
        if isinstance(v, (bytes, np.bytes_)):
            out.append(v.decode("utf-8").rstrip("\x00"))
        else:
            out.append(str(v).rstrip("\x00"))
    return out


def load_perdigao_pairings(
    obs_zarr: Path, *, height_target: float = 10.0,
    max_pairings: int | None = None,
) -> pd.DataFrame:
    """Pairings from the legacy `perdigao_obs.zarr` (sites/ schema, multi-height).

    Columns: station_id, lat, lon, elev, height_obs, timestamp_ns, u_obs, v_obs,
    speed_obs, u100_obs, v100_obs (100 m obs wind for the obs100 proxy).
    """
    import zarr
    g = zarr.open_group(str(obs_zarr), mode="r")
    sids = _decode(g["coords/site_id"][:])
    lats = np.asarray(g["coords/lat"][:], dtype=np.float32)
    lons = np.asarray(g["coords/lon"][:], dtype=np.float32)
    elevs = np.asarray(g["coords/altitude_m"][:], dtype=np.float32)
    times = np.asarray(g["coords/time"][:], dtype=np.int64)
    heights = np.asarray(g["coords/height_m"][:], dtype=np.float32)
    h_idx = int(np.argmin(np.abs(heights - height_target)))
    h100_idx = int(np.argmin(np.abs(heights - 100.0)))
    u = np.asarray(g["sites/u"][:, :, h_idx], dtype=np.float32)        # (T,S)
    v = np.asarray(g["sites/v"][:, :, h_idx], dtype=np.float32)
    u100 = np.asarray(g["sites/u"][:, :, h100_idx], dtype=np.float32)
    v100 = np.asarray(g["sites/v"][:, :, h100_idx], dtype=np.float32)
    frames = []
    for s in range(len(sids)):
        valid = np.isfinite(u[:, s]) & np.isfinite(v[:, s])
        if not valid.any():
            continue
        frames.append(pd.DataFrame({
            "station_id": f"perdigao_{sids[s]}"[:24],
            "lat": float(lats[s]), "lon": float(lons[s]), "elev": float(elevs[s]),
            "height_obs": float(heights[h_idx]),
            "timestamp_ns": times[valid],
            "u_obs": u[valid, s], "v_obs": v[valid, s],
            "speed_obs": np.hypot(u[valid, s], v[valid, s]),
            "u100_obs": u100[valid, s], "v100_obs": v100[valid, s],
        }))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if max_pairings is not None and len(df) > max_pairings:
        # Spread across stations/times rather than first-N of one station.
        df = df.sample(n=max_pairings, random_state=42).reset_index(drop=True)
    logger.info("Perdigão: %d pairings (%d stations) @ h=%.0f m",
                len(df), df["station_id"].nunique() if len(df) else 0,
                float(heights[h_idx]))
    return df


def load_unified_pairings(
    obs_zarr: Path, *, height_target: float = 10.0,
    max_pairings: int | None = None,
) -> pd.DataFrame:
    """Pairings from an `obs_unified_*.zarr` (stations/ schema)."""
    import zarr
    g = zarr.open_group(str(obs_zarr), mode="r")
    sids = _decode(g["stations/station_id"][:])
    lats = np.asarray(g["stations/lat"][:], dtype=np.float32)
    lons = np.asarray(g["stations/lon"][:], dtype=np.float32)
    elevs = np.asarray(g["stations/elev"][:], dtype=np.float32)
    times = np.asarray(g["coords/time"][:], dtype=np.int64)
    heights = np.asarray(g["heights/height_m"][:], dtype=np.float32)
    h_idx = int(np.argmin(np.abs(heights - height_target)))
    u = np.asarray(g["data/u"][:, :, h_idx], dtype=np.float32)
    v = np.asarray(g["data/v"][:, :, h_idx], dtype=np.float32)
    ws = np.asarray(g["data/wind_speed"][:, :, h_idx], dtype=np.float32)
    frames = []
    for s in range(len(sids)):
        valid = np.isfinite(u[:, s]) & np.isfinite(v[:, s])
        if not valid.any():
            continue
        frames.append(pd.DataFrame({
            "station_id": sids[s],
            "lat": float(lats[s]), "lon": float(lons[s]), "elev": float(elevs[s]),
            "height_obs": float(heights[h_idx]),
            "timestamp_ns": times[valid],
            "u_obs": u[valid, s], "v_obs": v[valid, s],
            "speed_obs": ws[valid, s],
            "u100_obs": np.full(int(valid.sum()), np.nan, np.float32),
            "v100_obs": np.full(int(valid.sum()), np.nan, np.float32),
        }))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if max_pairings is not None and len(df) > max_pairings:
        df = df.sample(n=max_pairings, random_state=42).reset_index(drop=True)
    logger.info("Unified obs: %d pairings (%d stations) @ h=%.0f m",
                len(df), df["station_id"].nunique() if len(df) else 0,
                float(heights[h_idx]))
    return df


# ─── Main pipeline ──────────────────────────────────────────────────────────

def run(
    *, df: pd.DataFrame, runner: FuxiRunner, dem: Path, worldcover: Path,
    uv100_source: str, era5_store: Path | None, era5_native_store: Path | None,
    output: Path, batch_size: int, max_era5_delta_h: float,
    cache_statics: bool,
) -> Path:
    out_rows: list[dict] = []
    static_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    n_ok = n_skip = 0
    t0 = time.time()

    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        xs: list[np.ndarray] = []
        metas: list[dict] = []
        for _, row in chunk.iterrows():
            try:
                key = f"{row['lat']:.5f}_{row['lon']:.5f}"
                if cache_statics and key in static_cache:
                    dem301, rough301 = static_cache[key]
                else:
                    dem301 = extract_fuxi_dem(dem, float(row["lat"]), float(row["lon"]))
                    rough301 = extract_fuxi_roughness(worldcover, float(row["lat"]), float(row["lon"]))
                    if cache_statics:
                        static_cache[key] = (dem301, rough301)

                # 100 m wind field provider
                if uv100_source == "obs100":
                    u100, v100 = float(row.get("u100_obs", np.nan)), float(row.get("v100_obs", np.nan))
                    if not (np.isfinite(u100) and np.isfinite(v100)):
                        raise ValueError("obs100 proxy: no finite 100 m obs wind")
                    u9, v9 = _uniform_9x9(u100, v100)
                    uv_delta_h, uv_label = 0.0, "obs100_PROXY"
                elif uv100_source == "era5_uv10":
                    u9, v9, uv_delta_h, uv_label = uv100_from_era5_uv10(
                        era5_store, float(row["lat"]), float(row["lon"]),
                        int(row["timestamp_ns"]), max_delta_h=max_era5_delta_h)
                elif uv100_source == "era5_native":
                    u9, v9, uv_delta_h, uv_label = uv100_from_era5_native(
                        era5_native_store, float(row["lat"]), float(row["lon"]),
                        int(row["timestamp_ns"]), max_delta_h=max_era5_delta_h)
                else:
                    raise ValueError(f"unknown uv100_source {uv100_source}")

                xs.append(runner.build_input(u9, v9, dem301, rough301))
                metas.append({
                    "row": row, "uv_delta_h": uv_delta_h, "uv_label": uv_label,
                    "u100_in": float(u9[4, 4]), "v100_in": float(v9[4, 4]),
                })
            except Exception as exc:
                logger.warning("skip %s @ %s: %s",
                               row.get("station_id"), row.get("timestamp_ns"), exc)
                n_skip += 1

        if not xs:
            continue
        x_batch = np.concatenate(xs, axis=0)                       # (B,4,300,300)
        speeds = runner.forward_speed10_centre(x_batch)
        for meta, speed in zip(metas, speeds):
            row = meta["row"]
            ts_iso = str(np.array(int(row["timestamp_ns"])).astype("datetime64[ns]"))
            out_rows.append({
                "station_id": row["station_id"], "timestamp": ts_iso,
                "lat": float(row["lat"]), "lon": float(row["lon"]),
                "elev": float(row["elev"]), "height_obs": float(row["height_obs"]),
                "speed_obs": float(row["speed_obs"]),
                "speed_fuxi": float(speed),
                "u100_in": meta["u100_in"], "v100_in": meta["v100_in"],
                "uv100_source": meta["uv_label"],
                "uv100_delta_h": meta["uv_delta_h"],
            })
            n_ok += 1
        logger.info("[%d/%d] ok=%d skip=%d elapsed=%.1fs",
                    min(start + batch_size, len(df)), len(df), n_ok, n_skip,
                    time.time() - t0)
        # Checkpoint after EVERY batch (terrain reproject is slow; survive kills).
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(out_rows).to_parquet(output, index=False)
        logger.info("checkpoint: %d rows -> %s", len(out_rows), output)

    if not out_rows:
        logger.error("No FuXi outputs produced — aborting.")
        sys.exit(2)
    output.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(out_rows)
    out.to_parquet(output, index=False)
    logger.info("WROTE %s (%d rows). speed_fuxi: mean=%.2f min=%.2f max=%.2f | "
                "speed_obs mean=%.2f | MAE=%.3f m/s",
                output, len(out), out["speed_fuxi"].mean(),
                out["speed_fuxi"].min(), out["speed_fuxi"].max(),
                out["speed_obs"].mean(),
                float(np.mean(np.abs(out["speed_fuxi"] - out["speed_obs"]))))
    return output


# ─── CLI ─────────────────────────────────────────────────────────────────────

@click.command(context_settings={"show_default": True})
@click.option("--obs-zarr", type=click.Path(exists=True, path_type=Path), required=True,
              help="OBS Zarr: perdigao_obs.zarr or obs_unified_*.zarr")
@click.option("--obs-schema", type=click.Choice(["perdigao", "unified"]),
              default="unified", help="Schema of --obs-zarr")
@click.option("--onnx", type=click.Path(exists=True, path_type=Path), required=True,
              help="FuXi-CFD ONNX weights")
@click.option("--scaler-dir", type=click.Path(exists=True, path_type=Path), required=True,
              help="Dir with scaler_input.npy + scaler_output.npy")
@click.option("--dem", type=click.Path(exists=True, path_type=Path), required=True,
              help="DEM GeoTIFF or directory of Copernicus DSM tiles")
@click.option("--worldcover", type=click.Path(exists=True, path_type=Path), required=True,
              help="ESA WorldCover GeoTIFF or directory of WC tiles")
@click.option("--uv100-source", type=click.Choice(["obs100", "era5_uv10", "era5_native"]),
              required=True,
              help="100 m wind input. obs100/era5_uv10 are SMOKE PROXIES; "
                   "era5_native is the real-scoring path (needs u100/v100 store).")
@click.option("--era5-store", type=click.Path(exists=False, path_type=Path), default=None,
              help="ERA5 store for era5_uv10 proxy (u10/v10)")
@click.option("--era5-native-store", type=click.Path(exists=False, path_type=Path), default=None,
              help="ERA5 store with surface/u100,v100 for era5_native")
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--device", type=click.Choice(["cpu", "cuda"]), default="cpu")
@click.option("--batch-size", type=int, default=8)
@click.option("--max-era5-delta-h", type=float, default=6.5)
@click.option("--height-target", type=float, default=10.0,
              help="OBS height (m AGL) used as ground truth")
@click.option("--max-pairings", type=int, default=None)
@click.option("--smoke", is_flag=True, default=False,
              help="Smoke: cap to 10 pairings if --max-pairings unset; verbose")
@click.option("--no-cache-statics", is_flag=True, default=False,
              help="Disable per-coord DEM/roughness caching")
@click.option("--verbose", "-v", is_flag=True, default=False)
def cli(obs_zarr, obs_schema, onnx, scaler_dir, dem, worldcover, uv100_source,
        era5_store, era5_native_store, output, device, batch_size,
        max_era5_delta_h, height_target, max_pairings, smoke,
        no_cache_statics, verbose):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    # Silence verbose third-party loggers (rasterio/GDAL/pyproj emit ~50 DEBUG
    # lines per terrain crop) — keep our own logger at the chosen level.
    for noisy in ("rasterio", "rasterio._env", "rasterio.env", "fiona",
                  "pyproj", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if smoke:
        logger.setLevel(logging.INFO)
    if smoke and max_pairings is None:
        max_pairings = 10
    if uv100_source == "era5_uv10" and era5_store is None:
        raise click.BadParameter("--uv100-source era5_uv10 needs --era5-store")
    if uv100_source == "era5_native" and era5_native_store is None:
        raise click.BadParameter("--uv100-source era5_native needs --era5-native-store")

    if obs_schema == "perdigao":
        df = load_perdigao_pairings(obs_zarr, height_target=height_target,
                                    max_pairings=max_pairings)
    else:
        df = load_unified_pairings(obs_zarr, height_target=height_target,
                                   max_pairings=max_pairings)
    if df.empty:
        logger.error("No usable OBS pairings — aborting.")
        sys.exit(2)

    runner = FuxiRunner(Path(onnx), Path(scaler_dir), device=device)
    run(
        df=df, runner=runner, dem=Path(dem), worldcover=Path(worldcover),
        uv100_source=uv100_source,
        era5_store=Path(era5_store) if era5_store else None,
        era5_native_store=Path(era5_native_store) if era5_native_store else None,
        output=Path(output), batch_size=batch_size,
        max_era5_delta_h=max_era5_delta_h,
        cache_statics=not no_cache_statics,
    )


if __name__ == "__main__":
    cli()
