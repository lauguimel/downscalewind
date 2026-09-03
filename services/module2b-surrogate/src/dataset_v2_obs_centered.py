"""
dataset_v2_obs_centered.py — OBS-centered patch dataset for Phase H' DEVINE-style
training (M_H'0).

Each item materialises a v2 grid.zarr centered on a station coord at a given
timestamp (via the M_G6 deliverable `extract_v2_input_at_coords.build_one`),
then loads it through the same normalisation pipeline used by
`WindV2DatasetViT` / `infer_at_stations.py` (utils.inference_batch.build_features).

Returns per __getitem__:
    terrain_2d      (2, 180, 180)        terrain_norm + z0_broadcast_norm
    era5_flat       (408,)               full ERA5 flat vector
    geo             (2, 180, 180, 24)    z_norm + agl_norm at AGL 0-100 m 24 levels
    topo_features   (8,) or (12,)        base topo vector, optionally with physical
                                          stability features appended
    speed_obs       scalar (float32)     in m/s
    k_obs           scalar (int64)       index of nearest AGL level to height_obs
    meta            dict                 station_id, timestamp_iso, source

The pairings come from a parquet (same schema as infer_at_stations output):
    columns: station_id, timestamp, lat, lon, elev, height_obs, speed_obs, ...

For the smoke run, the dataset pre-builds (materialises) grid.zarrs to a cache
directory at construction time so __getitem__ is fast (just normalisation).
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import Dataset

from .dataset_v2 import DEFAULT_NORM, parse_agl_levels

logger = logging.getLogger(__name__)

# Native grid + central pixel constants (must match utils.inference_input)
NI, NJ = 180, 180
I_CENTER, J_CENTER = NI // 2, NJ // 2  # (90, 90)

# Multi-height station ids encode the level as a suffix (e.g. icos_HPB_h093,
# perdigao_rne01_h010). All heights of one physical tower share lat/lon, hence
# the SAME input grid — the suffix is stripped for cache paths and split groups.
_HEIGHT_SUFFIX_RE = re.compile(r"_h\d+$")

PHYS_FEATURE_NORM = {
    "grad_T_850_surf": {"mean": -8.0, "std": 6.0},    # K, lapse surface->850 hPa
    "grad_T_500_850": {"mean": -20.0, "std": 8.0},    # K, 500-850 thickness lapse
    "RH_surface": {"mean": 60.0, "std": 25.0},         # %
    "q_surface": {"mean": 0.008, "std": 0.005},        # kg/kg
}
P_REF_SURFACE_HPA = 1013.25


# ─── Topo features (8 components) ────────────────────────────────────────────


def _normalise_phys_feature(name: str, raw_value: float | None) -> np.float32:
    stats = PHYS_FEATURE_NORM[name]
    mean = float(stats["mean"])
    std = max(float(stats["std"]), 1e-12)
    if raw_value is None or not np.isfinite(raw_value):
        return np.float32(0.0)
    value = (float(raw_value) - mean) / std
    return np.float32(value if np.isfinite(value) else 0.0)


def _read_zarr_array(g, key: str) -> np.ndarray | None:
    try:
        return np.asarray(g[key][:], dtype=np.float32)
    except (KeyError, TypeError, ValueError, IndexError, AttributeError):
        return None


def _centre_profile(g, key: str) -> np.ndarray | None:
    arr = _read_zarr_array(g, key)
    if arr is None or arr.ndim != 3 or arr.shape[0] <= 1 or arr.shape[1] <= 1:
        return None
    prof = np.asarray(arr[1, 1, :], dtype=np.float32).reshape(-1)
    return prof if prof.size > 0 and np.all(np.isfinite(prof)) else None


def _centre_scalar(g, key: str) -> float | None:
    arr = _read_zarr_array(g, key)
    if arr is None or arr.ndim != 2 or arr.shape[0] <= 1 or arr.shape[1] <= 1:
        return None
    value = float(arr[1, 1])
    return value if np.isfinite(value) else None


def _magnus_tetens_es_hpa(t_celsius: float | None) -> float | None:
    if t_celsius is None or not np.isfinite(t_celsius):
        return None
    denom = 243.12 + float(t_celsius)
    if abs(denom) < 1e-12:
        return None
    exponent = 17.62 * float(t_celsius) / denom
    e_hpa = 6.112 * math.exp(max(min(exponent, 80.0), -80.0))
    return e_hpa if np.isfinite(e_hpa) and e_hpa >= 0.0 else None


def compute_phys_features(g, norm: dict) -> np.ndarray:
    """Build the 4-dim normalised physical stability feature vector.

    Reads centre values from an already-open grid.zarr group and computes raw:
        [0] gradient_T_850_surf = T_centre[idx_850] - t2m_centre (K)
        [1] gradient_T_500_850  = T_centre[idx_500] - T_centre[idx_850] (K)
        [2] RH_surface          = 100 * es(d2m_C) / es(t2m_C), clipped to [0, 100] (%)
        [3] q_surface           = 0.622 * es(d2m_C) / (p_ref - 0.378 * es(d2m_C)) (kg/kg)

    `es` is Magnus-Tetens saturation vapour pressure in hPa:
        6.112 * exp(17.62 * T_celsius / (243.12 + T_celsius)).

    `q_surface` uses fixed p_ref = 1013.25 hPa because grid.zarr has no surface
    pressure (`sp`) array. Values are normalised with PHYS_FEATURE_NORM. Missing
    or invalid source arrays yield the affected feature's normalised mean (0.0).
    The returned order is:
        grad_T_850_surf_n, grad_T_500_850_n, RH_surface_n, q_surface_n.
    """
    _ = norm  # Kept for parity with other dataset feature builders.
    levels = _read_zarr_array(g, "input/era5_pressure_levels")
    t_prof = _centre_profile(g, "input/era5_3d/T")
    t2m = _centre_scalar(g, "input/era5_surface/t2m")
    d2m = _centre_scalar(g, "input/era5_surface/d2m")

    grad_850_surf: float | None = None
    grad_500_850: float | None = None
    if levels is not None and t_prof is not None:
        levels = np.asarray(levels, dtype=np.float32).reshape(-1)
        n = min(levels.size, t_prof.size)
        if n > 0:
            levels = levels[:n]
            t_prof = t_prof[:n]
            idx_850 = int(np.argmin(np.abs(levels - 850.0)))
            idx_500 = int(np.argmin(np.abs(levels - 500.0)))
            t_850 = float(t_prof[idx_850])
            t_500 = float(t_prof[idx_500])
            if t2m is not None:
                grad_850_surf = t_850 - t2m
            grad_500_850 = t_500 - t_850

    t2m_c = None if t2m is None else t2m - 273.15
    d2m_c = None if d2m is None else d2m - 273.15
    es_t2m = _magnus_tetens_es_hpa(t2m_c)
    es_d2m = _magnus_tetens_es_hpa(d2m_c)

    rh_surface: float | None = None
    if es_t2m is not None and es_d2m is not None and es_t2m > 0.0:
        rh_surface = float(np.clip(100.0 * es_d2m / es_t2m, 0.0, 100.0))

    q_surface: float | None = None
    if es_d2m is not None:
        denom = P_REF_SURFACE_HPA - 0.378 * es_d2m
        if denom > 1e-12:
            q_surface = max(0.0, 0.622 * es_d2m / denom)

    out = np.array(
        [
            _normalise_phys_feature("grad_T_850_surf", grad_850_surf),
            _normalise_phys_feature("grad_T_500_850", grad_500_850),
            _normalise_phys_feature("RH_surface", rh_surface),
            _normalise_phys_feature("q_surface", q_surface),
        ],
        dtype=np.float32,
    )
    out[~np.isfinite(out)] = 0.0
    return out


def compute_topo_features(
    terrain_raw: np.ndarray,
    z0_eff: float,
    lat: float,
    timestamp_iso: str,
    norm: dict,
    patch_half: int = 15,
    phys_features: np.ndarray | None = None,
) -> np.ndarray:
    """Build the topo feature vector for one pairing.

    Base 8 components (all roughly O(1) after normalisation):
        [0] mean_topo_local      mean elevation in 30×30 voxels around centre (m), scaled
        [1] std_topo_local       std of elevation in same patch (m), scaled
        [2] z0_eff_norm          z0_eff already normalised by z0_scale
        [3] lat_norm             lat / lat_scale
        [4] hour_sin             sin(2π·hour/24)
        [5] hour_cos             cos(2π·hour/24)
        [6] month_sin            sin(2π·month/12)
        [7] month_cos            cos(2π·month/12)

    If `phys_features` is provided, it must contain 4 already-normalised values
    appended after the base 8 in this exact 12-dim order:
        [8]  grad_T_850_surf_n
        [9]  grad_T_500_850_n
        [10] RH_surface_n
        [11] q_surface_n

    No `distance_to_coast`: omitted as the M_H'0 brief allows engineer call —
    we keep the feature vector tight and physically meaningful for smoke. It
    can be added in M_H'1 if needed.
    """
    i0 = max(0, I_CENTER - patch_half)
    i1 = min(NI, I_CENTER + patch_half)
    j0 = max(0, J_CENTER - patch_half)
    j1 = min(NJ, J_CENTER + patch_half)
    patch = terrain_raw[i0:i1, j0:j1]
    mean_topo = float(patch.mean())
    std_topo = float(patch.std())

    ts = np.datetime64(timestamp_iso).astype("datetime64[h]")
    hour = int(str(ts)[-2:]) if "T" in str(ts) else 0
    # Robust hour/month extraction from ISO:
    ts_full = np.datetime64(timestamp_iso)
    dt = ts_full.astype("datetime64[s]").astype(object)
    hour = dt.hour
    month = dt.month

    base = np.array(
        [
            mean_topo / max(norm.get("terrain_scale", 500.0), 1.0),
            std_topo / max(norm.get("terrain_scale", 500.0), 1.0),
            z0_eff / max(norm.get("z0_scale", 1.0), 1e-6),
            lat / max(norm.get("lat_scale", 90.0), 1.0),
            math.sin(2.0 * math.pi * hour / 24.0),
            math.cos(2.0 * math.pi * hour / 24.0),
            math.sin(2.0 * math.pi * month / 12.0),
            math.cos(2.0 * math.pi * month / 12.0),
        ],
        dtype=np.float32,
    )
    if phys_features is None:
        return base
    phys = np.asarray(phys_features, dtype=np.float32).reshape(-1)
    if phys.size != 4:
        raise ValueError(f"phys_features must have length 4, got {phys.size}")
    phys = np.where(np.isfinite(phys), phys, np.float32(0.0)).astype(np.float32)
    return np.concatenate([base, phys]).astype(np.float32)


# ─── Grid.zarr → normalised tensors (same as WindV2DatasetViT) ───────────────


def _build_features_from_grid_zarr(
    grid_zarr_path: Path,
    norm: dict,
    target_agl_levels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Load a grid.zarr and produce (terrain_2d, era5_flat, geo, levels, z0_eff_raw, lat_raw).

    Reproduces WindV2DatasetViT.__getitem__ logic for input-side fields (no
    target). Returns the un-normalised z0_eff/lat as well so topo_features can
    use the same values.
    """
    g = zarr.open_group(str(grid_zarr_path), mode="r")

    terrain_raw = np.asarray(g["input/terrain"][:], dtype=np.float32)
    terrain = terrain_raw / norm["terrain_scale"]
    z0_eff_raw = float(g["input"].attrs.get("z0_eff", 0.0))
    z0_eff_norm = z0_eff_raw / norm["z0_scale"]
    z0_map = np.full((NI, NJ), z0_eff_norm, dtype=np.float32)
    # 4 channels matching surrogate v2 base (trained with include_slopes=True):
    # [terrain_norm, z0_broadcast_norm, slope_x, slope_y]. dx=dy=33.333 m per pixel.
    slope_y, slope_x = np.gradient(terrain_raw, 33.333, 33.333)
    terrain_2d = np.stack(
        [terrain.astype(np.float32), z0_map,
         slope_x.astype(np.float32), slope_y.astype(np.float32)],
        axis=0,
    )

    # geo on target AGL levels
    levels = target_agl_levels.astype(np.float32)
    agl = np.broadcast_to(levels[None, None, :], (NI, NJ, levels.size)).copy()
    z = terrain_raw[:, :, None] + agl
    geo = np.stack(
        [z / norm["z_scale"], agl / norm["agl_scale"]], axis=0
    ).astype(np.float32)

    # era5_flat (408 components)
    plev = np.asarray(g["input/era5_pressure_levels"][:], dtype=np.float32)
    flat_parts: list[np.ndarray] = []
    for var, scale, offset in [
        ("u", norm["era5_u_scale"], norm["era5_u_offset"]),
        ("v", norm["era5_v_scale"], norm["era5_v_offset"]),
        ("T", norm["era5_T_scale"], norm["era5_T_offset"]),
        ("q", norm["era5_q_scale"], norm["era5_q_offset"]),
    ]:
        arr = np.asarray(g[f"input/era5_3d/{var}"][:], dtype=np.float32)
        flat_parts.append(((arr - offset) / scale).ravel())
    for var, scale, offset in [
        ("t2m", norm["t2m_scale"], norm["t2m_offset"]),
        ("d2m", norm["d2m_scale"], norm["d2m_offset"]),
        ("u10", norm["u10_scale"], norm["u10_offset"]),
        ("v10", norm["v10_scale"], norm["v10_offset"]),
    ]:
        arr = np.asarray(g[f"input/era5_surface/{var}"][:], dtype=np.float32)
        flat_parts.append(((arr - offset) / scale).ravel())
    flat_parts.append(
        ((plev - norm["pressure_offset"]) / norm["pressure_scale"]).astype(np.float32)
    )
    lat_raw = float(g["input"].attrs.get("lat", 0.0))
    lat_norm = lat_raw / norm["lat_scale"]
    flat_parts.append(np.array([lat_norm, z0_eff_norm], dtype=np.float32))
    era5_flat = np.concatenate(flat_parts).astype(np.float32)

    return terrain_2d, era5_flat, geo, levels, z0_eff_raw, lat_raw


# ─── Pairings ────────────────────────────────────────────────────────────────


@dataclass
class Pairing:
    station_id: str
    timestamp_iso: str
    lat: float
    lon: float
    elev: float
    height_obs: float
    speed_obs: float
    source: str
    grid_zarr_path: Path  # cached materialised grid.zarr
    weight: float = 1.0   # per-sample sampling weight (M_I7 per-pop reweighting)


# ─── Dataset ─────────────────────────────────────────────────────────────────


class ObsCenteredDataset(Dataset):
    """OBS-centered patch dataset for DEVINE-style training.

    Materialisation of grid.zarrs is done eagerly at construction time
    (parallelisable). Failed pairings are dropped with a warning.
    """

    def __init__(
        self,
        pairings_parquet: Path,
        *,
        era5_store: Path,
        dem: Path,
        worldcover: Path | None,
        cache_dir: Path,
        norm: dict | None = None,
        target_agl_levels: str = "agl_0_100_24",
        max_pairings: int | None = None,
        max_era5_delta_h: float = 3.5,
        station_filter: Iterable[str] | None = None,
        seed: int = 42,
        n_workers: int = 4,
        overwrite_cache: bool = False,
        require_cached: bool = False,
        enable_phys_features: bool = False,
    ) -> None:
        self.norm = {**DEFAULT_NORM, **(norm or {})}
        self.enable_phys_features = bool(enable_phys_features)
        self.target_agl_levels = parse_agl_levels(target_agl_levels)
        if self.target_agl_levels is None:
            raise ValueError("target_agl_levels must resolve to a non-None array")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_parquet(pairings_parquet)
        required_cols = {"station_id", "timestamp", "lat", "lon", "elev",
                         "speed_obs"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"pairings parquet missing columns: {missing}")
        if "height_obs" not in df.columns:
            # Backward compat: legacy 10 m parquets → fixed 10 m AGL level.
            df["height_obs"] = 10.0
            logger.info("No height_obs column: assuming 10 m for all %d rows",
                        len(df))

        if station_filter is not None:
            keep = set(station_filter)
            df = df[df["station_id"].isin(keep)].reset_index(drop=True)
        df = df.dropna(subset=["speed_obs", "lat", "lon", "height_obs"])
        df = df[df["speed_obs"] > 0.0].reset_index(drop=True)

        # M_I7: drop obs above the top AGL level of the frozen surrogate grid
        # (e.g. ICOS 120/131/180 m vs agl_0_100_24). Kept in the parquet for a
        # future v3 deep-crop retrain; unusable by the current k24<=100m head.
        agl_top = float(self.target_agl_levels[-1])
        above = df["height_obs"] > agl_top
        n_above = int(above.sum())
        if n_above > 0:
            logger.info("Filtered %d/%d pairings with height_obs > %.0f m "
                        "(reserved for v3)", n_above, len(df), agl_top)
            df = df[~above].reset_index(drop=True)

        if max_pairings is not None and len(df) > max_pairings:
            df = df.sample(n=max_pairings, random_state=seed).reset_index(drop=True)

        action = "Loading cached" if require_cached else "Materialising"
        logger.info("%s %d grid.zarrs from %s (n_workers=%d) ...",
                    action, len(df), self.cache_dir, n_workers)
        self.pairings = self._materialise_all(
            df,
            era5_store=Path(era5_store),
            dem=Path(dem),
            worldcover=Path(worldcover) if worldcover else None,
            max_era5_delta_h=max_era5_delta_h,
            n_workers=n_workers,
            overwrite=overwrite_cache,
            require_cached=require_cached,
        )
        logger.info("Dataset ready: %d usable pairings", len(self.pairings))

    @staticmethod
    def _cache_path(cache_dir: Path, station_id: str, ts_iso: str) -> Path:
        # Strip the multi-height suffix so all heights of one physical tower
        # share a single materialised grid.zarr (identical lat/lon → identical
        # inputs; only k_obs differs). Legacy ids are unaffected (no suffix).
        base_id = _HEIGHT_SUFFIX_RE.sub("", station_id)
        ts_tag = ts_iso.replace(":", "").replace("-", "")[:13]
        return cache_dir / f"{base_id}_{ts_tag}" / "grid.zarr"

    def _materialise_all(
        self,
        df: pd.DataFrame,
        *,
        era5_store: Path,
        dem: Path,
        worldcover: Path | None,
        max_era5_delta_h: float,
        n_workers: int,
        overwrite: bool,
        require_cached: bool,
    ) -> list[Pairing]:
        build_one = None
        if not require_cached:
            # Import M_G6 builder lazily so this module can be byte-compiled without
            # rasterio/pyproj on systems where they're absent.
            import sys
            scripts_dir = Path(__file__).resolve().parents[1]
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))
            from extract_v2_input_at_coords import build_one as _build_one  # noqa: E402

            build_one = _build_one

        ok: list[Pairing] = []
        n_err = 0
        n_missing_cached = 0

        # Sequential materialisation (multiprocessing build_one needs care with
        # rasterio file handles; n_workers=4 default but we keep sequential here
        # for simplicity. Cache hits are O(1) so a 2nd epoch is fast).
        for row in df.itertuples(index=False):
            sid = str(row.station_id)
            ts = str(row.timestamp)
            # Normalize pandas timestamp to ISO
            if not isinstance(ts, str):
                ts = pd.Timestamp(ts).isoformat()
            # Some pandas parquet reads return ns timestamps as numpy.datetime64
            try:
                ts_iso = pd.Timestamp(ts).isoformat()
            except Exception:
                ts_iso = ts
            cache_path = self._cache_path(self.cache_dir, sid, ts_iso)
            if require_cached:
                if not cache_path.exists():
                    n_missing_cached += 1
                    continue
            elif not cache_path.exists() or overwrite:
                assert build_one is not None
                # Per-row ERA5 store override (M_I7 multi-season merged
                # parquets carry an `era5_store` column); default otherwise.
                row_store = getattr(row, "era5_store", None)
                store = Path(str(row_store)) if row_store else era5_store
                try:
                    build_one(
                        site_id=sid,
                        lat=float(row.lat),
                        lon=float(row.lon),
                        timestamp_iso=ts_iso,
                        era5_store=store,
                        dem=dem,
                        worldcover=worldcover,
                        output=cache_path,
                        overwrite=overwrite,
                        max_era5_delta_h=max_era5_delta_h,
                        extra_meta={"station_elev": float(row.elev),
                                    "height_obs": float(row.height_obs)},
                    )
                except Exception as exc:
                    n_err += 1
                    logger.debug("materialise FAIL %s @ %s: %s", sid, ts_iso, exc)
                    continue
            if not cache_path.exists():
                n_err += 1
                continue
            ok.append(
                Pairing(
                    station_id=sid,
                    timestamp_iso=ts_iso,
                    lat=float(row.lat),
                    lon=float(row.lon),
                    elev=float(row.elev),
                    height_obs=float(row.height_obs),
                    speed_obs=float(row.speed_obs),
                    source=str(getattr(row, "source", "")),
                    grid_zarr_path=cache_path,
                    weight=float(getattr(row, "sample_weight", 1.0) or 1.0),
                )
            )
        if require_cached:
            logger.info("require_cached=True: kept=%d dropped=%d of %d",
                        len(ok), n_missing_cached, len(df))
        if n_err > 0:
            logger.warning("materialise: %d/%d pairings failed (kept %d)",
                           n_err, len(df), len(ok))
        return ok

    def __len__(self) -> int:
        return len(self.pairings)

    @property
    def sample_weights(self) -> list[float]:
        """Per-pairing sampling weights (all 1.0 unless the parquet carries a
        `sample_weight` column, e.g. the M_I7 merged multi-pop parquet)."""
        return [p.weight for p in self.pairings]

    def __getitem__(self, idx: int):
        p = self.pairings[idx]
        terrain_2d, era5_flat, geo, levels, z0_eff_raw, lat_raw = (
            _build_features_from_grid_zarr(p.grid_zarr_path, self.norm,
                                           self.target_agl_levels)
        )
        # k_obs: index of nearest AGL level to height_obs (typically 10 m)
        k_obs = int(np.argmin(np.abs(levels - float(p.height_obs))))

        # Reload terrain_raw for topo_features (cheap, same zarr)
        g = zarr.open_group(str(p.grid_zarr_path), mode="r")
        terrain_raw = np.asarray(g["input/terrain"][:], dtype=np.float32)
        if self.enable_phys_features:
            phys = compute_phys_features(g, self.norm)
            topo = compute_topo_features(
                terrain_raw, z0_eff_raw, lat_raw, p.timestamp_iso, self.norm,
                phys_features=phys,
            )
        else:
            topo = compute_topo_features(
                terrain_raw, z0_eff_raw, lat_raw, p.timestamp_iso, self.norm,
            )

        return (
            torch.from_numpy(terrain_2d),
            torch.from_numpy(era5_flat),
            torch.from_numpy(geo),
            torch.from_numpy(topo),
            torch.tensor(float(p.speed_obs), dtype=torch.float32),
            torch.tensor(int(k_obs), dtype=torch.long),
            {"station_id": p.station_id,
             "timestamp_iso": p.timestamp_iso,
             "source": p.source,
             "height_obs": float(p.height_obs)},
        )


def collate_obs_centered(batch: Sequence) -> tuple:
    """Default collate that stacks tensors and keeps `meta` as a list."""
    terrain = torch.stack([b[0] for b in batch], dim=0)
    era5 = torch.stack([b[1] for b in batch], dim=0)
    geo = torch.stack([b[2] for b in batch], dim=0)
    topo = torch.stack([b[3] for b in batch], dim=0)
    speed = torch.stack([b[4] for b in batch], dim=0)
    k_obs = torch.stack([b[5] for b in batch], dim=0)
    meta = [b[6] for b in batch]
    return terrain, era5, geo, topo, speed, k_obs, meta


# ─── Watertight station split ────────────────────────────────────────────────


def watertight_station_split(
    pairings_parquet: Path,
    *,
    val_frac: float = 0.20,
    seed: int = 42,
    exclude_substrings: Sequence[str] = ("perdigao",),
) -> tuple[list[str], list[str]]:
    """Split stations into train/val by station_id, no leakage.

    Returns (train_station_ids, val_station_ids). Excludes any station whose id
    contains a forbidden substring (Perdigão is reserved for M_H'1 IOP test).

    Multi-height ids (`<tower>_hNNN`) are grouped by their physical tower so
    all heights of one tower land in the SAME split (no height leakage).
    """
    df = pd.read_parquet(pairings_parquet, columns=["station_id"])
    sids = sorted(set(df["station_id"].astype(str)))
    for sub in exclude_substrings:
        sids = [s for s in sids if sub.lower() not in s.lower()]
    groups: dict[str, list[str]] = {}
    for s in sids:
        groups.setdefault(_HEIGHT_SUFFIX_RE.sub("", s), []).append(s)
    rng = np.random.default_rng(seed)
    keys = np.array(sorted(groups))
    rng.shuffle(keys)
    n_val = max(1, int(round(len(keys) * val_frac)))
    val = [s for k in keys[:n_val] for s in groups[k]]
    train = [s for k in keys[n_val:] for s in groups[k]]
    return train, val
