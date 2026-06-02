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
    topo_features   (8,)                 mean_topo, std_topo, z0_eff, lat_norm,
                                          hour_sin, hour_cos, month_sin, month_cos
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


# ─── Topo features (8 components) ────────────────────────────────────────────


def compute_topo_features(
    terrain_raw: np.ndarray,
    z0_eff: float,
    lat: float,
    timestamp_iso: str,
    norm: dict,
    patch_half: int = 15,
) -> np.ndarray:
    """Build the 8-dim topo feature vector for one pairing.

    Components (all roughly O(1) after normalisation):
        [0] mean_topo_local      mean elevation in 30×30 voxels around centre (m), scaled
        [1] std_topo_local       std of elevation in same patch (m), scaled
        [2] z0_eff_norm          z0_eff already normalised by z0_scale
        [3] lat_norm             lat / lat_scale
        [4] hour_sin             sin(2π·hour/24)
        [5] hour_cos             cos(2π·hour/24)
        [6] month_sin            sin(2π·month/12)
        [7] month_cos            cos(2π·month/12)

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

    return np.array(
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
    ) -> None:
        self.norm = {**DEFAULT_NORM, **(norm or {})}
        self.target_agl_levels = parse_agl_levels(target_agl_levels)
        if self.target_agl_levels is None:
            raise ValueError("target_agl_levels must resolve to a non-None array")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_parquet(pairings_parquet)
        required_cols = {"station_id", "timestamp", "lat", "lon", "elev",
                         "height_obs", "speed_obs"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"pairings parquet missing columns: {missing}")

        if station_filter is not None:
            keep = set(station_filter)
            df = df[df["station_id"].isin(keep)].reset_index(drop=True)
        df = df.dropna(subset=["speed_obs", "lat", "lon", "height_obs"])
        df = df[df["speed_obs"] > 0.0].reset_index(drop=True)

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
        ts_tag = ts_iso.replace(":", "").replace("-", "")[:13]
        return cache_dir / f"{station_id}_{ts_tag}" / "grid.zarr"

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
                try:
                    build_one(
                        site_id=sid,
                        lat=float(row.lat),
                        lon=float(row.lon),
                        timestamp_iso=ts_iso,
                        era5_store=era5_store,
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
    """
    df = pd.read_parquet(pairings_parquet, columns=["station_id"])
    sids = sorted(set(df["station_id"].astype(str)))
    for sub in exclude_substrings:
        sids = [s for s in sids if sub.lower() not in s.lower()]
    rng = np.random.default_rng(seed)
    sids_arr = np.array(sids)
    rng.shuffle(sids_arr)
    n_val = max(1, int(round(len(sids_arr) * val_frac)))
    val = sids_arr[:n_val].tolist()
    train = sids_arr[n_val:].tolist()
    return train, val
