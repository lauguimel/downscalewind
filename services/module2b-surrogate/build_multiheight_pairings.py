"""
build_multiheight_pairings.py — M_I7 palier A multi-height pairings assembler.

Builds a multi-HEIGHT observation pairings parquet for training a vertical
correction network, from two local sources:

1. ICOS tower zarrs (data/raw/icos_{ope,ipr,hpb,sac,trn}.zarr) — ws/wd per
   height (meteo/ws_<H>m, meteo/wd_<H>m). u,v derived with the meteorological
   convention: u = -ws*sin(wd*pi/180), v = -ws*cos(wd*pi/180). Heights <10 m
   are ignored; heights >100 m are KEPT (future v3). If wd is missing at a
   height for a given timestamp but present at the nearest height, that wd is
   borrowed (flagged in `wd_borrowed`).

2. Perdigão masts (data/raw/perdigao_obs.zarr) — u,v components at 14 heights
   (sites/u, sites/v [time, station, height]); all heights >=10 m kept, only
   on-the-hour timestamps retained (native 30-min IOP 2017).

QC: drop NaN u/v/speed, speed_obs <= 0 or > 60 m/s.

Usage:
    conda run -n downscalewind python services/module2b-surrogate/build_multiheight_pairings.py
"""
from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import pandas as pd
import zarr

REPO = Path(__file__).resolve().parents[2]

# ICOS station metadata — lat/lon/elev from services/data-ingestion/ingest_icos.py
# (STATIONS dict, "AS" tall towers). Values for HPB/SAC/TRN cross-checked against
# authoritative ICOS metadata (identical).
ICOS_META = {
    "ope": {"id": "OPE", "lat": 48.5619, "lon": 5.5036, "elev": 395.0,
            "heights": [10.0, 50.0, 120.0]},
    "ipr": {"id": "IPR", "lat": 45.8126, "lon": 8.6360, "elev": 210.0,
            "heights": [40.0, 60.0, 100.0]},
    "hpb": {"id": "HPB", "lat": 47.8011, "lon": 11.0246, "elev": 934.0,
            "heights": [50.0, 93.0, 131.0]},
    "sac": {"id": "SAC", "lat": 48.7227, "lon": 2.1420, "elev": 160.0,
            "heights": [10.0, 60.0, 100.0]},
    "trn": {"id": "TRN", "lat": 47.9647, "lon": 2.1125, "elev": 131.0,
            "heights": [50.0, 100.0, 180.0]},
}

MIN_HEIGHT_M = 10.0
SPEED_MAX = 60.0
# Common JJA2020 window (SAC store runs to 2021-05, TRN to 2020-09-20; clip so
# the season="jja2020" tag matches the ERA5 store used downstream).
ICOS_T0 = np.datetime64("2020-06-01T00:00:00")
ICOS_T1 = np.datetime64("2020-09-09T23:00:00")


def _uv_from_wswd(ws: np.ndarray, wd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Meteorological convention: wd = direction wind comes FROM."""
    rad = np.deg2rad(wd)
    return -ws * np.sin(rad), -ws * np.cos(rad)


def build_icos(raw_dir: Path) -> pd.DataFrame:
    frames = []
    for site, meta in ICOS_META.items():
        store = raw_dir / f"icos_{site}.zarr"
        g = zarr.open_group(str(store), mode="r")
        time = pd.to_datetime(g["coords/time"][:])
        in_win = (time >= ICOS_T0) & (time <= ICOS_T1)
        time = time[in_win]
        heights = [h for h in meta["heights"] if h >= MIN_HEIGHT_M]
        arrays = set(g["meteo"].array_keys())

        # Preload ws/wd per height (NaN column if array absent)
        ws_by_h, wd_by_h = {}, {}
        for h in heights:
            suf = f"{int(h)}m"
            ws_by_h[h] = (g[f"meteo/ws_{suf}"][:][in_win].astype(np.float64)
                          if f"ws_{suf}" in arrays else np.full(len(time), np.nan))
            wd_by_h[h] = (g[f"meteo/wd_{suf}"][:][in_win].astype(np.float64)
                          if f"wd_{suf}" in arrays else np.full(len(time), np.nan))

        for h in heights:
            ws, wd = ws_by_h[h], wd_by_h[h].copy()
            borrowed = np.zeros(len(time), dtype=bool)
            # Borrow wd from nearest height where wd missing but ws present
            need = np.isnan(wd) & ~np.isnan(ws)
            if need.any():
                for h2 in sorted((x for x in heights if x != h),
                                 key=lambda x: abs(x - h)):
                    donor = wd_by_h[h2]
                    take = need & ~np.isnan(donor)
                    if take.any():
                        wd[take] = donor[take]
                        borrowed[take] = True
                        need &= np.isnan(wd)
                    if not need.any():
                        break
            u, v = _uv_from_wswd(ws, wd)
            frames.append(pd.DataFrame({
                "station_id": f"icos_{meta['id']}_h{int(h):03d}",
                "timestamp": time,
                "lat": meta["lat"], "lon": meta["lon"], "elev": meta["elev"],
                "height_obs": h,
                "u_obs": u.astype(np.float32),
                "v_obs": v.astype(np.float32),
                "speed_obs": ws.astype(np.float32),
                "season": "jja2020", "pop": "tower_icos", "source": "icos",
                "wd_borrowed": borrowed,
            }))
    return pd.concat(frames, ignore_index=True)


def build_perdigao(raw_dir: Path) -> pd.DataFrame:
    g = zarr.open_group(str(raw_dir / "perdigao_obs.zarr"), mode="r")
    time = pd.to_datetime(g["coords/time"][:])
    heights = g["coords/height_m"][:].astype(np.float64)
    click.echo(f"Perdigao height_m values: {heights.tolist()}")
    lat = g["coords/lat"][:]
    lon = g["coords/lon"][:]
    alt = g["coords/altitude_m"][:]
    site_ids = [s.decode() if isinstance(s, bytes) else str(s)
                for s in g["coords/site_id"][:]]
    u = g["sites/u"][:]  # [time, station, height]
    v = g["sites/v"][:]

    # Native 30-min samples are window-centre labelled (:02:30 / :32:30).
    # Keep the on-the-hour slots (offset < 15 min) and snap them to the hour.
    offset_s = time.minute * 60 + time.second
    on_hour = offset_s < 900
    time = time[on_hour].floor("h")
    u, v = u[on_hour], v[on_hour]

    keep_h = np.where(heights >= MIN_HEIGHT_M)[0]
    frames = []
    for si, sid in enumerate(site_ids):
        for hi in keep_h:
            uu = u[:, si, hi].astype(np.float32)
            vv = v[:, si, hi].astype(np.float32)
            if np.all(np.isnan(uu)):
                continue
            frames.append(pd.DataFrame({
                "station_id": f"perdigao_{sid}_h{int(heights[hi]):03d}",
                "timestamp": time,
                "lat": float(lat[si]), "lon": float(lon[si]),
                "elev": float(alt[si]),
                "height_obs": float(heights[hi]),
                "u_obs": uu, "v_obs": vv,
                "speed_obs": np.hypot(uu, vv).astype(np.float32),
                "season": "iop2017", "pop": "perdigao_mast",
                "source": "perdigao", "wd_borrowed": False,
            }))
    return pd.concat(frames, ignore_index=True)


def apply_qc(df: pd.DataFrame, label: str) -> pd.DataFrame:
    n0 = len(df)
    ok = (df["u_obs"].notna() & df["v_obs"].notna() & df["speed_obs"].notna()
          & (df["speed_obs"] > 0) & (df["speed_obs"] <= SPEED_MAX))
    out = df[ok].reset_index(drop=True)
    click.echo(f"QC {label}: {n0} -> {len(out)} rows (dropped {n0 - len(out)})")
    return out


@click.command()
@click.option("--raw-dir", type=click.Path(exists=True, path_type=Path),
              default=REPO / "data/raw", show_default=True)
@click.option("--output", type=click.Path(path_type=Path),
              default=REPO / "data/inference/multiheight_towers_v1.parquet",
              show_default=True)
def main(raw_dir: Path, output: Path) -> None:
    """Build the multi-height towers pairings parquet (M_I7 palier A)."""
    icos = apply_qc(build_icos(raw_dir), "icos")
    perd = apply_qc(build_perdigao(raw_dir), "perdigao")

    df = pd.concat([icos, perd], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for c in ("lat", "lon", "elev", "height_obs"):
        df[c] = df[c].astype(np.float64)
    for c in ("u_obs", "v_obs", "speed_obs"):
        df[c] = df[c].astype(np.float32)
    df["wd_borrowed"] = df["wd_borrowed"].astype(bool)
    df = df.sort_values(["pop", "station_id", "timestamp"]).reset_index(drop=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    click.echo(f"Wrote {len(df)} rows -> {output}")

    # Summary
    summ = (df.groupby(["pop", "height_obs"])
              .agg(rows=("speed_obs", "size"),
                   stations=("station_id", "nunique"),
                   t_min=("timestamp", "min"), t_max=("timestamp", "max"),
                   speed_mean=("speed_obs", "mean"),
                   speed_max=("speed_obs", "max"),
                   wd_borrowed=("wd_borrowed", "sum")))
    click.echo(summ.to_string())


if __name__ == "__main__":
    main()
