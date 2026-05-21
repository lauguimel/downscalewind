"""
ingest_synop_meteofr.py - Ingestion SYNOP Meteo-France pour DownscaleWind.

Telecharge les archives mensuelles SYNOP "essentielles OMM" de Meteo-France,
parse les CSV.gz, filtre les postes de France metropolitaine, puis ecrit un
store Zarr au schema observations unifie.

Schema cible (mandate §7):

```
data/raw/obs_unified_synop_fr.zarr/
  stations/
    station_id  (S,)   bytes (S16)        e.g. b"synop_07005"
    lat, lon, elev      float32 (S,)
    source              (S,) bytes (S16)  = b"synop_fr" (constant)
    country             (S,) bytes (S2)   = b"FR"
    z0_class_wc         int8 (S,)         = -1 (not computed in M_G2)
  heights/
    height_m            float32 (H,)      = [10.0]  (H=1 for SYNOP)
  data/  chunks=(time=720, S=1, H=-1)
    u                   float32 (T, S, H)  m/s, NaN-padded
    v                   float32 (T, S, H)  m/s, NaN-padded
    wind_speed          float32 (T, S, H)  m/s, NaN-padded
    wind_dir            float32 (T, S, H)  degrees, NaN-padded
    t2m                 float32 (T, S, H)  K, NaN-padded
    rh                  float32 (T, S, H)  %, NaN-padded
  coords/
    time                int64 (T,) ns UTC hourly  (NaN-pad rows for non-3h slots)

attrs.global:
  sources             ["synop_fr"]
  n_stations          int
  time_range          [start_iso, end_iso]
  native_cadence_h    3
  resample_cadence_h  1
  resample_method     "nan_pad"
  created_at          ISO string
  schema_version      "1.0"
  source_url          "https://donneespubliques.meteofrance.fr/..."
```

Usage:
    python services/data-ingestion/ingest_synop_meteofr.py \\
      --out data/raw/obs_unified_synop_fr.zarr \\
      --start 2023-01 --end 2023-03 --smoke --overwrite
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import click
import numpy as np
import pandas as pd
import zarr
from zarr.codecs import BloscCodec

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.data_io import wind_speed_direction_to_uv
from shared.logging_config import get_logger
from utils.synop_parser import (
    FetchResult,
    POSTES_URL,
    SYNOP_ARCHIVE_URL,
    Stations,
    fetch_with_cache,
    load_mainland_stations,
    load_month,
)

log = get_logger("ingest_synop_meteofr")

DATA_COMPRESSOR = BloscCodec(cname="lz4", clevel=5, shuffle="bitshuffle")


def _parse_month(value: str) -> pd.Period:
    try:
        period = pd.Period(value, freq="M")
    except ValueError as exc:
        raise click.BadParameter("expected YYYY-MM") from exc
    if str(period) != value:
        raise click.BadParameter("expected YYYY-MM")
    return period


def _iter_months(start: pd.Period, end: pd.Period) -> list[str]:
    if start > end:
        raise click.BadParameter("--start must be <= --end")
    months: list[str] = []
    cur = start
    while cur <= end:
        months.append(f"{cur.year:04d}{cur.month:02d}")
        cur += 1
    return months


def _month_time_bounds(start: pd.Period, end: pd.Period) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_ts = start.start_time.tz_localize("UTC")
    end_ts = (end + 1).start_time.tz_localize("UTC") - pd.Timedelta(hours=1)
    return start_ts, end_ts


def _temperature_to_kelvin(t_values: np.ndarray) -> tuple[np.ndarray, str]:
    """Return K. The current mission spec says Celsius; some MF files are K."""
    out = t_values.astype(np.float32)
    finite = out[np.isfinite(out)]
    if finite.size and float(np.nanmedian(finite)) > 150.0:
        return out, "already_kelvin"
    return (out + np.float32(273.15)).astype(np.float32), "celsius_to_kelvin"


def _build_hourly_arrays(
    df: pd.DataFrame,
    stations: Stations,
    start: pd.Period,
    end: pd.Period,
) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray], str]:
    start_ts, end_ts = _month_time_bounds(start, end)
    times = pd.date_range(start_ts, end_ts, freq="h", tz="UTC")
    n_times = len(times)
    n_stations = len(stations.ids)
    shape = (n_times, n_stations, 1)
    arrays = {
        "u": np.full(shape, np.nan, dtype=np.float32),
        "v": np.full(shape, np.nan, dtype=np.float32),
        "wind_speed": np.full(shape, np.nan, dtype=np.float32),
        "wind_dir": np.full(shape, np.nan, dtype=np.float32),
        "t2m": np.full(shape, np.nan, dtype=np.float32),
        "rh": np.full(shape, np.nan, dtype=np.float32),
    }

    if df.empty:
        return times, arrays, "no_temperature_data"

    df = df.sort_values(["timestamp", "numer_sta"]).drop_duplicates(
        ["timestamp", "numer_sta"],
        keep="last",
    )
    station_pos = {sid: idx for idx, sid in enumerate(stations.ids)}
    time_pos = times.get_indexer(df["timestamp"])
    station_idx = df["numer_sta"].map(station_pos).to_numpy(dtype=np.int64)
    valid = (time_pos >= 0) & (station_idx >= 0)
    if not valid.any():
        return times, arrays, "no_temperature_data"

    time_pos = time_pos[valid]
    station_idx = station_idx[valid]
    speed = df.loc[valid, "ff"].to_numpy(dtype=np.float32)
    direction = df.loc[valid, "dd"].to_numpy(dtype=np.float32)
    temp_native = df.loc[valid, "t"].to_numpy(dtype=np.float32)
    rh = df.loc[valid, "u"].to_numpy(dtype=np.float32)

    u_comp, v_comp = wind_speed_direction_to_uv(speed, direction)
    temp_k, temp_conversion = _temperature_to_kelvin(temp_native)

    arrays["u"][time_pos, station_idx, 0] = u_comp
    arrays["v"][time_pos, station_idx, 0] = v_comp
    arrays["wind_speed"][time_pos, station_idx, 0] = speed
    arrays["wind_dir"][time_pos, station_idx, 0] = direction
    arrays["t2m"][time_pos, station_idx, 0] = temp_k
    arrays["rh"][time_pos, station_idx, 0] = rh
    return times, arrays, temp_conversion


def _nan_percent(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        name: round(float(np.mean(~np.isfinite(values)) * 100.0), 3)
        for name, values in arrays.items()
    }


def _array_attrs(name: str) -> dict[str, str]:
    attrs = {
        "u": {"long_name": "eastward wind component", "units": "m s-1"},
        "v": {"long_name": "northward wind component", "units": "m s-1"},
        "wind_speed": {"long_name": "10 m wind speed", "units": "m s-1"},
        "wind_dir": {"long_name": "wind direction from north", "units": "degrees"},
        "t2m": {"long_name": "2 m air temperature", "units": "K"},
        "rh": {"long_name": "relative humidity", "units": "%"},
    }
    return attrs[name]


def _write_zarr(
    out_path: Path,
    stations: Stations,
    times: pd.DatetimeIndex,
    arrays: dict[str, np.ndarray],
    months: list[str],
    station_fetch: FetchResult,
    monthly_fetches: list[FetchResult],
    overwrite: bool,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "w-"
    root = zarr.open_group(str(out_path), mode=mode)

    n_times = len(times)
    n_stations = len(stations.ids)
    data_chunks = (min(720, n_times), 1, 1)

    stations_grp = root.require_group("stations")
    station_ids = np.array([f"synop_{sid}".encode("ascii") for sid in stations.ids], dtype="S16")
    source = np.array([b"synop_fr"] * n_stations, dtype="S16")
    country = np.array([b"FR"] * n_stations, dtype="S2")

    arr = stations_grp.create_array("station_id", shape=(n_stations,), dtype="S16",
                                    chunks=(n_stations,), overwrite=True)
    arr[:] = station_ids
    arr.attrs.update({"long_name": "station identifier"})
    for name, values, attrs in (
        ("lat", stations.lat, {"long_name": "latitude", "units": "degrees_north"}),
        ("lon", stations.lon, {"long_name": "longitude", "units": "degrees_east"}),
        ("elev", stations.elev, {"long_name": "station elevation", "units": "m"}),
    ):
        arr = stations_grp.create_array(name, shape=(n_stations,), dtype=np.float32,
                                        chunks=(n_stations,), overwrite=True)
        arr[:] = values
        arr.attrs.update(attrs)
    arr = stations_grp.create_array("source", shape=(n_stations,), dtype="S16",
                                    chunks=(n_stations,), overwrite=True)
    arr[:] = source
    arr.attrs.update({"long_name": "observation source"})
    arr = stations_grp.create_array("country", shape=(n_stations,), dtype="S2",
                                    chunks=(n_stations,), overwrite=True)
    arr[:] = country
    arr.attrs.update({"long_name": "country code"})
    arr = stations_grp.create_array("z0_class_wc", shape=(n_stations,), dtype=np.int8,
                                    chunks=(n_stations,), overwrite=True)
    arr[:] = np.full(n_stations, -1, dtype=np.int8)
    arr.attrs.update({"long_name": "ESA WorldCover roughness class", "flag_values": [-1]})

    heights_grp = root.require_group("heights")
    arr = heights_grp.create_array("height_m", shape=(1,), dtype=np.float32,
                                   chunks=(1,), overwrite=True)
    arr[:] = np.array([10.0], dtype=np.float32)
    arr.attrs.update({"long_name": "height above ground", "units": "m"})

    data_grp = root.require_group("data")
    for name, values in arrays.items():
        arr = data_grp.create_array(
            name,
            shape=values.shape,
            dtype=np.float32,
            chunks=data_chunks,
            compressors=DATA_COMPRESSOR,
            overwrite=True,
        )
        arr[...] = values
        arr.attrs.update(_array_attrs(name))

    coords_grp = root.require_group("coords")
    arr = coords_grp.create_array("time", shape=(n_times,), dtype=np.int64,
                                  chunks=(min(720, n_times),), overwrite=True)
    arr[:] = times.asi8.astype(np.int64)
    arr.attrs.update({"long_name": "time UTC", "units": "ns since epoch"})

    root.attrs.update({
        "sources": ["synop_fr"],
        "n_stations": n_stations,
        "time_range": [times[0].isoformat(), times[-1].isoformat()],
        "native_cadence_h": 3,
        "resample_cadence_h": 1,
        "resample_method": "nan_pad",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": "1.0",
        "source_url": "https://donneespubliques.meteofrance.fr/...",
        "archive_url_template": SYNOP_ARCHIVE_URL,
        "station_list_url": POSTES_URL,
        "months": months,
        "station_list_cache_status": station_fetch.status,
        "months_downloaded": [f.path.name for f in monthly_fetches if f.status == "downloaded"],
        "months_cached": [f.path.name for f in monthly_fetches if f.status == "cached"],
    })


@click.command()
@click.option("--out", "out_path", required=True, type=click.Path(path_type=Path),
              help="Output Zarr store path")
@click.option("--start", required=True, help="Start month, inclusive (YYYY-MM)")
@click.option("--end", required=True, help="End month, inclusive (YYYY-MM)")
@click.option("--smoke", is_flag=True,
              help="Allow >=1 successful month and require >=50 stations")
@click.option("--cache-dir", default="tmp/synop_cache", show_default=True,
              type=click.Path(path_type=Path),
              help="Local cache for postesSynop.csv and monthly CSV.gz archives")
@click.option("--overwrite/--no-overwrite", default=False, show_default=True,
              help="Overwrite output Zarr store if it exists")
def main(
    out_path: Path,
    start: str,
    end: str,
    smoke: bool,
    cache_dir: Path,
    overwrite: bool,
) -> None:
    """Download, parse, and write Meteo-France SYNOP observations."""
    start_period = _parse_month(start)
    end_period = _parse_month(end)
    months = _iter_months(start_period, end_period)

    if out_path.exists() and not overwrite:
        raise click.ClickException(f"Output exists; pass --overwrite: {out_path}")

    log.info("Starting SYNOP ingestion", extra={
        "out": str(out_path),
        "start": start,
        "end": end,
        "months": months,
        "smoke": smoke,
        "cache_dir": str(cache_dir),
        "overwrite": overwrite,
    })

    stations, station_fetch = load_mainland_stations(cache_dir)
    if smoke and len(stations.ids) < 50:
        raise click.ClickException(f"Smoke requires >=50 stations, got {len(stations.ids)}")

    frames: list[pd.DataFrame] = []
    monthly_fetches: list[FetchResult] = []
    failed_months: list[str] = []
    station_id_set = set(stations.ids)
    for yyyymm in months:
        url = SYNOP_ARCHIVE_URL.format(yyyymm=yyyymm)
        cache_path = cache_dir / f"synop.{yyyymm}.csv.gz"
        try:
            fetch = fetch_with_cache(url, cache_path)
            monthly_fetches.append(fetch)
            month_df = load_month(fetch.path, station_id_set)
            frames.append(month_df)
            log.info("Parsed SYNOP month", extra={
                "month": yyyymm,
                "status": fetch.status,
                "n_rows": len(month_df),
                "n_stations_with_rows": (
                    int(month_df["numer_sta"].nunique()) if not month_df.empty else 0
                ),
            })
        except Exception as exc:
            failed_months.append(yyyymm)
            if not smoke:
                raise
            log.warning("Skipping failed smoke month", extra={
                "month": yyyymm,
                "error": str(exc),
            })

    if not frames or all(frame.empty for frame in frames):
        raise click.ClickException("No SYNOP rows parsed for requested period")
    if smoke and len(monthly_fetches) < 1:
        raise click.ClickException("Smoke requires at least one successful month")
    if failed_months and not smoke:
        raise click.ClickException(f"Failed months: {failed_months}")

    obs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    times, arrays, temp_conversion = _build_hourly_arrays(obs, stations, start_period, end_period)
    valid_ws = int(np.sum(np.isfinite(arrays["wind_speed"])))
    if smoke and valid_ws == 0:
        raise click.ClickException("Smoke produced all-NaN wind_speed")

    _write_zarr(
        out_path=out_path,
        stations=stations,
        times=times,
        arrays=arrays,
        months=months,
        station_fetch=station_fetch,
        monthly_fetches=monthly_fetches,
        overwrite=overwrite,
    )

    summary = {
        "output": str(out_path),
        "n_stations": len(stations.ids),
        "n_times": len(times),
        "time_range": [times[0].isoformat(), times[-1].isoformat()],
        "n_valid_wind_speed": valid_ws,
        "nan_pct": _nan_percent(arrays),
        "months_downloaded": [f.path.name for f in monthly_fetches if f.status == "downloaded"],
        "months_cached": [f.path.name for f in monthly_fetches if f.status == "cached"],
        "failed_months": failed_months,
        "temperature_conversion": temp_conversion,
    }
    log.info("SYNOP ingestion complete", extra=summary)
    print(
        "Done. Zarr store: "
        f"{out_path} ({len(stations.ids)} stations, {len(times)} hourly steps, "
        f"{valid_ws} valid wind_speed values)"
    )


if __name__ == "__main__":
    main()
