"""OGIMET decoded SYNOP archive ingestion for Portugal Phase G observations."""

from __future__ import annotations

import calendar
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode

import click
import numpy as np
import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.logging_config import get_logger
from utils.ogimet_parser import parse_ogimet_html, relative_humidity_from_dewpoint
from utils.obs_zarr_writer import (
    StationRecord,
    ensure_obs_store,
    read_station_ids,
    wind_to_uv,
    write_station_timeseries,
)


log = get_logger("ingest_ogimet")
OGIMET_URL = "http://www.ogimet.com/cgi-bin/gsynres"
LAST_GET_TS: float | None = None


@dataclass(frozen=True)
class WmoStation:
    wmo_id: str
    name: str
    lat: float
    lon: float
    elev: float

    def record(self) -> StationRecord:
        return StationRecord(station_id=f"ogimet_{self.wmo_id}", lat=self.lat, lon=self.lon, elev=self.elev)


PT_WMO: dict[str, WmoStation] = {
    "08535": WmoStation("08535", "Lisboa Geofisico", 38.72, -9.15, 77.0),
    "08545": WmoStation("08545", "Porto Pedras Rubras", 41.24, -8.68, 69.0),
    "08554": WmoStation("08554", "Faro", 37.02, -7.97, 8.0),
    "08549": WmoStation("08549", "Beja", 38.08, -7.93, 247.0),
    "08562": WmoStation("08562", "Sagres", 37.01, -8.95, 25.0),
    "08570": WmoStation("08570", "Funchal", 32.70, -16.77, 58.0),
    "08509": WmoStation("08509", "Braganca", 41.81, -6.76, 690.0),
    "08515": WmoStation("08515", "Viseu", 40.73, -7.90, 644.0),
    "08537": WmoStation("08537", "Coimbra", 40.20, -8.42, 179.0),
    "08571": WmoStation("08571", "Portalegre", 39.29, -7.42, 590.0),
    "08575": WmoStation("08575", "Evora", 38.53, -7.90, 309.0),
    "08579": WmoStation("08579", "Castelo Branco", 39.84, -7.48, 386.0),
}


class TransientOgimetError(RuntimeError):
    pass


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=5, min=5, max=30),
    retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError, TransientOgimetError)),
    reraise=True,
)
def _download_ogimet(url: str) -> str:
    global LAST_GET_TS
    if LAST_GET_TS is not None:
        wait_s = 5.0 - (time.monotonic() - LAST_GET_TS)
        if wait_s > 0:
            time.sleep(wait_s)
    resp = requests.get(url, timeout=45, headers={"User-Agent": "downscalewind-phase-g/1.0"})
    LAST_GET_TS = time.monotonic()
    if resp.status_code == 429 or resp.status_code >= 500:
        raise TransientOgimetError(f"OGIMET HTTP {resp.status_code}")
    resp.raise_for_status()
    text = resp.text
    if "captcha" in text.lower():
        raise TransientOgimetError("OGIMET captcha page")
    return text


def _fetch_month(wmo_id: str, year: int, month: int, cache_dir: Path) -> str:
    cache_path = cache_dir / f"{wmo_id}_{year:04d}_{month:02d}.html"
    if cache_path.exists():
        return cache_path.read_text(errors="replace")
    ndays = calendar.monthrange(year, month)[1]
    params = {
        "ind": wmo_id,
        "ano": year,
        "mes": f"{month:02d}",
        "day": "01",
        "hora": "00",
        "min": "0",
        "ndays": ndays,
        "lang": "en",
        "decoded": "yes",
    }
    url = f"{OGIMET_URL}?{urlencode(params)}"
    html = _download_ogimet(url)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(html)
    return html


def _months_between(start: datetime, end: datetime):
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        yield year, month
        month += 1
        if month == 13:
            year += 1
            month = 1


def _station_dataframe(station: WmoStation, start: datetime, end: datetime, cache_dir: Path) -> pd.DataFrame:
    frames = []
    for year, month in _months_between(start, end):
        try:
            html = _fetch_month(station.wmo_id, year, month, cache_dir)
            parsed = parse_ogimet_html(html)
        except Exception as exc:
            log.warning(
                "OGIMET month unavailable",
                extra={"wmo_id": station.wmo_id, "year": year, "month": month, "error": str(exc)},
            )
            continue
        frames.append(parsed)
    if not frames:
        return pd.DataFrame(columns=["time", "temp_c", "dewpoint_c", "wind_dir_deg", "wind_speed_ms"])
    df = pd.concat(frames, ignore_index=True)
    start64 = np.datetime64(start, "ns")
    end64 = np.datetime64(end, "ns")
    mask = (df["time"].values.astype("datetime64[ns]") >= start64) & (
        df["time"].values.astype("datetime64[ns]") <= end64
    )
    return df.loc[mask].drop_duplicates(subset=["time"], keep="last").sort_values("time")


def _hourly_grid(start: datetime, end: datetime) -> np.ndarray:
    start64 = np.datetime64(start, "h")
    end64 = np.datetime64(end, "h")
    return np.arange(start64, end64 + np.timedelta64(1, "h"), np.timedelta64(1, "h"), dtype="datetime64[ns]")


def _values_on_grid(df: pd.DataFrame, times: np.ndarray) -> dict[str, np.ndarray]:
    n = len(times)
    values = {name: np.full(n, np.nan, dtype=np.float32) for name in ("u", "v", "wind_speed", "wind_dir", "t2m", "rh")}
    if df.empty:
        return values
    lookup = {int(t): i for i, t in enumerate(times.astype("datetime64[ns]").astype(np.int64))}
    row_times = df["time"].values.astype("datetime64[ns]").astype(np.int64)
    temp_c = df["temp_c"].to_numpy(dtype=np.float32)
    dew_c = df["dewpoint_c"].to_numpy(dtype=np.float32)
    speed = df["wind_speed_ms"].to_numpy(dtype=np.float32)
    direction = df["wind_dir_deg"].to_numpy(dtype=np.float32)
    rh = relative_humidity_from_dewpoint(temp_c, dew_c)
    u, v = wind_to_uv(speed, direction)
    for i, t_ns in enumerate(row_times):
        slot = lookup.get(int(t_ns))
        if slot is None:
            continue
        values["wind_speed"][slot] = speed[i]
        values["wind_dir"][slot] = direction[i]
        values["u"][slot] = u[i]
        values["v"][slot] = v[i]
        values["t2m"][slot] = temp_c[i] + 273.15 if np.isfinite(temp_c[i]) else np.nan
        values["rh"][slot] = rh[i]
    return values


def _parse_start_end(start: str, end: str) -> tuple[datetime, datetime]:
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    if len(start) == 10:
        start_dt = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    if len(end) == 10:
        end_dt = end_dt.replace(hour=23, minute=0, second=0, microsecond=0)
    if end_dt < start_dt:
        raise click.ClickException("--end must be >= --start")
    return start_dt, end_dt


def _select_stations(stations: str | None, smoke: bool, root) -> list[WmoStation]:
    if stations:
        ids = [part.strip().zfill(5) for part in stations.split(",") if part.strip()]
    else:
        ids = list(PT_WMO)
    selected = [PT_WMO[wmo_id] for wmo_id in ids if wmo_id in PT_WMO]
    missing = [wmo_id for wmo_id in ids if wmo_id not in PT_WMO]
    for wmo_id in missing:
        log.warning("Unknown Portugal WMO station skipped", extra={"wmo_id": wmo_id})
    if smoke:
        projected = set(read_station_ids(root)) | {f"ogimet_{s.wmo_id}" for s in selected}
        for station in PT_WMO.values():
            if len(projected) >= 3:
                break
            if station not in selected:
                selected.append(station)
                projected.add(f"ogimet_{station.wmo_id}")
                log.info("Added smoke fallback station to satisfy schema invariant", extra={"wmo_id": station.wmo_id})
    return selected


@click.command()
@click.option("--out", default="data/raw/obs_unified_ipma_pt.zarr", show_default=True, help="Output Zarr store")
@click.option("--stations", default=None, help="Comma-separated WMO IDs; default is static PT list")
@click.option("--start", required=True, help="Start date/time, e.g. 2023-01-01")
@click.option("--end", required=True, help="End date/time, inclusive; date-only means 23:00")
@click.option("--cache-dir", default="tmp/ogimet_cache", show_default=True, help="Monthly OGIMET HTML cache")
@click.option("--smoke", is_flag=True, help="Smoke mode; stays non-fatal on remote failures")
def main(out: str, stations: str | None, start: str, end: str, cache_dir: str, smoke: bool) -> None:
    start_dt, end_dt = _parse_start_end(start, end)
    root = ensure_obs_store(out)
    selected = _select_stations(stations, smoke, root)
    if not selected:
        raise click.ClickException("No valid stations selected")
    times = _hourly_grid(start_dt, end_dt)
    cache = Path(cache_dir)

    ingested = 0
    valid_rows = 0
    for station in selected:
        df = _station_dataframe(station, start_dt, end_dt, cache)
        values = _values_on_grid(df, times)
        write_station_timeseries(root, station.record(), times, values)
        ingested += 1
        valid = int(np.isfinite(values["wind_speed"]).sum())
        valid_rows += valid
        log.info(
            "OGIMET station ingested",
            extra={"wmo_id": station.wmo_id, "n_times": int(times.size), "n_valid_wind": valid},
        )

    print(f"Done. OGIMET stations ingested: {ingested}; valid wind rows: {valid_rows}; output: {out}")
    print(f"Time grid: {times[0]} to {times[-1]} ({len(times)} hourly slots)")


if __name__ == "__main__":
    main()
