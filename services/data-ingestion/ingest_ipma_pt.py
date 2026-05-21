"""IPMA Portugal live hourly observations ingestion for Phase G."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.logging_config import get_logger
from utils.obs_zarr_writer import StationRecord, ensure_obs_store, wind_to_uv, write_station_timeseries


log = get_logger("ingest_ipma_pt")

IPMA_BASE = "https://api.ipma.pt/open-data/observation/meteorology/stations"
STATIONS_URL = f"{IPMA_BASE}/stations.json"
DIRECTION_DEG = {
    0: 0.0,
    1: 45.0,
    2: 90.0,
    3: 135.0,
    4: 180.0,
    5: 225.0,
    6: 270.0,
    7: 315.0,
}


class TransientIPMAError(RuntimeError):
    pass


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=10),
    retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError, TransientIPMAError)),
    reraise=True,
)
def _download_json(url: str) -> object:
    resp = requests.get(url, timeout=20, headers={"User-Agent": "downscalewind-phase-g/1.0"})
    if resp.status_code == 429 or resp.status_code >= 500:
        raise TransientIPMAError(f"IPMA HTTP {resp.status_code}: {url}")
    resp.raise_for_status()
    return resp.json()


def _fetch_json(url: str, cache_path: Path) -> object:
    try:
        data = _download_json(url)
    except Exception as exc:
        if cache_path.exists():
            log.warning("IPMA fetch failed, using cache", extra={"url": url, "cache": str(cache_path), "error": str(exc)})
            return json.loads(cache_path.read_text())
        raise
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data, ensure_ascii=False))
    return data


def _parse_stations(payload: object) -> list[StationRecord]:
    features = payload.get("features", payload) if isinstance(payload, dict) else payload
    stations: list[StationRecord] = []
    if not isinstance(features, list):
        return stations
    for item in features:
        if not isinstance(item, dict):
            continue
        props = item.get("properties", item)
        raw_id = props.get("idEstacao") or props.get("id") or props.get("stationId")
        coords = item.get("geometry", {}).get("coordinates", [])
        if raw_id is None or len(coords) < 2:
            continue
        elev = _first_float(props, ("altitude", "altitudeEstacao", "elevation", "cota"))
        stations.append(
            StationRecord(
                station_id=f"ipma_{raw_id}",
                lon=float(coords[0]),
                lat=float(coords[1]),
                elev=elev,
            )
        )
    stations.sort(key=lambda s: s.station_id)
    return stations


def _parse_live_obs(payload: object) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not isinstance(payload, dict):
        return np.array([], dtype="datetime64[ns]"), {}
    rows = []
    for time_text, obs in payload.items():
        if not isinstance(obs, dict):
            continue
        ts = pd.to_datetime(time_text, errors="coerce", utc=True)
        if pd.isna(ts):
            continue
        speed_kmh = _clean_float(obs.get("intensidadeVentoKM"))
        speed_ms = speed_kmh / 3.6 if np.isfinite(speed_kmh) else np.nan
        direction = _direction(obs.get("idDireccVento"))
        temp_c = _clean_float(obs.get("temperatura"))
        rh = _clean_float(obs.get("humidade"))
        rows.append((ts.tz_convert("UTC").tz_localize(None).to_datetime64(), speed_ms, direction, temp_c, rh))
    if not rows:
        return np.array([], dtype="datetime64[ns]"), {}

    rows.sort(key=lambda row: row[0])
    times = np.array([row[0] for row in rows], dtype="datetime64[ns]")
    wind_speed = np.array([row[1] for row in rows], dtype=np.float32)
    wind_dir = np.array([row[2] for row in rows], dtype=np.float32)
    temp_c = np.array([row[3] for row in rows], dtype=np.float32)
    rh = np.array([row[4] for row in rows], dtype=np.float32)
    u, v = wind_to_uv(wind_speed, wind_dir)
    t2m = np.where(np.isfinite(temp_c), temp_c + 273.15, np.nan).astype(np.float32)
    return times, {"u": u, "v": v, "wind_speed": wind_speed, "wind_dir": wind_dir, "t2m": t2m, "rh": rh}


def _direction(value) -> float:
    val = _clean_float(value)
    if not np.isfinite(val):
        return np.nan
    code = int(val)
    return DIRECTION_DEG.get(code, np.nan)


def _first_float(mapping: dict, keys: tuple[str, ...]) -> float:
    for key in keys:
        val = _clean_float(mapping.get(key))
        if np.isfinite(val):
            return val
    return np.nan


def _clean_float(value) -> float:
    if value is None:
        return np.nan
    try:
        val = float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return np.nan
    return np.nan if val <= -98.0 else val


@click.command()
@click.option("--out", default="data/raw/obs_unified_ipma_pt.zarr", show_default=True, help="Output Zarr store")
@click.option("--cache-dir", default="tmp/ipma_cache", show_default=True, help="Local IPMA JSON cache")
@click.option("--max-stations", default=0, show_default=True, help="Limit stations for non-smoke runs; 0 means all")
@click.option("--smoke-live", is_flag=True, help="Fetch enough live stations for a fast smoke test")
def main(out: str, cache_dir: str, max_stations: int, smoke_live: bool) -> None:
    root = ensure_obs_store(out)
    cache = Path(cache_dir)
    try:
        stations = _parse_stations(_fetch_json(STATIONS_URL, cache / "stations.json"))
    except Exception as exc:
        log.warning("IPMA station metadata unavailable; continuing with empty store", extra={"error": str(exc)})
        print(f"Done. IPMA stations ingested: 0; output: {out}")
        return
    if not stations:
        log.warning("No IPMA stations found in metadata")
        print(f"Done. IPMA stations ingested: 0; output: {out}")
        return

    limit = max_stations if max_stations > 0 else len(stations)
    if smoke_live:
        limit = min(limit, 40)
    ingested = 0
    attempted = 0
    for station in stations[:limit]:
        attempted += 1
        raw_id = station.station_id.removeprefix("ipma_")
        url = f"{IPMA_BASE}/obs-{raw_id}.json"
        try:
            payload = _fetch_json(url, cache / f"obs_{raw_id}.json")
            times, values = _parse_live_obs(payload)
        except Exception as exc:
            log.warning("IPMA station obs unavailable", extra={"station_id": station.station_id, "error": str(exc)})
            continue
        if times.size == 0:
            log.warning("IPMA station has no parseable live observations", extra={"station_id": station.station_id})
            continue
        write_station_timeseries(root, station, times, values)
        ingested += 1
        log.info("IPMA station ingested", extra={"station_id": station.station_id, "n_times": int(times.size)})
        if smoke_live and ingested >= 2:
            break

    if smoke_live and ingested < 2:
        log.warning("IPMA smoke live completed with fewer than two stations", extra={"ingested": ingested, "attempted": attempted})
    print(f"Done. IPMA stations ingested: {ingested}; output: {out}")


if __name__ == "__main__":
    main()
