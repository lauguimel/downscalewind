"""Ingest AEMET Spain observations into the unified OBS Zarr schema."""

from __future__ import annotations

import json
import os
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import click
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.logging_config import get_logger
from shared.obs_io import append_obs_data, create_obs_store, read_obs
from utils.aemet_client import (
    AemetCacheMiss,
    AemetClient,
    AemetHTTPError,
    _dms_to_deg,
    wind_speed_dir_to_uv,
)
from utils.checkpointing import Checkpointer

log = get_logger("ingest_aemet")

MISSING_KEY_MESSAGE = "Set AEMET_API_KEY env var. Get key at https://opendata.aemet.es/centrodedescargas/altaUsuario"
REGION_ORDER = ("Norte", "Centro", "Sur", "Baleares", "Canarias")
REGION_PROVINCES = {
    "Norte": {
        "A CORUNA", "LUGO", "OURENSE", "PONTEVEDRA", "ASTURIAS", "CANTABRIA",
        "BIZKAIA", "GIPUZKOA", "ARABA/ALAVA", "ARABA", "ALAVA", "NAVARRA",
    },
    "Centro": {
        "MADRID", "AVILA", "BURGOS", "LEON", "PALENCIA", "SALAMANCA",
        "SEGOVIA", "SORIA", "VALLADOLID", "ZAMORA", "ALBACETE", "CIUDAD REAL",
        "CUENCA", "GUADALAJARA", "TOLEDO", "HUESCA", "TERUEL", "ZARAGOZA",
        "LA RIOJA", "BADAJOZ", "CACERES",
    },
    "Sur": {
        "ALMERIA", "CADIZ", "CORDOBA", "GRANADA", "HUELVA", "JAEN", "MALAGA",
        "SEVILLA", "MURCIA", "ALICANTE", "CASTELLON", "VALENCIA", "BARCELONA",
        "GIRONA", "LLEIDA", "TARRAGONA",
    },
    "Baleares": {"ILLES BALEARS", "BALEARES"},
    "Canarias": {"LAS PALMAS", "SANTA CRUZ DE TENERIFE"},
}

FIELD_MAPPING = {
    "wind_speed": ["velmedia", "vv"],
    "wind_dir": ["dir", "dv"],
    "t2m": ["tmed", "ta"],
    "rh": ["hr", "humedad_relativa"],
}
DATA_VARS = ("u", "v", "wind_speed", "wind_dir", "t2m", "rh")


@dataclass(frozen=True)
class Station:
    idema: str
    station_id_str: str
    provincia: str
    region: str
    lat: float
    lon: float
    elev: float


@click.command()
@click.option("--out", "out_path", required=True, type=click.Path(path_type=Path))
@click.option("--region", required=True, type=click.Choice([*REGION_ORDER, "all"]))
@click.option("--start", required=True, help="Start month YYYY-MM")
@click.option("--end", required=True, help="End month YYYY-MM, inclusive")
@click.option("--smoke", is_flag=True, help="Cap to 50 stations and one month")
@click.option("--cache-dir", default="tmp/aemet_cache", type=click.Path(path_type=Path))
@click.option("--checkpoint-dir", default=None, type=click.Path(path_type=Path))
@click.option("--dry-run", is_flag=True, help="Plan only; uses cached inventory if present")
@click.option("--mock", is_flag=True, help="Run offline parser/mapping/Zarr scaffolding checks")
def main(
    out_path: Path,
    region: str,
    start: str,
    end: str,
    smoke: bool,
    cache_dir: Path,
    checkpoint_dir: Path | None,
    dry_run: bool,
    mock: bool,
) -> None:
    start_dt, end_exclusive, effective_end = _month_window(start, end, smoke)
    regions = list(REGION_ORDER) if region == "all" else [region]
    if mock:
        _run_mock(out_path, cache_dir, regions, start_dt)
        return

    api_key = os.environ.get("AEMET_API_KEY", "")
    if not api_key:
        _fail_missing_api_key()

    client = AemetClient(api_key, cache_dir=cache_dir, logger=log)
    cp_dir = checkpoint_dir or (out_path / ".checkpoints")
    cp = Checkpointer(cp_dir)

    try:
        inventory = client.station_inventory(cache_only=dry_run)
    except AemetCacheMiss:
        log.info("Dry-run inventory cache miss; no HTTP performed", extra={"cache_dir": str(cache_dir)})
        _log_plan(regions, start, effective_end, smoke, stations=[])
        return

    stations = _stations_from_inventory(inventory, regions)
    if smoke:
        stations = stations[:50]
    _log_plan(regions, start, effective_end, smoke, stations=stations)
    if dry_run:
        return
    if not stations:
        raise click.ClickException("No AEMET stations matched the selected region(s)")

    cadence = _select_cadence(client, stations, start_dt, end_exclusive)
    frames = _fetch_station_frames(client, cp, stations, regions, start_dt, end_exclusive, cadence)
    usable = [(st, df) for st, df in zip(stations, frames) if _has_usable_wind(df)]
    if not usable:
        raise click.ClickException("No station yielded usable wind observations for the requested period")

    use_stations, use_frames = zip(*usable)
    time_index = _time_index(start_dt, end_exclusive, cadence)
    _write_obs_store(out_path, list(use_stations), list(use_frames), time_index)
    log.info(
        "AEMET ingestion complete",
        extra={"out": str(out_path), "n_stations": len(use_stations), "n_times": len(time_index), "cadence": cadence},
    )


def _fail_missing_api_key() -> None:
    click.echo(MISSING_KEY_MESSAGE, err=True)
    raise click.exceptions.Exit(2)


def _run_mock(out_path: Path, cache_dir: Path, regions: list[str], start_dt: pd.Timestamp) -> None:
    client = AemetClient("mock-no-api", cache_dir=cache_dir, logger=log)
    fixture_payload = _dummy_inventory_records(per_region=1)
    fixture_path = client._cache_path("GET", client._url("/valores/climatologicos/inventarioestaciones/todasestaciones"))
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_path.write_text(
        json.dumps({"_aemet_cache_payload": fixture_payload, "endpoint_url": "mock://station_inventory"}, ensure_ascii=False),
        encoding="utf-8",
    )
    parsed_fixture = _stations_from_inventory(client.station_inventory(cache_only=True), list(REGION_ORDER))
    if len(parsed_fixture) != len(fixture_payload):
        raise click.ClickException("Mock inventory fixture parsing failed")

    dummy_payload = _dummy_inventory_records(per_region=50)
    dummy_stations = _stations_from_inventory(dummy_payload, list(REGION_ORDER))
    _assert_dummy_region_mapping(dummy_payload, dummy_stations)

    pool = [st for st in dummy_stations if st.region in regions]
    if len(pool) < 3:
        raise click.ClickException("Mock station pool has fewer than 3 stations")
    mock_stations = pool[:3]
    time_index = pd.date_range(start_dt, periods=24, freq="h")
    speed = np.full(len(time_index), 5.0, dtype=np.float32)
    direction = np.full(len(time_index), 180.0, dtype=np.float32)
    u, v = wind_speed_dir_to_uv(speed, direction)
    frame = pd.DataFrame({
        "u": u, "v": v, "wind_speed": speed, "wind_dir": direction,
        "t2m": np.full(len(time_index), 288.15, dtype=np.float32),
        "rh": np.full(len(time_index), 70.0, dtype=np.float32),
    }, index=time_index)
    _write_obs_store(out_path, mock_stations, [frame.copy() for _ in mock_stations], time_index)

    df = read_obs(out_path)
    expected_ids = {st.station_id_str for st in mock_stations}
    if len(df) == 0:
        raise click.ClickException("Mock read_obs returned no rows")
    if set(df["source"].unique()) != {"aemet_es"}:
        raise click.ClickException("Mock source invariant failed")
    if set(df["height_m"].astype(float).unique()) != {10.0}:
        raise click.ClickException("Mock height invariant failed")
    if not expected_ids.issubset(set(df["station_id"].unique())):
        raise click.ClickException("Mock station_id invariant failed")
    click.echo(f"MOCK_OK n_stations={len(mock_stations)} n_rows={len(df)}")


def _dummy_inventory_records(per_region: int) -> list[dict]:
    records: list[dict] = []
    region_coords = {
        "Norte": (43, 8, "W"),
        "Centro": (40, 3, "W"),
        "Sur": (37, 4, "W"),
        "Baleares": (39, 2, "E"),
        "Canarias": (28, 15, "W"),
    }
    for region_idx, region in enumerate(REGION_ORDER):
        provinces = sorted(REGION_PROVINCES[region])
        lat_deg, lon_deg, lon_hemi = region_coords[region]
        for i in range(per_region):
            minutes = i % 60
            records.append({
                "indicativo": f"M{region_idx}{i:03d}",
                "nombre": f"Mock {region} {i:03d}",
                "provincia": provinces[i % len(provinces)],
                "latitud": f"{lat_deg:02d}{minutes:02d}00N",
                "longitud": f"{lon_deg:03d}{minutes:02d}00{lon_hemi}",
                "altitud": str(50 + i),
                "_expected_region": region,
            })
    return records


def _assert_dummy_region_mapping(payload: list[dict], stations: list[Station]) -> None:
    expected = {str(raw["indicativo"]): raw["_expected_region"] for raw in payload}
    if len(stations) != len(expected):
        raise click.ClickException("Mock region mapping dropped dummy stations")
    actual = {station.idema: station.region for station in stations}
    mismatches = [sid for sid, expected_region in expected.items() if actual.get(sid) != expected_region]
    if mismatches:
        raise click.ClickException(f"Mock region mapping failed for {mismatches[:5]}")


def _month_window(start: str, end: str, smoke: bool) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    start_dt = pd.Timestamp(f"{start}-01", tz="UTC")
    end_dt = pd.Timestamp(f"{end}-01", tz="UTC")
    if end_dt < start_dt:
        raise click.ClickException("--end must be >= --start")
    if smoke and _month_ordinal(end_dt) > _month_ordinal(start_dt):
        end_dt = start_dt
    end_exclusive = _add_month(end_dt)
    effective_end = f"{end_dt.year:04d}-{end_dt.month:02d}"
    return start_dt, end_exclusive, effective_end


def _month_ordinal(ts: pd.Timestamp) -> int:
    return ts.year * 12 + ts.month


def _add_month(ts: pd.Timestamp) -> pd.Timestamp:
    year = ts.year + (1 if ts.month == 12 else 0)
    month = 1 if ts.month == 12 else ts.month + 1
    return pd.Timestamp(year=year, month=month, day=1, tz="UTC")


def _stations_from_inventory(inventory: list[dict], regions: list[str]) -> list[Station]:
    wanted = set(regions)
    stations: list[Station] = []
    for raw in inventory:
        idema = str(raw.get("indicativo", "")).strip()
        provincia = _norm_province(raw.get("provincia", ""))
        st_region = _region_for_province(provincia)
        if not idema or st_region not in wanted:
            continue
        try:
            lat = _dms_to_deg(str(raw["latitud"]))
            lon = _dms_to_deg(str(raw["longitud"]))
            elev = _parse_float(raw.get("altitud"))
        except (KeyError, ValueError) as exc:
            log.warning("Skipping malformed AEMET station", extra={"idema": idema, "error": str(exc)})
            continue
        stations.append(
            Station(
                idema=idema,
                station_id_str=f"aemet_{idema}".encode("ascii", errors="ignore").decode("ascii")[:16],
                provincia=provincia,
                region=st_region or "",
                lat=lat,
                lon=lon,
                elev=elev,
            )
        )
    return stations


def _write_obs_store(out_path: Path, stations: list[Station], frames: list[pd.DataFrame], time_index: pd.DatetimeIndex) -> None:
    time_array = _datetime64_ns(time_index)
    create_obs_store(
        out_path,
        stations_df=_stations_dataframe(stations),
        heights_array=np.array([10.0], dtype=np.float32),
        time_array=time_array,
    )
    for station, frame in zip(stations, frames):
        aligned = frame.reindex(time_index)
        append_obs_data(
            out_path,
            source="aemet_es",
            station_id=station.station_id_str,
            time_array=time_array,
            data_dict={
                var: aligned[var].to_numpy(dtype=np.float32).reshape(-1, 1)
                for var in DATA_VARS
            },
            height_idx_map={10.0: 0},
        )


def _stations_dataframe(stations: list[Station]) -> pd.DataFrame:
    return pd.DataFrame({
        "station_id": [st.station_id_str for st in stations],
        "lat": [st.lat for st in stations],
        "lon": [st.lon for st in stations],
        "elev": [st.elev for st in stations],
        "source": ["aemet_es"] * len(stations),
        "country": ["ES"] * len(stations),
        "z0_class_wc": np.full(len(stations), -1, dtype=np.int8),
    })


def _datetime64_ns(index: pd.DatetimeIndex) -> np.ndarray:
    if index.tz is None:
        utc_naive = index.tz_localize("UTC").tz_localize(None)
    else:
        utc_naive = index.tz_convert("UTC").tz_localize(None)
    return utc_naive.to_numpy(dtype="datetime64[ns]")


def _norm_province(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value).strip().upper())
    return "".join(c for c in text if not unicodedata.combining(c))


def _region_for_province(provincia: str) -> str | None:
    for region, provinces in REGION_PROVINCES.items():
        if provincia in provinces:
            return region
    return None


def _select_cadence(
    client: AemetClient,
    stations: list[Station],
    start_dt: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> str:
    probe = stations[: max(3, min(3, len(stations)))]
    if len(probe) < 3:
        probe = stations[:]
    failures = 0
    for station in probe:
        try:
            records = client.hourly_archive(station.idema, start_dt.year)
        except AemetHTTPError as exc:
            msg = str(exc).lower()
            if exc.status_code in {401, 404} or "deprecated" in msg:
                failures += 1
                continue
            raise
        if _parse_records(records, start_dt, end_exclusive).empty:
            failures += 1
            continue
        log.info("AEMET hourly archive selected", extra={"probe_station": station.idema})
        return "hourly"
    if failures >= min(3, len(probe)):
        log.warning("AEMET hourly probe failed; falling back to daily", extra={"probe_failures": failures})
        return "daily"
    return "hourly"


def _fetch_station_frames(
    client: AemetClient,
    cp: Checkpointer,
    stations: list[Station],
    regions: list[str],
    start_dt: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    cadence: str,
) -> list[pd.DataFrame]:
    frames = {st.idema: [] for st in stations}
    for reg in regions:
        reg_stations = [s for s in stations if s.region == reg]
        for year, y0, y1 in _iter_year_windows(start_dt, end_exclusive):
            key = f"aemet_{reg}_{year}"
            for station in reg_stations:
                try:
                    if cadence == "hourly":
                        records = client.hourly_archive(station.idema, year)
                    else:
                        records = client.daily_archive(station.idema, _fmt_aemet_dt(y0), _fmt_aemet_dt(y1))
                except AemetHTTPError as exc:
                    log.warning("AEMET station fetch failed", extra={"station": station.idema, "year": year, "error": str(exc)})
                    records = []
                frames[station.idema].append(_parse_records(records, y0, y1 + pd.Timedelta(nanoseconds=1)))
            cp.mark_done(key, extra_meta={"n_stations": len(reg_stations), "year": year, "cadence": cadence})
            log.info("AEMET region-year checkpointed", extra={"key": key, "already_done": cp.is_done(key)})
    return [_concat_frames(frames[st.idema]) for st in stations]


def _iter_year_windows(
    start_dt: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> Iterable[tuple[int, pd.Timestamp, pd.Timestamp]]:
    for year in range(start_dt.year, end_exclusive.year + 1):
        y0 = max(start_dt, pd.Timestamp(year=year, month=1, day=1, tz="UTC"))
        y1_excl = min(end_exclusive, pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC"))
        if y0 < y1_excl:
            yield year, y0, y1_excl - pd.Timedelta(nanoseconds=1)


def _fmt_aemet_dt(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%SUTC")


def _parse_records(records: list[dict], start_dt: pd.Timestamp, end_exclusive: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for rec in records:
        ts = _record_timestamp(rec)
        if ts is None or ts < start_dt or ts >= end_exclusive:
            continue
        speed = _first_float(rec, FIELD_MAPPING["wind_speed"])
        direction = _first_float(rec, FIELD_MAPPING["wind_dir"])
        temp_c = _first_float(rec, FIELD_MAPPING["t2m"])
        rh = _first_float(rec, FIELD_MAPPING["rh"])
        if not np.isfinite(speed) or speed < 0:
            speed = np.nan
        if not np.isfinite(direction) or direction < 0 or direction > 360:
            direction = np.nan
        t2m = temp_c + 273.15 if np.isfinite(temp_c) else np.nan
        u, v = wind_speed_dir_to_uv(np.asarray([speed], dtype=np.float32), np.asarray([direction], dtype=np.float32))
        rows.append((ts, u[0], v[0], speed, direction, t2m, rh))
    if not rows:
        return pd.DataFrame(columns=DATA_VARS)
    df = pd.DataFrame(rows, columns=["time", *DATA_VARS]).set_index("time").sort_index()
    return df.groupby(level=0).mean(numeric_only=True).astype(np.float32)


def _record_timestamp(rec: dict) -> pd.Timestamp | None:
    for key in ("fhora", "fecha", "fint", "fin", "datetime", "time"):
        value = rec.get(key)
        if value:
            ts = pd.to_datetime(value, utc=True, errors="coerce")
            if not pd.isna(ts):
                return pd.Timestamp(ts)
    return None


def _first_float(rec: dict, keys: Iterable[str]) -> float:
    for key in keys:
        value = rec.get(key)
        parsed = _parse_float(value, default=np.nan)
        if np.isfinite(parsed):
            return parsed
    return np.nan


def _parse_float(value: object, default: float | None = None) -> float:
    if value is None or value == "":
        if default is not None:
            return default
        raise ValueError("missing numeric value")
    try:
        return float(str(value).strip().replace(",", "."))
    except ValueError:
        if default is not None:
            return default
        raise


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    parts = [df for df in frames if not df.empty]
    if not parts:
        return pd.DataFrame(columns=DATA_VARS)
    return pd.concat(parts).sort_index().groupby(level=0).mean(numeric_only=True).astype(np.float32)


def _has_usable_wind(df: pd.DataFrame) -> bool:
    return not df.empty and np.isfinite(df[["wind_speed", "wind_dir"]].to_numpy()).all(axis=1).any()


def _time_index(start_dt: pd.Timestamp, end_exclusive: pd.Timestamp, cadence: str) -> pd.DatetimeIndex:
    freq = "h" if cadence == "hourly" else "D"
    return pd.date_range(start_dt, end_exclusive, freq=freq, inclusive="left")


def _log_plan(regions: list[str], start: str, end: str, smoke: bool, stations: list[Station]) -> None:
    log.info(
        "AEMET ingestion plan",
        extra={
            "regions": regions,
            "start": start,
            "end": end,
            "smoke": smoke,
            "n_stations": len(stations),
            "station_ids": [s.idema for s in stations[:10]],
        },
    )


if __name__ == "__main__":
    main()
