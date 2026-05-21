"""Ingest NOAA ISD-Lite hourly observations into the unified OBS Zarr schema."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import click
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.logging_config import get_logger
from shared.obs_io import DATA_VARS, append_obs_data, create_obs_store, read_obs
from utils.isd_parser import (
    NoaaDownloadError,
    fetch_isd_history,
    fetch_isd_lite_year,
    filter_stations,
    parse_isd_lite_gz,
)

log = get_logger("ingest_noaa_isd")

SOURCE = "noaa_isd"
HEIGHTS = np.array([10.0], dtype=np.float32)
SMOKE_PER_COUNTRY = 3
SMOKE_MAX_STATIONS = 9
PREFERRED_SMOKE = {
    "FR": ["071490-99999", "076500-99999", "074810-99999", "072220-99999", "070150-99999"],
    "ES": ["082210-99999", "081810-99999", "083600-99999", "084820-99999", "080010-99999"],
    "PT": ["085360-99999", "085450-99999", "085540-99999", "085790-99999", "085220-99999"],
}


@click.command()
@click.option("--out", "out_path", required=True, type=click.Path(path_type=Path))
@click.option("--countries", required=True, help="Comma-separated ISO country codes: FR,ES,PT")
@click.option("--start", required=True, help="Start month YYYY-MM")
@click.option("--end", required=True, help="End month YYYY-MM, inclusive")
@click.option("--smoke", is_flag=True, help="Cap to about 8 active stations distributed by country")
@click.option("--cache-dir", default="tmp/noaa_cache", type=click.Path(path_type=Path))
def main(out_path: Path, countries: str, start: str, end: str, smoke: bool, cache_dir: Path) -> None:
    start_dt, end_exclusive = _month_window(start, end)
    end_inclusive = end_exclusive - pd.Timedelta(nanoseconds=1)
    country_order = _parse_countries(countries)
    time_index = pd.date_range(start_dt, end_exclusive, freq="h", inclusive="left")
    if len(time_index) == 0:
        raise click.ClickException("Requested period has no hourly timestamps")

    try:
        history = fetch_isd_history(cache_dir)
    except NoaaDownloadError as exc:
        raise click.ClickException(str(exc)) from exc
    _echo_country_probe(history)
    stations = filter_stations(history, country_order, start_dt, end_inclusive)
    if stations.empty:
        raise click.ClickException("No active NOAA ISD stations matched the selected countries and period")
    if smoke:
        stations = _select_smoke_stations(stations, country_order)

    click.echo(
        "NOAA ISD plan: "
        f"countries={','.join(country_order)} start={start_dt:%Y-%m-%d} "
        f"end={end_inclusive:%Y-%m-%d} smoke={smoke} candidates={len(stations)}"
    )

    usable_stations, frames, caveats = _load_station_frames(stations, start_dt, end_exclusive, cache_dir)
    if usable_stations.empty:
        raise click.ClickException("No selected NOAA ISD station yielded usable wind observations")

    _write_obs_store(out_path, usable_stations, frames, time_index)
    report = _validate_output(out_path, country_order, smoke=smoke)
    _print_report(out_path, time_index, report, caveats, smoke=smoke)


def _parse_countries(countries: str) -> list[str]:
    aliases = {"FR": "FR", "ES": "ES", "PT": "PT", "SP": "ES", "PO": "PT"}
    parsed: list[str] = []
    for raw in countries.split(","):
        code = raw.strip().upper()
        if not code:
            continue
        if code not in aliases:
            raise click.ClickException(f"Unsupported country code {raw!r}; expected FR, ES, PT")
        iso = aliases[code]
        if iso not in parsed:
            parsed.append(iso)
    if not parsed:
        raise click.ClickException("--countries must include at least one of FR, ES, PT")
    return parsed


def _month_window(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    try:
        start_dt = pd.Timestamp(f"{start}-01")
        end_dt = pd.Timestamp(f"{end}-01")
    except ValueError as exc:
        raise click.ClickException("--start and --end must use YYYY-MM") from exc
    if end_dt < start_dt:
        raise click.ClickException("--end must be >= --start")
    return start_dt, _add_month(end_dt)


def _add_month(ts: pd.Timestamp) -> pd.Timestamp:
    year = ts.year + (1 if ts.month == 12 else 0)
    month = 1 if ts.month == 12 else ts.month + 1
    return pd.Timestamp(year=year, month=month, day=1)


def _echo_country_probe(history: pd.DataFrame) -> None:
    first_codes = [code for code in history.get("CTRY", pd.Series(dtype=str)).astype(str).str.strip().head(20) if code]
    target_counts = history.get("CTRY", pd.Series(dtype=str)).astype(str).str.strip().value_counts()
    counts = {code: int(target_counts.get(code, 0)) for code in ("FR", "SP", "PO")}
    click.echo(f"NOAA isd-history CTRY probe: first20={first_codes[:10]} target_counts={counts}")


def _select_smoke_stations(stations: pd.DataFrame, countries: list[str]) -> pd.DataFrame:
    selected_parts: list[pd.DataFrame] = []
    selected_idx: set[int] = set()
    for country in countries:
        pool = stations.loc[stations["country"] == country].copy()
        if pool.empty:
            continue
        ranked = _rank_smoke_pool(pool, country)
        take = ranked.head(SMOKE_PER_COUNTRY)
        selected_parts.append(take)
        selected_idx.update(int(idx) for idx in take.index)

    if selected_parts:
        selected = pd.concat(selected_parts).drop_duplicates(subset=["usaf", "wban"], keep="first")
    else:
        selected = pd.DataFrame(columns=stations.columns)
    if len(selected) < min(5, len(stations)):
        extras = stations.loc[~stations.index.isin(selected_idx)].copy()
        extras = extras.sort_values(["country", "usaf", "wban"]).head(min(5, len(stations)) - len(selected))
        selected = pd.concat([selected, extras]).drop_duplicates(subset=["usaf", "wban"], keep="first")
    return selected.head(SMOKE_MAX_STATIONS).reset_index(drop=True)


def _rank_smoke_pool(pool: pd.DataFrame, country: str) -> pd.DataFrame:
    preferred = {key: rank for rank, key in enumerate(PREFERRED_SMOKE.get(country, []))}
    keys = pool["usaf"].astype(str) + "-" + pool["wban"].astype(str)
    ranked = pool.copy()
    ranked["_preferred"] = keys.map(preferred).fillna(10_000).astype(int)
    ranked["_has_icao"] = ranked["icao"].astype(str).str.strip().ne("").map({True: 0, False: 1}).astype(int)
    ranked["_wban99999"] = ranked["wban"].astype(str).eq("99999").map({True: 0, False: 1}).astype(int)
    ranked = ranked.sort_values(["_preferred", "_has_icao", "_wban99999", "usaf", "wban"])
    return ranked.drop(columns=["_preferred", "_has_icao", "_wban99999"])


def _load_station_frames(
    stations: pd.DataFrame,
    start_dt: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    cache_dir: Path,
) -> tuple[pd.DataFrame, list[pd.DataFrame], Counter[str]]:
    usable_rows: list[dict] = []
    usable_frames: list[pd.DataFrame] = []
    caveats: Counter[str] = Counter()
    for station in stations.to_dict("records"):
        parts: list[pd.DataFrame] = []
        for year in _years_covered(start_dt, end_exclusive):
            try:
                path = fetch_isd_lite_year(str(station["usaf"]), str(station["wban"]), year, cache_dir)
            except NoaaDownloadError as exc:
                raise click.ClickException(str(exc)) from exc
            if path is None:
                caveats["missing_station_year_files"] += 1
                log.warning(
                    "NOAA ISD-Lite station-year missing",
                    extra={"station_id": station["station_id"], "year": year},
                )
                continue
            try:
                parsed = parse_isd_lite_gz(path)
            except Exception as exc:
                caveats["parse_failures"] += 1
                log.warning(
                    "NOAA ISD-Lite parse failed",
                    extra={"station_id": station["station_id"], "year": year, "error": str(exc)},
                )
                continue
            parsed = parsed.loc[(parsed["time"] >= start_dt) & (parsed["time"] < end_exclusive)]
            if not parsed.empty:
                parts.append(parsed)

        frame = _concat_station_parts(parts)
        if not _has_usable_wind(frame):
            caveats["stations_without_usable_wind"] += 1
            continue
        usable_rows.append(station)
        usable_frames.append(frame)
        click.echo(f"Loaded {station['station_id']} {station['country']} rows={len(frame)}")

    return pd.DataFrame(usable_rows), usable_frames, caveats


def _years_covered(start_dt: pd.Timestamp, end_exclusive: pd.Timestamp) -> Iterable[int]:
    last = end_exclusive - pd.Timedelta(nanoseconds=1)
    yield from range(int(start_dt.year), int(last.year) + 1)


def _concat_station_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=DATA_VARS)
    frame = pd.concat(parts, ignore_index=True)
    frame = frame.sort_values("time").groupby("time", as_index=True).mean(numeric_only=True)
    for var in DATA_VARS:
        if var not in frame.columns:
            frame[var] = np.nan
    return frame.loc[:, list(DATA_VARS)].astype(np.float32)


def _has_usable_wind(frame: pd.DataFrame) -> bool:
    if frame.empty:
        return False
    values = frame[["wind_speed", "wind_dir"]].to_numpy(dtype=np.float32)
    return bool(np.isfinite(values).all(axis=1).any())


def _write_obs_store(out_path: Path, stations: pd.DataFrame, frames: list[pd.DataFrame], time_index: pd.DatetimeIndex) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    time_array = _datetime64_ns(time_index)
    create_obs_store(
        out_path,
        stations_df=_stations_dataframe(stations),
        heights_array=HEIGHTS,
        time_array=time_array,
    )
    for station, frame in zip(stations.to_dict("records"), frames):
        aligned = frame.reindex(time_index)
        append_obs_data(
            out_path,
            source=SOURCE,
            station_id=str(station["station_id"]),
            time_array=time_array,
            data_dict={var: aligned[var].to_numpy(dtype=np.float32).reshape(-1, 1) for var in DATA_VARS},
            height_idx_map={10.0: 0},
        )


def _stations_dataframe(stations: pd.DataFrame) -> pd.DataFrame:
    station_ids = stations["station_id"].astype(str)
    too_long = station_ids.map(len) > 16
    if too_long.any():
        bad = station_ids.loc[too_long].head(3).tolist()
        raise click.ClickException(f"NOAA station_id exceeds 16 chars: {bad}")
    return pd.DataFrame(
        {
            "station_id": station_ids.tolist(),
            "lat": stations["lat"].to_numpy(dtype=np.float32),
            "lon": stations["lon"].to_numpy(dtype=np.float32),
            "elev": stations["elev"].to_numpy(dtype=np.float32),
            "source": [SOURCE] * len(stations),
            "country": stations["country"].astype(str).tolist(),
            "z0_class_wc": np.full(len(stations), -1, dtype=np.int8),
        }
    )


def _datetime64_ns(index: pd.DatetimeIndex) -> np.ndarray:
    if index.tz is None:
        return index.to_numpy(dtype="datetime64[ns]")
    return index.tz_convert("UTC").tz_localize(None).to_numpy(dtype="datetime64[ns]")


def _validate_output(out_path: Path, requested_countries: list[str], *, smoke: bool) -> dict:
    df = read_obs(out_path)
    rows = int(len(df))
    stations = int(df["station_id"].nunique()) if rows else 0
    countries = sorted(str(code) for code in df["country"].dropna().unique()) if rows else []
    by_country = {str(k): int(v) for k, v in df.groupby("country")["station_id"].nunique().items()} if rows else {}
    ws = df["wind_speed"].dropna().astype(float) if rows else pd.Series(dtype=float)
    frac_in_0_50 = float(ws.between(0, 50).mean()) if len(ws) else 0.0

    if rows < 1000:
        raise click.ClickException(f"Validation failed: only {rows} rows")
    if smoke and stations < 5:
        raise click.ClickException(f"Validation failed: only {stations} stations ingested in smoke")
    if not set(countries).issubset({"FR", "ES", "PT"}):
        raise click.ClickException(f"Validation failed: unexpected country codes {countries}")
    missing_requested = sorted(set(requested_countries).difference(countries))
    if smoke and missing_requested:
        raise click.ClickException(f"Validation failed: missing requested countries {missing_requested}")
    if frac_in_0_50 < 0.99:
        raise click.ClickException(f"Validation failed: wind_speed physical fraction={frac_in_0_50:.4f}")

    return {
        "rows": rows,
        "stations": stations,
        "countries": countries,
        "by_country": by_country,
        "valid_wind_speed": int(len(ws)),
        "frac_in_0_50": frac_in_0_50,
    }


def _print_report(
    out_path: Path,
    time_index: pd.DatetimeIndex,
    report: dict,
    caveats: Counter[str],
    *,
    smoke: bool,
) -> None:
    abs_path = out_path.resolve()
    by_country = ", ".join(f"{code}={report['by_country'].get(code, 0)}" for code in ("FR", "ES", "PT"))
    caveat_text = ", ".join(f"{key}={value}" for key, value in sorted(caveats.items()) if value) or "none"
    click.echo("Verdict GREEN")
    click.echo(f"Stations ingested: {report['stations']} ({by_country})")
    click.echo(
        f"Output Zarr: {abs_path} rows={report['rows']} "
        f"valid_wind_speed={report['valid_wind_speed']} ws_frac_in_0_50={report['frac_in_0_50']:.4f}"
    )
    click.echo(f"Caveats: {caveat_text}")
    if smoke:
        click.echo(
            f"Smoke OK. NOAA ISD Zarr: {abs_path} "
            f"(T={len(time_index)} x S={report['stations']} x H=1, valid wind_speed={report['valid_wind_speed']})"
        )
    else:
        click.echo(
            f"NOAA ISD Zarr OK: {abs_path} "
            f"(T={len(time_index)} x S={report['stations']} x H=1, valid wind_speed={report['valid_wind_speed']})"
        )


if __name__ == "__main__":
    main()
