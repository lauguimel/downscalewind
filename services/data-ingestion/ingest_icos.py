"""
ingest_icos.py — ICOS atmospheric station data ingestion for fire weather validation.

Downloads hourly meteorological data from ICOS Carbon Portal
(https://data.icos-cp.eu/) for target stations in southern Europe.

Target stations (fire weather validation):
  - FR-Pue  Puéchabon          43.7414°N,  3.5958°E  Mediterranean holm oak
  - FR-OHP  Obs. Haute-Provence 43.9319°N,  5.7122°E  Complex terrain
  - ES-Arn  El Arenosillo       37.1047°N, -6.7333°E  SW Spain

Data access:
  Uses the `icoscp` Python package (pip install icoscp) for metadata and
  data object retrieval. Falls back to direct SPARQL queries if not available.

Output Zarr schema:
  icos_{station_id}.zarr/
    meteo/
      T        [time]  float32  °C
      RH       [time]  float32  %
      ws       [time]  float32  m/s
      wd       [time]  float32  degrees
      p        [time]  float32  hPa (if available)
    coords/
      time     [time]  int64    ns since epoch
    attrs: station_id, lat, lon, altitude_m, heights_m

Also produces daily FWI CSV at data/validation/fwi/icos_{station_id}_fwi.csv.

Usage:
    python ingest_icos.py --stations FR-Pue FR-OHP ES-Arn \\
        --start 2017-06-01 --end 2017-08-31 \\
        --output-dir ../../data/raw
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import click
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.logging_config import get_logger
from shared.fwi import compute_fwi_series

log = get_logger("ingest_icos")


# ── Station metadata ─────────────────────────────────────────────────────────

ICOS_STATIONS: dict[str, dict] = {
    "FR-Pue": {
        "name": "Puéchabon",
        "lat": 43.7414,
        "lon": 3.5958,
        "altitude_m": 270.0,
        "country": "FR",
        "description": "Mediterranean holm oak forest",
        "heights_m": [2.0, 10.0],
    },
    "FR-OHP": {
        "name": "Observatoire de Haute-Provence",
        "lat": 43.9319,
        "lon": 5.7122,
        "altitude_m": 680.0,
        "country": "FR",
        "description": "Complex terrain, observatory",
        "heights_m": [10.0, 50.0, 100.0],
    },
    "ES-Arn": {
        "name": "El Arenosillo",
        "lat": 37.1047,
        "lon": -6.7333,
        "altitude_m": 41.0,
        "country": "ES",
        "description": "Southwestern Spain, coastal pine forest",
        "heights_m": [10.0, 50.0],
    },
}


# ── ICOS variable mapping ───────────────────────────────────────────────────

# ICOS atmospheric stations typically provide these variables.
# Exact column names depend on the data product level.
VARIABLE_MAP = {
    "T": {
        "icos_names": ["TA", "Ta", "air_temperature", "t_air"],
        "units": "degC",
        "long_name": "air temperature",
    },
    "RH": {
        "icos_names": ["RH", "rh", "relative_humidity"],
        "units": "%",
        "long_name": "relative humidity",
    },
    "ws": {
        "icos_names": ["WS", "ws", "wind_speed"],
        "units": "m s-1",
        "long_name": "wind speed",
    },
    "wd": {
        "icos_names": ["WD", "wd", "wind_direction"],
        "units": "degrees",
        "long_name": "wind direction (from north, clockwise)",
    },
    "p": {
        "icos_names": ["PA", "Pa", "air_pressure", "p_air"],
        "units": "hPa",
        "long_name": "air pressure",
    },
}


# ── SPARQL fallback ──────────────────────────────────────────────────────────

SPARQL_ENDPOINT = "https://meta.icos-cp.eu/sparql"

SPARQL_QUERY_TEMPLATE = """
PREFIX cpmeta: <http://meta.icos-cp.eu/ontologies/cpmeta/>
PREFIX prov: <http://www.w3.org/ns/prov#>

SELECT ?dobj ?spec ?fileName ?timeStart ?timeEnd
WHERE {{
  ?dobj cpmeta:hasObjectSpec ?spec .
  ?dobj cpmeta:hasName ?fileName .
  ?dobj cpmeta:hasSizeInBytes ?size .
  ?dobj cpmeta:wasAcquiredBy [
    prov:wasAssociatedWith ?station ;
    prov:startedAtTime ?timeStart ;
    prov:endedAtTime ?timeEnd
  ] .
  ?station cpmeta:hasStationId "{station_id}" .
  FILTER(?timeStart <= "{end_date}T23:59:59Z"^^xsd:dateTime)
  FILTER(?timeEnd   >= "{start_date}T00:00:00Z"^^xsd:dateTime)
  FILTER(CONTAINS(LCASE(STR(?spec)), "atmo"))
}}
ORDER BY DESC(?timeEnd)
LIMIT 20
"""


def _query_sparql(station_id: str, start_date: str, end_date: str) -> list[dict]:
    """Query ICOS SPARQL endpoint for data object URIs."""
    try:
        import requests
    except ImportError:
        log.error("requests package required for SPARQL fallback")
        return []

    query = SPARQL_QUERY_TEMPLATE.format(
        station_id=station_id,
        start_date=start_date,
        end_date=end_date,
    )
    log.info("Querying ICOS SPARQL endpoint", extra={"station": station_id})
    try:
        resp = requests.post(
            SPARQL_ENDPOINT,
            data={"query": query},
            headers={"Accept": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        results = resp.json().get("results", {}).get("bindings", [])
        return [
            {
                "dobj": r["dobj"]["value"],
                "fileName": r["fileName"]["value"],
                "timeStart": r["timeStart"]["value"],
                "timeEnd": r["timeEnd"]["value"],
            }
            for r in results
        ]
    except Exception as e:
        log.warning("SPARQL query failed", extra={"error": str(e)})
        return []


# ── Data fetching ────────────────────────────────────────────────────────────


def _try_icoscp_download(
    station_id: str, start_date: str, end_date: str
) -> pd.DataFrame | None:
    """Attempt to download data using the icoscp package.

    Returns a DataFrame with columns: time, T, RH, ws, wd, p (where available).
    Returns None if icoscp is not installed or data retrieval fails.
    """
    try:
        from icoscp.station import station as icos_station  # type: ignore[import]
    except ImportError:
        log.info("icoscp not installed, will use SPARQL fallback")
        return None

    try:
        log.info("Fetching via icoscp", extra={"station": station_id})
        st = icos_station.get(station_id)
        if st is None:
            log.warning("Station not found via icoscp", extra={"station": station_id})
            return None

        # List available data objects for the station
        data_objects = st.data()
        if data_objects is None or (hasattr(data_objects, "empty") and data_objects.empty):
            log.warning("No data objects found", extra={"station": station_id})
            return None

        log.info(
            "Found data objects via icoscp",
            extra={"station": station_id, "n_objects": len(data_objects)},
        )

        # Filter to atmospheric meteorological data overlapping the period
        t_start = pd.Timestamp(start_date)
        t_end = pd.Timestamp(end_date)

        # Try to load data objects that cover our period
        from icoscp.cpb.dobj import Dobj  # type: ignore[import]

        frames: list[pd.DataFrame] = []
        for _, row in data_objects.iterrows():
            try:
                dobj_uri = row.get("dobj", row.get("uri", ""))
                if not dobj_uri:
                    continue
                d = Dobj(dobj_uri)
                if d.data is not None:
                    df = d.data
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        frames.append(df)
            except Exception as e:
                log.debug("Could not load data object", extra={"error": str(e)})
                continue

        if not frames:
            log.warning("No data frames loaded via icoscp", extra={"station": station_id})
            return None

        df_all = pd.concat(frames, ignore_index=True)
        return _harmonize_dataframe(df_all, t_start, t_end)

    except Exception as e:
        log.warning("icoscp download failed", extra={"station": station_id, "error": str(e)})
        return None


def _try_sparql_download(
    station_id: str, start_date: str, end_date: str
) -> pd.DataFrame | None:
    """Attempt to download data via SPARQL + direct CSV/NetCDF download.

    Returns a harmonized DataFrame or None.
    """
    results = _query_sparql(station_id, start_date, end_date)
    if not results:
        log.warning("No SPARQL results", extra={"station": station_id})
        return None

    try:
        import requests
    except ImportError:
        log.error("requests package required for direct download")
        return None

    t_start = pd.Timestamp(start_date)
    t_end = pd.Timestamp(end_date)

    frames: list[pd.DataFrame] = []
    for r in results:
        dobj_url = r["dobj"]
        # ICOS data portal provides CSV download at /csv endpoint
        csv_url = dobj_url.replace("/meta/", "/objects/") if "/meta/" in dobj_url else dobj_url
        log.info("Downloading data object", extra={"url": csv_url, "file": r["fileName"]})

        try:
            resp = requests.get(csv_url, timeout=120, stream=True)
            resp.raise_for_status()

            # Try CSV parse (most ICOS Level 2 are CSV)
            from io import StringIO

            content = resp.text
            # Skip ICOS header lines (start with #)
            lines = content.split("\n")
            data_lines = [l for l in lines if not l.startswith("#")]
            if not data_lines:
                continue

            df = pd.read_csv(StringIO("\n".join(data_lines)), sep=None, engine="python")
            if not df.empty:
                frames.append(df)
        except Exception as e:
            log.debug("Failed to download/parse data object", extra={"error": str(e)})
            continue

    if not frames:
        return None

    df_all = pd.concat(frames, ignore_index=True)
    return _harmonize_dataframe(df_all, t_start, t_end)


def _harmonize_dataframe(
    df: pd.DataFrame, t_start: pd.Timestamp, t_end: pd.Timestamp
) -> pd.DataFrame | None:
    """Harmonize column names and filter to time range.

    Maps ICOS variable names to standard names (T, RH, ws, wd, p).
    Resamples to hourly if needed.
    """
    if df.empty:
        return None

    # Identify time column
    time_col = None
    for candidate in ["TIMESTAMP", "timestamp", "time", "Time", "datetime", "date"]:
        if candidate in df.columns:
            time_col = candidate
            break
    if time_col is None:
        # Try first column if it looks like a timestamp
        first_col = df.columns[0]
        try:
            pd.to_datetime(df[first_col].iloc[:5])
            time_col = first_col
        except (ValueError, TypeError):
            log.warning("No time column found in dataframe", extra={"columns": list(df.columns)})
            return None

    df["time"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=["time"])
    df = df.set_index("time").sort_index()

    # Filter to requested period
    df = df[t_start:t_end]
    if df.empty:
        log.warning("No data in requested period after filtering")
        return None

    # Map variable names
    result = pd.DataFrame(index=df.index)
    for std_name, info in VARIABLE_MAP.items():
        for icos_name in info["icos_names"]:
            if icos_name in df.columns:
                result[std_name] = pd.to_numeric(df[icos_name], errors="coerce").astype(np.float32)
                break

    if result.empty or len(result.columns) == 0:
        log.warning(
            "No recognized variables found",
            extra={"available_columns": list(df.columns)[:20]},
        )
        return None

    # Resample to hourly (mean)
    result = result.resample("1h").mean()

    # Reset index for clean output
    result = result.reset_index()
    result.rename(columns={"index": "time"}, inplace=True)
    if "time" not in result.columns:
        result = result.rename_axis("time").reset_index()

    return result


# ── Zarr writer ──────────────────────────────────────────────────────────────


def _write_zarr(
    df: pd.DataFrame,
    station_id: str,
    station_meta: dict,
    output_dir: Path,
) -> Path:
    """Write station data to Zarr store."""
    import zarr

    zarr_path = output_dir / f"icos_{station_id.replace('-', '_').lower()}.zarr"
    zarr_path.parent.mkdir(parents=True, exist_ok=True)

    times = pd.to_datetime(df["time"]).values.astype("datetime64[ns]")
    n_times = len(times)

    log.info("Writing Zarr", extra={
        "path": str(zarr_path),
        "n_times": n_times,
        "station": station_id,
    })

    root = zarr.open_group(str(zarr_path), mode="w")

    # Coordinates
    coords = root.require_group("coords")
    arr = coords.create_array("time", shape=(n_times,), dtype=np.int64,
                              chunks=(n_times,), overwrite=True)
    arr[:] = times.astype(np.int64)
    arr.attrs.update({"long_name": "time UTC", "units": "ns since epoch"})

    # Meteo group
    meteo = root.require_group("meteo")
    for var_name, info in VARIABLE_MAP.items():
        if var_name in df.columns:
            vals = df[var_name].values.astype(np.float32)
            arr = meteo.create_array(var_name, shape=(n_times,), dtype=np.float32,
                                     chunks=(min(720, n_times),), overwrite=True)
            arr[:] = vals
            arr.attrs.update({
                "long_name": info["long_name"],
                "units": info["units"],
            })

    # Global attributes
    root.attrs.update({
        "Conventions": "CF-1.9",
        "title": f"ICOS atmospheric station — {station_id} ({station_meta['name']})",
        "station_id": station_id,
        "lat": station_meta["lat"],
        "lon": station_meta["lon"],
        "altitude_m": station_meta["altitude_m"],
        "heights_m": station_meta["heights_m"],
        "description": station_meta["description"],
        "source": "ICOS Carbon Portal (https://data.icos-cp.eu/)",
        "time_resolution": "1h",
    })

    return zarr_path


# ── FWI computation ──────────────────────────────────────────────────────────


def _compute_daily_fwi(
    df: pd.DataFrame,
    station_id: str,
    output_dir: Path,
) -> Path | None:
    """Compute daily noon FWI from hourly station data.

    Requires at minimum: T, RH, ws columns.
    Rain is set to 0 (ICOS atmospheric stations do not measure precipitation —
    ERA5 precipitation should be used for operational FWI).

    Returns path to CSV or None if insufficient data.
    """
    required = {"T", "RH", "ws"}
    available = set(df.columns) & required
    if available != required:
        missing = required - available
        log.warning(
            "Cannot compute FWI: missing variables",
            extra={"station": station_id, "missing": list(missing)},
        )
        return None

    # Extract noon values (11:00–13:00 UTC mean as proxy for local noon)
    df_ts = df.copy()
    df_ts["time"] = pd.to_datetime(df_ts["time"], utc=True)
    df_ts = df_ts.set_index("time")

    # Select hours 11-13 UTC and take daily mean
    noon_mask = df_ts.index.hour.isin([11, 12, 13])
    df_noon = df_ts[noon_mask].resample("1D").mean()
    df_noon = df_noon.dropna(subset=["T", "RH", "ws"])

    if df_noon.empty:
        log.warning("No noon data available for FWI", extra={"station": station_id})
        return None

    # Prepare FWI inputs
    t_c = df_noon["T"].values.astype(np.float64)
    rh = np.clip(df_noon["RH"].values.astype(np.float64), 0, 100)
    ws_kmh = df_noon["ws"].values.astype(np.float64) * 3.6  # m/s → km/h
    # No precipitation from ICOS — use zero (conservative, underestimates moisture codes)
    rain_mm = np.zeros_like(t_c)
    months = df_noon.index.month.values.astype(np.int32)

    fwi_results = compute_fwi_series(t_c, rh, ws_kmh, rain_mm, months)

    # Build output DataFrame
    fwi_df = pd.DataFrame({
        "date": df_noon.index.date,
        "T_noon_C": np.round(t_c, 1),
        "RH_noon_pct": np.round(rh, 1),
        "ws_noon_kmh": np.round(ws_kmh, 1),
        "rain_mm": rain_mm,
        "FFMC": np.round(fwi_results["ffmc"], 1),
        "DMC": np.round(fwi_results["dmc"], 1),
        "DC": np.round(fwi_results["dc"], 1),
        "ISI": np.round(fwi_results["isi"], 1),
        "BUI": np.round(fwi_results["bui"], 1),
        "FWI": np.round(fwi_results["fwi"], 1),
    })

    fwi_dir = output_dir / "fwi"
    fwi_dir.mkdir(parents=True, exist_ok=True)
    csv_path = fwi_dir / f"icos_{station_id.replace('-', '_').lower()}_fwi.csv"
    fwi_df.to_csv(csv_path, index=False)

    log.info("FWI CSV written", extra={
        "path": str(csv_path),
        "n_days": len(fwi_df),
        "fwi_mean": round(float(fwi_df["FWI"].mean()), 1),
        "fwi_max": round(float(fwi_df["FWI"].max()), 1),
    })

    return csv_path


# ── CLI ──────────────────────────────────────────────────────────────────────


@click.command()
@click.option(
    "--stations", "-s",
    multiple=True,
    default=list(ICOS_STATIONS.keys()),
    show_default=True,
    help="ICOS station IDs to download (can specify multiple times)",
)
@click.option(
    "--start",
    default="2017-06-01",
    show_default=True,
    help="Start date (YYYY-MM-DD)",
)
@click.option(
    "--end",
    default="2017-08-31",
    show_default=True,
    help="End date (YYYY-MM-DD)",
)
@click.option(
    "--output-dir",
    default=str(Path(__file__).resolve().parents[2] / "data" / "raw"),
    show_default=True,
    help="Output directory for Zarr stores",
)
@click.option(
    "--fwi-dir",
    default=str(Path(__file__).resolve().parents[2] / "data" / "validation"),
    show_default=True,
    help="Output directory for FWI CSV files",
)
@click.option(
    "--skip-fwi",
    is_flag=True,
    default=False,
    help="Skip FWI computation",
)
def main(
    stations: tuple[str, ...],
    start: str,
    end: str,
    output_dir: str,
    fwi_dir: str,
    skip_fwi: bool,
):
    """Download ICOS atmospheric station data for fire weather validation.

    Fetches hourly T, RH, wind speed/direction, and pressure from the
    ICOS Carbon Portal. Saves as Zarr stores and optionally computes
    daily FWI indices.

    Tries the `icoscp` Python package first, falls back to SPARQL queries.

    Install icoscp:
        pip install icoscp
    """
    output_path = Path(output_dir)
    fwi_path = Path(fwi_dir)

    log.info("Starting ICOS ingestion", extra={
        "stations": list(stations),
        "period": f"{start} to {end}",
        "output_dir": str(output_path),
    })

    # Validate station IDs
    for sid in stations:
        if sid not in ICOS_STATIONS:
            known = list(ICOS_STATIONS.keys())
            log.error(
                f"Unknown station '{sid}'. Known stations: {known}",
                extra={"station": sid},
            )
            sys.exit(1)

    summary: dict[str, dict] = {}

    for sid in stations:
        meta = ICOS_STATIONS[sid]
        log.info("Processing station", extra={
            "station": sid,
            "name": meta["name"],
            "lat": meta["lat"],
            "lon": meta["lon"],
        })

        # Try icoscp first, then SPARQL fallback
        df = _try_icoscp_download(sid, start, end)
        if df is None:
            df = _try_sparql_download(sid, start, end)

        if df is None or df.empty:
            log.warning(
                "No data retrieved for station. "
                "This may be because the station does not have data for the "
                "requested period, or data access requires authentication. "
                "Visit https://data.icos-cp.eu/ to check data availability.",
                extra={"station": sid},
            )
            summary[sid] = {"status": "NO_DATA"}
            continue

        # Count available variables
        available_vars = [v for v in VARIABLE_MAP if v in df.columns]
        n_valid = {v: int(df[v].notna().sum()) for v in available_vars}

        log.info("Data retrieved", extra={
            "station": sid,
            "n_hours": len(df),
            "variables": available_vars,
            "valid_counts": n_valid,
        })

        # Write Zarr
        zarr_path = _write_zarr(df, sid, meta, output_path)

        # Compute FWI
        fwi_csv = None
        if not skip_fwi:
            fwi_csv = _compute_daily_fwi(df, sid, fwi_path)

        summary[sid] = {
            "status": "OK",
            "n_hours": len(df),
            "variables": available_vars,
            "zarr": str(zarr_path),
            "fwi_csv": str(fwi_csv) if fwi_csv else None,
        }

    # ── Final summary ────────────────────────────────────────────────────────
    print("\n=== ICOS Ingestion Summary ===")
    for sid, info in summary.items():
        meta = ICOS_STATIONS[sid]
        status = info["status"]
        if status == "OK":
            print(
                f"  {sid} ({meta['name']}): {info['n_hours']} hours, "
                f"vars={info['variables']}"
            )
            print(f"    Zarr: {info['zarr']}")
            if info.get("fwi_csv"):
                print(f"    FWI:  {info['fwi_csv']}")
        else:
            print(f"  {sid} ({meta['name']}): {status}")

    n_ok = sum(1 for v in summary.values() if v["status"] == "OK")
    print(f"\n{n_ok}/{len(stations)} stations ingested successfully.")

    if n_ok == 0:
        log.warning(
            "No stations were successfully ingested. "
            "Try installing icoscp: pip install icoscp"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
