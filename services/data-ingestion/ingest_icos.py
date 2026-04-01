"""
ingest_icos.py — ICOS/FLUXNET station data ingestion for fire weather validation.

Downloads half-hourly meteorological data from ICOS Carbon Portal
(https://data.icos-cp.eu/) for target ecosystem stations in southern Europe.

These are ICOS *ecosystem* stations (eddy-covariance flux towers) that also
record standard meteorological variables (T, RH, wind speed, pressure, precip).
Data is packaged as FLUXNET Product (half-hourly CSVs inside ZIP archives).

Target stations (fire weather validation):
  - FR-Pue  Puéchabon          43.7414°N,  3.5958°E  Mediterranean holm oak
  - FR-OHP  Obs. Haute-Provence 43.9319°N,  5.7122°E  Complex terrain
  - ES-Arn  El Arenosillo       37.1047°N, -6.7333°E  SW Spain

Data access:
  Uses `icoscp_core` for authenticated downloads from the ICOS Carbon Portal.
  Requires a one-time authentication setup:
    1. Create an account at https://cpauth.icos-cp.eu/
    2. Run: python -c "from icoscp_core.icos import auth; auth.init_config_file()"
    3. Enter your email and password when prompted.
  Config stored at ~/.icoscp/cpauthToken_auth_conf.json

Output Zarr schema:
  icos_{station_id}.zarr/
    meteo/
      T        [time]  float32  °C
      RH       [time]  float32  %  (computed from VPD + T if RH not direct)
      ws       [time]  float32  m/s
      wd       [time]  float32  degrees
      p        [time]  float32  hPa (if available)
    coords/
      time     [time]  int64    ns since epoch
    attrs: station_id, lat, lon, altitude_m, heights_m

Also produces daily FWI CSV at data/validation/fwi/icos_{station_id}_fwi.csv.

Usage:
    python ingest_icos.py --stations FR-Pue --stations FR-OHP --stations ES-Arn \\
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

# Station URI prefix for ecosystem stations in ICOS
STATION_URI_PREFIX = "http://meta.icos-cp.eu/resources/stations/ES_"


# ── SPARQL: find Fluxnet Product data objects ──────────────────────────────

SPARQL_FLUXNET_QUERY = """
PREFIX cpmeta: <http://meta.icos-cp.eu/ontologies/cpmeta/>
PREFIX prov: <http://www.w3.org/ns/prov#>

SELECT ?dobj ?specLabel ?fileName ?timeStart ?timeEnd ?size ?nRows
WHERE {{
  ?dobj cpmeta:hasObjectSpec ?spec .
  ?dobj cpmeta:hasName ?fileName .
  ?dobj cpmeta:hasSizeInBytes ?size .
  ?dobj cpmeta:wasAcquiredBy [
    prov:wasAssociatedWith <{station_uri}> ;
    prov:startedAtTime ?timeStart ;
    prov:endedAtTime ?timeEnd
  ] .
  ?spec rdfs:label ?specLabel .
  OPTIONAL {{ ?dobj cpmeta:hasNumberOfRows ?nRows }}
  FILTER(?specLabel IN ('Fluxnet Product', 'ETC L2 Fluxnet (half-hourly)', 'Fluxnet Archive Product'))
  FILTER(?timeStart <= "{end_date}T23:59:59Z"^^xsd:dateTime)
  FILTER(?timeEnd   >= "{start_date}T00:00:00Z"^^xsd:dateTime)
}}
ORDER BY DESC(?timeEnd)
LIMIT 5
"""


def _find_fluxnet_dobj(station_id: str, start_date: str, end_date: str) -> list[dict]:
    """Find FLUXNET data object URIs for a station via SPARQL."""
    import requests

    station_uri = STATION_URI_PREFIX + station_id
    query = SPARQL_FLUXNET_QUERY.format(
        station_uri=station_uri,
        start_date=start_date,
        end_date=end_date,
    )
    log.info("Querying ICOS SPARQL for FLUXNET products", extra={"station": station_id})
    try:
        resp = requests.post(
            SPARQL_ENDPOINT,
            data={"query": query},
            headers={"Accept": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        bindings = resp.json().get("results", {}).get("bindings", [])
        results = []
        for r in bindings:
            results.append({
                "dobj": r["dobj"]["value"],
                "specLabel": r["specLabel"]["value"],
                "fileName": r["fileName"]["value"],
                "timeStart": r["timeStart"]["value"],
                "timeEnd": r["timeEnd"]["value"],
                "size": int(r["size"]["value"]),
            })
        log.info(
            "Found FLUXNET data objects",
            extra={"station": station_id, "n_objects": len(results)},
        )
        for r in results:
            log.info(
                "  Data object",
                extra={"spec": r["specLabel"], "file": r["fileName"], "size_mb": round(r["size"] / 1e6, 1)},
            )
        return results
    except Exception as e:
        log.warning("SPARQL query failed", extra={"error": str(e)})
        return []


# ── Data fetching ────────────────────────────────────────────────────────────


def _download_fluxnet_zip(dobj_url: str) -> bytes | None:
    """Download a FLUXNET ZIP archive from ICOS CP.

    Requires authentication via icoscp_core. If auth is not configured,
    falls back to unauthenticated download (will fail for most data).
    """
    try:
        from icoscp_core.icos import data as icos_data
        stream = icos_data.get_csv_byte_stream(dobj_url)
        content = stream.read()
        log.info("Downloaded via icoscp_core", extra={"size_mb": round(len(content) / 1e6, 1)})
        return content
    except Exception as e:
        log.warning("icoscp_core download failed (auth configured?)", extra={"error": str(e)})

    # Fallback: try direct HTTP (usually returns HTML license page)
    import requests
    try:
        resp = requests.get(
            dobj_url.replace("/meta/", "/objects/") if "/meta/" in dobj_url else dobj_url,
            timeout=300,
            stream=True,
        )
        resp.raise_for_status()
        content = resp.content
        # Check if it's actually a ZIP (starts with PK) or HTML
        if content[:2] == b"PK":
            log.info("Downloaded via direct HTTP", extra={"size_mb": round(len(content) / 1e6, 1)})
            return content
        else:
            log.warning("Direct download returned HTML (license page), not ZIP. "
                       "Please configure icoscp_core authentication.")
            return None
    except Exception as e:
        log.warning("Direct HTTP download failed", extra={"error": str(e)})
        return None


def _extract_hh_csv_from_zip(zip_bytes: bytes) -> pd.DataFrame | None:
    """Extract the half-hourly CSV from a FLUXNET ZIP archive.

    FLUXNET ZIPs contain multiple CSVs at different time resolutions.
    We want the HH (half-hourly) or HR (hourly) file with FULLSET or FLUXMET.
    """
    import zipfile
    from io import BytesIO

    try:
        with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            log.info("ZIP contents", extra={"n_files": len(names), "files": names[:10]})

            # Priority order for file selection
            hh_candidates = [n for n in names if "_HH_" in n and n.endswith(".csv")]
            hr_candidates = [n for n in names if "_HR_" in n and n.endswith(".csv")]

            # Prefer FULLSET or FLUXMET over other variants
            for candidates in [hh_candidates, hr_candidates]:
                for preferred in ["FULLSET", "FLUXMET"]:
                    for c in candidates:
                        if preferred in c:
                            log.info("Selected CSV", extra={"file": c})
                            with zf.open(c) as f:
                                return pd.read_csv(f)

            # Fallback: any HH or HR CSV
            for candidates in [hh_candidates, hr_candidates]:
                if candidates:
                    csv_name = candidates[0]
                    log.info("Selected CSV (fallback)", extra={"file": csv_name})
                    with zf.open(csv_name) as f:
                        return pd.read_csv(f)

            # Last resort: any CSV
            csv_files = [n for n in names if n.endswith(".csv")]
            if csv_files:
                csv_name = csv_files[0]
                log.info("Selected CSV (last resort)", extra={"file": csv_name})
                with zf.open(csv_name) as f:
                    return pd.read_csv(f)

            log.warning("No CSV files found in ZIP", extra={"files": names})
            return None
    except zipfile.BadZipFile:
        log.warning("Invalid ZIP file")
        return None


def _fetch_station_data(
    station_id: str, start_date: str, end_date: str
) -> pd.DataFrame | None:
    """Fetch FLUXNET data for an ICOS ecosystem station.

    Strategy:
    1. Find FLUXNET Product data objects via SPARQL
    2. Download the ZIP archive (requires icoscp_core auth)
    3. Extract the half-hourly CSV
    4. Harmonize variable names and filter to requested period

    Returns a harmonized DataFrame or None.
    """
    results = _find_fluxnet_dobj(station_id, start_date, end_date)
    if not results:
        log.warning("No FLUXNET data objects found", extra={"station": station_id})
        return None

    t_start = pd.Timestamp(start_date, tz="UTC")
    t_end = pd.Timestamp(end_date, tz="UTC")

    # Try each data object (prefer Fluxnet Product over Archive)
    for r in results:
        log.info("Trying data object", extra={
            "station": station_id,
            "spec": r["specLabel"],
            "file": r["fileName"],
        })

        # Get the access URL
        access_url = r["dobj"].replace("meta.icos-cp.eu", "data.icos-cp.eu")

        zip_bytes = _download_fluxnet_zip(access_url)
        if zip_bytes is None:
            continue

        df = _extract_hh_csv_from_zip(zip_bytes)
        if df is None:
            continue

        return _harmonize_dataframe(df, t_start, t_end)

    return None


def _harmonize_dataframe(
    df: pd.DataFrame, t_start: pd.Timestamp, t_end: pd.Timestamp
) -> pd.DataFrame | None:
    """Harmonize FLUXNET column names and filter to time range.

    FLUXNET standard columns:
      TIMESTAMP_START / TIMESTAMP — YYYYMMDDHHmm format
      TA_F / TA_F_MDS — air temperature (°C)
      RH — relative humidity (%)  (not always present)
      VPD_F / VPD_F_MDS — vapor pressure deficit (hPa)
      WS_F / WS — wind speed (m/s)
      WD — wind direction (degrees)
      PA_F / PA — atmospheric pressure (kPa)
      P_F / P — precipitation (mm per timestep)

    If RH is missing, computes it from VPD and TA using Buck equation.
    Resamples to hourly.
    """
    if df.empty:
        return None

    log.info("Harmonizing dataframe", extra={
        "shape": list(df.shape),
        "columns": list(df.columns)[:30],
    })

    # ── Parse timestamps ────────────────────────────────────────────────
    time_col = None
    for candidate in ["TIMESTAMP_START", "TIMESTAMP", "timestamp", "time", "datetime"]:
        if candidate in df.columns:
            time_col = candidate
            break
    if time_col is None:
        log.warning("No time column found", extra={"columns": list(df.columns)[:20]})
        return None

    # FLUXNET timestamps are in YYYYMMDDHHmm format (int64)
    ts_raw = df[time_col]
    if ts_raw.dtype in (np.int64, np.int32, np.float64):
        # Convert YYYYMMDDHHmm to datetime
        df["time"] = pd.to_datetime(ts_raw.astype(np.int64).astype(str), format="%Y%m%d%H%M", utc=True, errors="coerce")
    else:
        df["time"] = pd.to_datetime(ts_raw, utc=True, errors="coerce")

    df = df.dropna(subset=["time"])
    df = df.set_index("time").sort_index()

    # Filter to requested period
    df = df[t_start:t_end]
    if df.empty:
        log.warning("No data in requested period after filtering")
        return None

    # ── Map variables ───────────────────────────────────────────────────
    result = pd.DataFrame(index=df.index)

    # Temperature
    for col in ["TA_F", "TA_F_MDS", "TA", "air_temperature"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            vals = vals.where(vals > -9990)  # FLUXNET missing value = -9999
            result["T"] = vals.astype(np.float32)
            break

    # Relative humidity — direct or from VPD
    rh_found = False
    for col in ["RH", "rh", "relative_humidity"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            vals = vals.where(vals > -9990)
            result["RH"] = vals.astype(np.float32)
            rh_found = True
            break

    if not rh_found and "T" in result.columns:
        # Compute RH from VPD: RH = 100 * (1 - VPD / e_s(T))
        # Buck equation: e_s(T) = 0.61121 * exp((18.678 - T/234.5) * T/(257.14 + T)) [kPa]
        for vpd_col in ["VPD_F", "VPD_F_MDS", "VPD"]:
            if vpd_col in df.columns:
                vpd = pd.to_numeric(df[vpd_col], errors="coerce")
                vpd = vpd.where(vpd > -9990)
                t_c = result["T"]
                # FLUXNET VPD is in hPa, convert to kPa
                vpd_kpa = vpd * 0.1
                e_sat = 0.61121 * np.exp((18.678 - t_c / 234.5) * t_c / (257.14 + t_c))
                rh = 100.0 * (1.0 - vpd_kpa / e_sat)
                result["RH"] = np.clip(rh, 0, 100).astype(np.float32)
                log.info("Computed RH from VPD", extra={"vpd_col": vpd_col})
                break

    # Wind speed
    for col in ["WS_F", "WS", "wind_speed"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            vals = vals.where(vals > -9990)
            result["ws"] = vals.astype(np.float32)
            break

    # Wind direction
    for col in ["WD", "wd", "wind_direction"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            vals = vals.where(vals > -9990)
            result["wd"] = vals.astype(np.float32)
            break

    # Pressure (FLUXNET: PA in kPa → convert to hPa)
    for col in ["PA_F", "PA", "air_pressure"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            vals = vals.where(vals > -9990)
            # FLUXNET PA is in kPa, convert to hPa
            result["p"] = (vals * 10.0).astype(np.float32)
            break

    if result.empty or len(result.columns) == 0:
        log.warning(
            "No recognized variables found",
            extra={"available_columns": list(df.columns)[:30]},
        )
        return None

    available = list(result.columns)
    n_valid = {v: int(result[v].notna().sum()) for v in available}
    log.info("Variables mapped", extra={"variables": available, "valid_counts": n_valid})

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
    """Download ICOS/FLUXNET station data for fire weather validation.

    Fetches hourly T, RH, wind speed/direction, and pressure from the
    ICOS Carbon Portal (ecosystem stations with FLUXNET data products).
    Saves as Zarr stores and optionally computes daily FWI indices.

    Requires icoscp_core authentication. Setup:

    \b
        pip install icoscp
        python -c "from icoscp_core.icos import auth; auth.init_config_file()"
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
            "station_name": meta["name"],
            "lat": meta["lat"],
            "lon": meta["lon"],
        })

        # Fetch FLUXNET data via SPARQL + icoscp_core download
        df = _fetch_station_data(sid, start, end)

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
            "Check that icoscp_core authentication is configured: "
            "python -c 'from icoscp_core.icos import auth; auth.init_config_file()'"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
