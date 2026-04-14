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

# Station network:
#   type="ES" → ICOS Ecosystem station (FLUXNET product, T/RH/WS/WD/precip at 1-2 levels)
#   type="AS" → ICOS Atmosphere tall-tower (ATC Meteo L2, multi-level wind 10-200 m)
#
# Selection criteria for fire weather + wind downscaling validation:
#   - Mediterranean / complex-terrain ecosystem sites (FWI obs with precip)
#   - European tall-towers with multi-level wind (FuXi-CFD benchmark sites)
ICOS_STATIONS: dict[str, dict] = {
    # ── ICOS Ecosystem — Mediterranean / fire-prone ──────────────────────────
    "FR-Pue": {"type": "ES", "name": "Puéchabon",            "lat": 43.7414, "lon":  3.5958, "altitude_m":  270.0, "country": "FR", "description": "Mediterranean holm oak forest", "heights_m": [2.0, 10.0]},
    "FR-OHP": {"type": "ES", "name": "Obs. Haute-Provence",  "lat": 43.9319, "lon":  5.7122, "altitude_m":  680.0, "country": "FR", "description": "Complex terrain observatory",  "heights_m": [10.0, 50.0, 100.0]},
    "ES-Arn": {"type": "ES", "name": "El Arenosillo",        "lat": 37.1047, "lon": -6.7333, "altitude_m":   41.0, "country": "ES", "description": "SW Spain, coastal pine",       "heights_m": [10.0, 50.0]},
    "ES-LJu": {"type": "ES", "name": "Llano de los Juanes",  "lat": 36.9266, "lon": -2.7521, "altitude_m": 1600.0, "country": "ES", "description": "Sierra Nevada, shrubland",     "heights_m": [3.0]},
    "ES-Cnd": {"type": "ES", "name": "Conde",                "lat": 37.9148, "lon": -3.2277, "altitude_m":  790.0, "country": "ES", "description": "Mediterranean savanna",        "heights_m": [5.0]},
    "PT-Mi1": {"type": "ES", "name": "Mitra IV Tojal",       "lat": 38.5412, "lon": -8.0001, "altitude_m":  243.0, "country": "PT", "description": "Mediterranean grassland",      "heights_m": [2.0]},
    "IT-Cp2": {"type": "ES", "name": "Castelporziano 2",     "lat": 41.7042, "lon": 12.3572, "altitude_m":   19.0, "country": "IT", "description": "Mediterranean holm oak (Rome)","heights_m": [10.0, 25.0]},
    "IT-Noe": {"type": "ES", "name": "Arca di Noè (Sardinia)","lat": 40.6062, "lon":  8.1512, "altitude_m":   25.0, "country": "IT", "description": "Mediterranean maquis",         "heights_m": [7.0]},
    # ── ICOS Ecosystem — Alpine / complex terrain ─────────────────────────────
    "CH-Dav": {"type": "ES", "name": "Davos",                "lat": 46.8153, "lon":  9.8559, "altitude_m": 1639.0, "country": "CH", "description": "Swiss Alps, conifer forest",   "heights_m": [35.0]},
    "IT-Ren": {"type": "ES", "name": "Renon (Ritten)",       "lat": 46.5869, "lon": 11.4337, "altitude_m": 1735.0, "country": "IT", "description": "Italian Alps conifer",         "heights_m": [32.0]},
    "IT-Lav": {"type": "ES", "name": "Lavarone",             "lat": 45.9553, "lon": 11.2812, "altitude_m": 1353.0, "country": "IT", "description": "Pre-Alps conifer",             "heights_m": [29.0]},
    "IT-MBo": {"type": "ES", "name": "Monte Bondone",        "lat": 46.0147, "lon": 11.0458, "altitude_m": 1550.0, "country": "IT", "description": "Alpine grassland",             "heights_m": [2.5]},
    "FR-LBr": {"type": "ES", "name": "Le Bray",              "lat": 44.7171, "lon": -0.7693, "altitude_m":   61.0, "country": "FR", "description": "Landes pine forest",           "heights_m": [26.0]},
    # ── ICOS Atmosphere — European tall-towers (FuXi-CFD benchmark sites) ────
    "OPE":    {"type": "AS", "name": "Houdelaincourt OPE",   "lat": 48.5619, "lon":  5.5036, "altitude_m":  395.0, "country": "FR", "description": "ANDRA tall tower (10/50/120 m)","heights_m": [10.0, 50.0, 120.0]},
    "IPR":    {"type": "AS", "name": "Ispra JRC",            "lat": 45.8126, "lon":  8.6360, "altitude_m":  210.0, "country": "IT", "description": "JRC tall tower (northern Italy)","heights_m": [40.0, 60.0, 100.0]},
    "HPB":    {"type": "AS", "name": "Hohenpeißenberg",      "lat": 47.8011, "lon": 11.0246, "altitude_m":  934.0, "country": "DE", "description": "Bavarian pre-Alps tall tower", "heights_m": [50.0, 93.0, 131.0]},
    "JFJ":    {"type": "AS", "name": "Jungfraujoch",         "lat": 46.5475, "lon":  7.9851, "altitude_m": 3572.0, "country": "CH", "description": "Swiss high-alpine (3572 m)",   "heights_m": [5.0, 14.0]},
    "PUY":    {"type": "AS", "name": "Puy de Dôme",          "lat": 45.7722, "lon":  2.9658, "altitude_m": 1465.0, "country": "FR", "description": "Massif Central summit",        "heights_m": [10.0]},
    "TRN":    {"type": "AS", "name": "Trainou",              "lat": 47.9647, "lon":  2.1125, "altitude_m":  131.0, "country": "FR", "description": "Loire valley tall tower",     "heights_m": [5.0, 50.0, 100.0, 180.0]},
    "SAC":    {"type": "AS", "name": "Saclay",               "lat": 48.7227, "lon":  2.1420, "altitude_m":  160.0, "country": "FR", "description": "Paris basin tall tower",       "heights_m": [15.0, 60.0, 100.0]},
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

# Station URI prefix depends on station type:
#   ES_ → Ecosystem station (FLUXNET products)
#   AS_ → Atmosphere station (ATC Meteo Level 2 products with multi-level wind)
STATION_URI_PREFIXES = {
    "ES": "http://meta.icos-cp.eu/resources/stations/ES_",
    "AS": "http://meta.icos-cp.eu/resources/stations/AS_",
}


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

# ICOS Atmosphere Meteo L2 — CSV tables with multi-level wind, T, RH, p, precip
# Spec labels observed on ICOS CP:
#   'ICOS ATC Meteo Release'
#   'ICOS ATC Meteo NRT Growing Time Series'
#   'ICOS ATC Meteo Release (Level 2)'
# SPARQL for ICOS Atmosphere Meteo Level 2 (processed, calibrated, distributable).
# Also accepts NRT growing time series (most recent data, near-real-time).
# Raw (uncalibrated) data is explicitly excluded — ICOS does not distribute it.
SPARQL_ATMOS_METEO_QUERY = """
PREFIX cpmeta: <http://meta.icos-cp.eu/ontologies/cpmeta/>
PREFIX prov: <http://www.w3.org/ns/prov#>

SELECT ?dobj ?specLabel ?fileName ?timeStart ?timeEnd ?size
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
  FILTER(
    ?specLabel IN (
      'ICOS ATC Meteo Release',
      'ICOS ATC meteo  release',
      'ICOS ATC NRT Meteo growing time series'
    )
  )
  FILTER(?timeStart <= "{end_date}T23:59:59Z"^^xsd:dateTime)
  FILTER(?timeEnd   >= "{start_date}T00:00:00Z"^^xsd:dateTime)
  FILTER NOT EXISTS {{ ?dobj cpmeta:isNextVersionOf ?other }}
}}
ORDER BY DESC(?timeEnd)
LIMIT 20
"""


def _find_fluxnet_dobj(station_id: str, start_date: str, end_date: str) -> list[dict]:
    """Find FLUXNET data object URIs for a station via SPARQL."""
    import requests

    station_uri = STATION_URI_PREFIXES["ES"] + station_id
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


_ICOS_DATA_CLIENT = None


def _icos_data_client():
    """Build (and cache) an ICOS data client using cookie-token auth.

    Reads ~/.icoscp/cpauthToken_auth_conf.json — if a 'token' field is
    present, uses bootstrap.fromCookieToken() (no password needed).
    Otherwise falls back to the default `data` module which expects an
    interactive credentials file.
    """
    global _ICOS_DATA_CLIENT
    if _ICOS_DATA_CLIENT is not None:
        return _ICOS_DATA_CLIENT
    import json
    from pathlib import Path as _P

    conf_path = _P.home() / ".icoscp" / "cpauthToken_auth_conf.json"
    if conf_path.exists():
        try:
            conf = json.loads(conf_path.read_text())
            token = conf.get("token")
            if token:
                from icoscp_core.icos import bootstrap
                # icoscp_core expects the cookie *value* in the form
                # 'cpauthToken=<token>' (it parses the '=' separator).
                cookie_value = token if token.startswith("cpauthToken=") else f"cpauthToken={token}"
                _meta, _ICOS_DATA_CLIENT = bootstrap.fromCookieToken(cookie_value)
                log.info("ICOS auth via cookie token", extra={"user": conf.get("user_id")})
                return _ICOS_DATA_CLIENT
        except Exception as e:
            log.warning("Cookie-token auth failed", extra={"error": str(e)})

    # Fallback: default module-level client (uses ~/.icoscp credentials file)
    from icoscp_core.icos import data as icos_data
    _ICOS_DATA_CLIENT = icos_data
    return _ICOS_DATA_CLIENT


def _download_fluxnet_zip(dobj_url: str) -> bytes | None:
    """Download a FLUXNET ZIP archive from ICOS CP.

    Requires authentication via icoscp_core. Uses get_file_stream (raw
    binary download, returns (stream, filename) tuple).
    """
    try:
        client = _icos_data_client()
        result = client.get_file_stream(dobj_url)
        # get_file_stream returns (filename: str, stream: HTTPResponse)
        stream = result[1] if isinstance(result, tuple) else result
        content = stream.read()
        log.info("Downloaded FLUXNET file", extra={"size_mb": round(len(content) / 1e6, 1)})
        return content
    except Exception as e:
        log.warning("get_file_stream failed", extra={"error": str(e)})

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


def _find_atmos_meteo_dobj(station_id: str, start_date: str, end_date: str) -> list[dict]:
    """Find ICOS Atmosphere Meteo L2 data objects for a tall-tower station."""
    import requests

    station_uri = STATION_URI_PREFIXES["AS"] + station_id
    query = SPARQL_ATMOS_METEO_QUERY.format(
        station_uri=station_uri, start_date=start_date, end_date=end_date,
    )
    log.info("Querying ICOS SPARQL for Atmosphere Meteo", extra={"station": station_id})
    try:
        resp = requests.post(
            SPARQL_ENDPOINT, data={"query": query},
            headers={"Accept": "application/json"}, timeout=30,
        )
        resp.raise_for_status()
        bindings = resp.json().get("results", {}).get("bindings", [])
        out = [
            {
                "dobj": r["dobj"]["value"],
                "specLabel": r["specLabel"]["value"],
                "fileName": r["fileName"]["value"],
                "timeStart": r["timeStart"]["value"],
                "timeEnd": r["timeEnd"]["value"],
                "size": int(r["size"]["value"]),
            }
            for r in bindings
        ]
        log.info("Found Atmos Meteo objects", extra={"station": station_id, "n": len(out)})
        for r in out[:5]:
            log.info("  dobj", extra={"spec": r["specLabel"], "file": r["fileName"],
                                      "size_mb": round(r["size"]/1e6, 2)})
        return out
    except Exception as e:
        log.warning("Atmos SPARQL failed", extra={"error": str(e)})
        return []


def _download_atmos_file(dobj_url: str) -> bytes | None:
    """Download an ICOS Atmosphere Meteo file (ZIP or CSV).

    Uses get_file_stream (raw file download), NOT get_csv_byte_stream which
    only works for ICOS binary-table objects (not ATC meteo ZIPs/CSVs).
    """
    try:
        client = _icos_data_client()
        result = client.get_file_stream(dobj_url)
        stream = result[1] if isinstance(result, tuple) else result
        content = stream.read()
        log.info("Downloaded ATC meteo file", extra={"size_kb": round(len(content)/1e3, 1)})
        return content
    except Exception as e1:
        log.warning("get_file_stream failed, trying direct HTTP", extra={"error": str(e1)})

    # Fallback: direct HTTP with auth cookie
    import json, requests
    from pathlib import Path as _P
    try:
        conf = json.loads((_P.home() / ".icoscp" / "cpauthToken_auth_conf.json").read_text())
        token = conf.get("token", "")
        # The download URL uses /objects/ endpoint
        dl_url = dobj_url.replace("meta.icos-cp.eu", "data.icos-cp.eu")
        if "/objects/" not in dl_url:
            dl_url = dl_url.replace("/csv/", "/objects/")
        resp = requests.get(
            dl_url, timeout=120,
            cookies={"cpauthToken": token},
        )
        if resp.status_code == 200 and len(resp.content) > 100:
            log.info("Downloaded via direct HTTP", extra={"size_kb": round(len(resp.content)/1e3, 1)})
            return resp.content
        log.warning("Direct HTTP returned", extra={"status": resp.status_code, "size": len(resp.content)})
    except Exception as e2:
        log.warning("Direct HTTP download failed", extra={"error": str(e2)})
    return None


def _parse_atmos_meteo_csv(content: bytes) -> pd.DataFrame | None:
    """Parse an ICOS ATC Meteo MTO file into a harmonized hourly frame.

    The ATC MTO format is a semicolon-separated file with header lines
    starting with '#'. Columns:
      Site;SamplingHeight;Year;Month;Day;Hour;Minute;DecimalDate;
      AT;AT-Stdev;AT-NbPoints;AT-Flag;...;
      RH;RH-Stdev;...;WS;WS-Stdev;...;WD;WD-Stdev;...

    Each file covers ONE sampling height (e.g. 10 m, 120 m). The height is
    in the 'SamplingHeight' column. We rename the variables to include the
    height: T_10m, ws_120m, etc.
    """
    from io import StringIO
    try:
        txt = content.decode("utf-8", errors="replace")
    except Exception:
        return None

    # The MTO format has comment lines starting with '#'. The LAST comment line
    # (starting with '#Site;' or '#  Site;') contains the column header.
    all_lines = txt.splitlines()
    header_line = None
    data_start = 0
    for i, ln in enumerate(all_lines):
        if ln.startswith("#"):
            # Check if this looks like the header (contains 'Site;' or 'Year')
            stripped = ln.lstrip("# ")
            if "Site" in stripped and "Year" in stripped:
                header_line = stripped
            data_start = i + 1
        else:
            break

    data_lines = all_lines[data_start:]
    if not data_lines:
        return None
    if header_line is None:
        log.warning("ATC MTO: no header line found")
        return None

    try:
        df = pd.read_csv(StringIO(header_line + "\n" + "\n".join(data_lines)), sep=";")
    except Exception as e:
        log.warning("ATC MTO parse failed", extra={"error": str(e)})
        return None

    if df.empty:
        return None

    # Build timestamp from Year/Month/Day/Hour/Minute columns
    time_cols = {"Year", "Month", "Day", "Hour", "Minute"}
    if not time_cols.issubset(df.columns):
        log.warning("ATC MTO: missing time columns", extra={"cols": list(df.columns)[:15]})
        return None

    df["time"] = pd.to_datetime(
        df[["Year", "Month", "Day", "Hour", "Minute"]].rename(
            columns={"Year": "year", "Month": "month", "Day": "day",
                      "Hour": "hour", "Minute": "minute"}
        ),
        utc=True, errors="coerce",
    )
    df = df.dropna(subset=["time"]).set_index("time").sort_index()

    # Get the sampling height
    h = int(float(df["SamplingHeight"].iloc[0]))

    def _clean(s):
        s = pd.to_numeric(s, errors="coerce")
        return s.where(s > -990).astype("float32")

    result = pd.DataFrame(index=df.index)
    for src, tgt in [("AT", "T"), ("RH", "RH"), ("WS", "ws"), ("WD", "wd")]:
        if src in df.columns:
            result[f"{tgt}_{h}m"] = _clean(df[src])

    # Also set primary aliases (ws, T, etc.) — will be overridden if
    # multiple heights are merged later, keeping the highest ws and lowest T.
    if "WS" in df.columns:
        result["ws"] = _clean(df["WS"])
    if "WD" in df.columns:
        result["wd"] = _clean(df["WD"])
    if "AT" in df.columns:
        result["T"] = _clean(df["AT"])
    if "RH" in df.columns:
        result["RH"] = _clean(df["RH"])
    if "AP" in df.columns:
        result["p"] = _clean(df["AP"])
    if "PA" in df.columns and "p" not in result.columns:
        result["p"] = _clean(df["PA"]) * 10.0  # kPa → hPa

    # Add height metadata for downstream merging
    result.attrs["sampling_height"] = h

    result = result.resample("1h").mean().reset_index()
    return result


def _fetch_atmos_station(station_id: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """Fetch ICOS Atmosphere tall-tower meteo data."""
    t_start = pd.Timestamp(start_date, tz="UTC")
    t_end = pd.Timestamp(end_date, tz="UTC")
    results = _find_atmos_meteo_dobj(station_id, start_date, end_date)
    if not results:
        return None
    # Prefer releases over NRT
    results.sort(key=lambda r: ("release" not in r["specLabel"].lower(), r["fileName"]))
    frames: list[pd.DataFrame] = []
    for r in results:
        url = r["dobj"]  # keep meta URL — client handles rewrite
        content = _download_atmos_file(url)
        if content is None:
            continue
        # Handle ZIP files (ATC meteo are packaged as ZIP with .MTO inside)
        if r["fileName"].endswith(".zip") or (content[:2] == b"PK"):
            import zipfile
            from io import BytesIO
            try:
                with zipfile.ZipFile(BytesIO(content)) as zf:
                    # Accept .csv or .MTO files
                    data_names = [n for n in zf.namelist()
                                  if n.endswith(".csv") or n.endswith(".MTO")]
                    if not data_names:
                        log.warning("ZIP has no data file", extra={"file": r["fileName"],
                                                                    "contents": zf.namelist()})
                        continue
                    with zf.open(data_names[0]) as cf:
                        content = cf.read()
            except Exception as e:
                log.warning("ZIP extraction failed", extra={"file": r["fileName"], "error": str(e)})
                continue
        df = _parse_atmos_meteo_csv(content)
        if df is None or df.empty:
            continue
        df = df.set_index("time")[t_start:t_end].reset_index()
        if not df.empty:
            frames.append(df)
            log.info("ATC data chunk", extra={
                "file": r["fileName"], "rows": len(df),
                "start": str(df["time"].iloc[0]), "end": str(df["time"].iloc[-1]),
            })
    if not frames:
        return None
    # Merge multi-level frames on time. Each frame has height-specific columns
    # (ws_10m, ws_120m, T_10m, ...) plus generic aliases (ws, T, ...).
    # We merge on time, keeping all height-specific columns and resolving
    # conflicts in generic aliases: ws = highest level, T = lowest level.
    merged = frames[0].set_index("time")
    for df in frames[1:]:
        df2 = df.set_index("time")
        # Identify height-specific columns (like ws_120m) vs generic (ws)
        new_cols = [c for c in df2.columns if c not in merged.columns]
        overlap = [c for c in df2.columns if c in merged.columns]
        if new_cols:
            merged = merged.join(df2[new_cols], how="outer")
        # For overlapping generic cols (ws, T, etc.), keep the value from
        # the frame with more data or higher/lower height as appropriate
        for c in overlap:
            merged[c] = merged[c].combine_first(df2[c])

    # Re-assign generic aliases based on height columns present
    h_cols: dict[str, list[tuple[int, str]]] = {}
    import re
    for c in merged.columns:
        m = re.match(r"^(ws|wd|T|RH)_(\d+)m$", c)
        if m:
            var, h = m.group(1), int(m.group(2))
            h_cols.setdefault(var, []).append((h, c))
    if "ws" in h_cols:
        h_top = max(h_cols["ws"], key=lambda x: x[0])
        merged["ws"] = merged[h_top[1]]
    if "wd" in h_cols:
        h_top = max(h_cols["wd"], key=lambda x: x[0])
        merged["wd"] = merged[h_top[1]]
    if "T" in h_cols:
        h_low = min(h_cols["T"], key=lambda x: x[0])
        merged["T"] = merged[h_low[1]]
    if "RH" in h_cols:
        h_low = min(h_cols["RH"], key=lambda x: x[0])
        merged["RH"] = merged[h_low[1]]

    heights_found = sorted(set(h for hl in h_cols.values() for h, _ in hl))
    log.info("Multi-level merge done", extra={
        "station": station_id, "heights_m": heights_found,
        "total_rows": len(merged), "columns": list(merged.columns)[:20],
    })
    return merged.reset_index()


def _fetch_station_data(
    station_id: str, start_date: str, end_date: str
) -> pd.DataFrame | None:
    """Fetch meteorological data for an ICOS station.

    Routes to the appropriate ICOS product based on station type:
      - type 'ES' → FLUXNET Product (Ecosystem stations)
      - type 'AS' → ATC Meteo Level 2 (Atmosphere tall-towers)
    """
    meta = ICOS_STATIONS.get(station_id, {})
    if meta.get("type") == "AS":
        return _fetch_atmos_station(station_id, start_date, end_date)

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

    # Meteo group — write all numeric columns (primary aliases + per-level)
    meteo = root.require_group("meteo")
    skip_cols = {"time"}
    for col in df.columns:
        if col in skip_cols:
            continue
        vals = df[col].values
        if vals.dtype.kind not in ("f", "i"):
            continue
        vals = vals.astype(np.float32)
        arr = meteo.create_array(
            col, shape=(n_times,), dtype=np.float32,
            chunks=(min(720, n_times),), overwrite=True,
        )
        arr[:] = vals
        info = VARIABLE_MAP.get(col.split("_")[0])
        if info is not None:
            arr.attrs.update({"long_name": info["long_name"], "units": info["units"]})
        else:
            arr.attrs.update({"long_name": col})

    # Global attributes
    root.attrs.update({
        "Conventions": "CF-1.9",
        "title": f"ICOS atmospheric station — {station_id} ({station_meta['name']})",
        "station_id": station_id,
        "station_type": station_meta.get("type", "ES"),
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
