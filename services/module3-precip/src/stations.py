"""Ground truth precipitation from GHCN-D and Meteo-France SYNOP."""
from __future__ import annotations

import gzip
import io
import logging
import time
from typing import Any

import pandas as pd
import requests

log = logging.getLogger(__name__)

# --- Meteo-France SYNOP hardcoded station coordinates ---
# WMO ID -> (lat, lon, name)
MF_STATION_COORDS: dict[int, tuple[float, float, str]] = {
    7643: (43.58, 3.96, "MONTPELLIER"),
    7560: (44.12, 3.58, "MONT_AIGOUAL"),
    7558: (44.12, 3.02, "MILLAU"),
    7630: (43.91, 4.90, "AVIGNON"),
    7645: (43.87, 4.40, "NIMES"),
    7646: (43.76, 4.42, "NIMES_GARONS"),
    7647: (43.52, 4.92, "ISTRES"),
    7535: (44.83, 2.42, "AURILLAC"),
    7510: (45.73, 3.15, "CLERMONT_FERRAND"),
    7434: (46.17, -1.15, "LA_ROCHELLE"),
    7481: (44.83, -0.69, "BORDEAUX"),
    7747: (42.74, 2.87, "PERPIGNAN"),
    7190: (48.07, -1.73, "RENNES"),
    7255: (47.27, 5.08, "DIJON"),
    7149: (48.77, 2.00, "TRAPPES"),
}

GHCND_INVENTORY_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt"
GHCND_STATION_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/{station_id}.csv.gz"
MF_SYNOP_URL = "https://donneespubliques.meteofrance.fr/donnees_libres/Txt/Synop/Archive/synop.{yyyymm}.csv.gz"


# ---- GHCN-D ----

def download_ghcnd_inventory(
    bbox: tuple[float, float, float, float],
) -> pd.DataFrame:
    """Download GHCN-D inventory and filter by bbox and PRCP element.

    Args:
        bbox: (lon_min, lat_min, lon_max, lat_max)

    Returns:
        DataFrame with columns: id, lat, lon, first_year, last_year
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    log.info("Downloading GHCN-D inventory...")
    resp = requests.get(GHCND_INVENTORY_URL, timeout=60)
    resp.raise_for_status()

    rows = []
    for line in resp.text.splitlines():
        if not line.strip():
            continue
        # Fixed-width format: ID(0:11) LAT(12:20) LON(21:30) ELEM(31:35) FIRST(36:40) LAST(41:45)
        sid = line[0:11].strip()
        lat = float(line[12:20])
        lon = float(line[21:30])
        elem = line[31:35].strip()
        first_year = int(line[36:40])
        last_year = int(line[41:45])
        if elem != "PRCP":
            continue
        if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
            continue
        rows.append(
            {"id": sid, "lat": lat, "lon": lon, "first_year": first_year, "last_year": last_year}
        )

    df = pd.DataFrame(rows)
    log.info("Found %d GHCN-D PRCP stations in bbox", len(df))
    return df


def download_ghcnd_station(
    station_id: str,
    year: int,
    min_valid_days: int = 300,
) -> pd.DataFrame | None:
    """Download PRCP for a single GHCN-D station.

    Returns DataFrame indexed by date with column rain_mm, or None if
    fewer than min_valid_days valid observations.
    """
    url = GHCND_STATION_URL.format(station_id=station_id)
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.warning("Failed to download %s: %s", station_id, e)
        return None

    buf = io.BytesIO(resp.content)
    try:
        df = pd.read_csv(
            buf,
            compression="gzip",
            header=None,
            names=["id", "date", "element", "value", "mflag", "qflag", "sflag", "obs_time"],
            dtype={"id": str, "date": str, "element": str},
            low_memory=False,
        )
    except Exception as e:
        log.warning("Failed to parse %s: %s", station_id, e)
        return None

    # Filter PRCP element
    df = df[df["element"] == "PRCP"].copy()
    if df.empty:
        return None

    # Parse dates and filter year
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df = df[df["date"].dt.year == year]

    # Exclude quality-flagged observations (qflag not empty/NaN)
    df = df[df["qflag"].isna() | (df["qflag"] == " ") | (df["qflag"] == "")]

    if len(df) < min_valid_days:
        return None

    # Convert tenths of mm to mm
    result = pd.DataFrame({"rain_mm": df["value"].values / 10.0}, index=df["date"].values)
    result.index.name = "date"
    return result


def download_ghcnd_batch(
    stations_df: pd.DataFrame,
    year: int,
    max_stations: int | None = None,
) -> pd.DataFrame:
    """Download PRCP for multiple GHCN-D stations.

    Returns long-format DataFrame: station_id, date, lat, lon, rain_mm.
    """
    sdf = stations_df.head(max_stations) if max_stations else stations_df
    frames: list[pd.DataFrame] = []

    for i, row in enumerate(sdf.itertuples()):
        if (i + 1) % 50 == 0:
            log.info("GHCN-D download progress: %d / %d", i + 1, len(sdf))

        data = download_ghcnd_station(row.id, year)
        if data is None:
            continue

        chunk = data.reset_index()
        chunk["station_id"] = row.id
        chunk["lat"] = row.lat
        chunk["lon"] = row.lon
        frames.append(chunk)

        time.sleep(0.05)

    if not frames:
        log.warning("No valid GHCN-D stations found for year %d", year)
        return pd.DataFrame(columns=["station_id", "date", "lat", "lon", "rain_mm"])

    result = pd.concat(frames, ignore_index=True)
    result = result[["station_id", "date", "lat", "lon", "rain_mm"]]
    log.info("Downloaded %d station-days from %d stations", len(result), len(frames))
    return result


# ---- Meteo-France SYNOP ----

def download_mf_synop(year: int, bbox: tuple[float, float, float, float]) -> pd.DataFrame:
    """Download Meteo-France SYNOP hourly data and extract daily rr24.

    Args:
        year: target year
        bbox: (lon_min, lat_min, lon_max, lat_max)

    Returns:
        Long-format DataFrame: station_id, date, lat, lon, rain_mm
    """
    lon_min, lat_min, lon_max, lat_max = bbox

    # Filter stations by bbox
    stations_in_bbox = {
        wmo_id: (lat, lon, name)
        for wmo_id, (lat, lon, name) in MF_STATION_COORDS.items()
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max
    }
    if not stations_in_bbox:
        log.info("No MF SYNOP stations in bbox")
        return pd.DataFrame(columns=["station_id", "date", "lat", "lon", "rain_mm"])

    log.info("Downloading MF SYNOP for %d stations, year %d", len(stations_in_bbox), year)

    frames: list[pd.DataFrame] = []
    for month in range(1, 13):
        yyyymm = f"{year}{month:02d}"
        url = MF_SYNOP_URL.format(yyyymm=yyyymm)
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
        except requests.RequestException as e:
            log.warning("Failed to download SYNOP %s: %s", yyyymm, e)
            continue

        buf = io.BytesIO(resp.content)
        try:
            raw = pd.read_csv(buf, compression="gzip", sep=";", low_memory=False)
        except Exception as e:
            log.warning("Failed to parse SYNOP %s: %s", yyyymm, e)
            continue

        # Columns: numer_sta, date, rr24, ...
        if "numer_sta" not in raw.columns or "rr24" not in raw.columns:
            log.warning("Unexpected columns in SYNOP %s: %s", yyyymm, list(raw.columns)[:10])
            continue

        raw["numer_sta"] = pd.to_numeric(raw["numer_sta"], errors="coerce")
        raw = raw[raw["numer_sta"].isin(stations_in_bbox.keys())]
        if raw.empty:
            continue

        raw["rr24"] = pd.to_numeric(raw["rr24"], errors="coerce")
        raw["date"] = pd.to_datetime(raw["date"], format="%Y%m%d%H%M%S", errors="coerce")

        # Keep only rows with valid rr24
        valid = raw[raw["rr24"].notna()].copy()
        if valid.empty:
            continue

        # Take the daily value: one rr24 per station per day (typically at 06 UTC)
        valid["day"] = valid["date"].dt.date
        daily = valid.groupby(["numer_sta", "day"])["rr24"].first().reset_index()
        daily.columns = ["wmo_id", "date", "rain_mm"]
        daily["date"] = pd.to_datetime(daily["date"])
        daily["wmo_id"] = daily["wmo_id"].astype(int)
        daily["station_id"] = daily["wmo_id"].apply(lambda x: f"MF_{x}")
        daily["lat"] = daily["wmo_id"].map(lambda x: stations_in_bbox[x][0])
        daily["lon"] = daily["wmo_id"].map(lambda x: stations_in_bbox[x][1])
        frames.append(daily[["station_id", "date", "lat", "lon", "rain_mm"]])

    if not frames:
        return pd.DataFrame(columns=["station_id", "date", "lat", "lon", "rain_mm"])

    result = pd.concat(frames, ignore_index=True)
    log.info("MF SYNOP: %d station-days from %d stations", len(result), result["station_id"].nunique())
    return result


# ---- Orchestration ----

def load_all_stations(
    year: int,
    bbox: list | tuple,
    sources: dict | None = None,
    max_stations: int | None = None,
    min_days: int = 300,
) -> pd.DataFrame:
    """Download and merge GHCN-D + MF SYNOP precipitation."""
    bbox = tuple(bbox)
    sources = sources or {"ghcnd": True, "mf_synop": True}
    max_ghcnd = max_stations

    # GHCN-D
    inventory = download_ghcnd_inventory(bbox)
    inventory = inventory[inventory["last_year"] >= year]
    ghcnd = download_ghcnd_batch(inventory, year, max_stations=max_ghcnd)
    log.info("GHCN-D: %d rows", len(ghcnd))

    # MF SYNOP (optional)
    include_mf = sources.get("mf_synop", True)
    if include_mf:
        mf = download_mf_synop(year, bbox)
        log.info("MF SYNOP: %d rows", len(mf))
        combined = pd.concat([ghcnd, mf], ignore_index=True)
    else:
        combined = ghcnd

    # Deduplicate: drop MF stations that overlap with GHCN-D (within 0.01 deg)
    if include_mf and len(combined) > 0:
        ghcnd_locs = combined[~combined["station_id"].str.startswith("MF_")][["lat", "lon"]].drop_duplicates()
        mf_mask = combined["station_id"].str.startswith("MF_")
        keep = []
        for _, row in combined[mf_mask].iterrows():
            dists = ((ghcnd_locs["lat"] - row["lat"]).abs() + (ghcnd_locs["lon"] - row["lon"]).abs())
            if dists.min() > 0.01 if len(dists) > 0 else True:
                keep.append(True)
            else:
                keep.append(False)
        if keep:
            n_dupes = sum(1 for k in keep if not k)
            if n_dupes > 0:
                log.info("Removed %d MF stations overlapping with GHCN-D", n_dupes)
            mf_kept = combined[mf_mask][keep]
            combined = pd.concat([combined[~mf_mask], mf_kept], ignore_index=True)

    log.info("Total: %d station-days, %d unique stations", len(combined), combined["station_id"].nunique())
    return combined
