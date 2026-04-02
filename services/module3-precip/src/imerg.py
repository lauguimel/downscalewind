"""Download IMERG V07 satellite precipitation at station locations via GEE."""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import ee
import pandas as pd

log = logging.getLogger(__name__)

IMERG_COLLECTION = "NASA/GPM_L3/IMERG_V07"
IMERG_BAND = "precipitation"  # mm/hr
IMERG_SCALE = 11132  # ~0.1 deg in meters


def download_imerg_daily(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    gee_project: str,
) -> pd.DataFrame:
    """Download IMERG daily precipitation at a single point.

    Args:
        lat, lon: station coordinates
        start_date, end_date: "YYYY-MM-DD"
        gee_project: GEE cloud project ID

    Returns:
        DataFrame indexed by date with column rain_imerg (mm/day).
    """
    ee.Initialize(project=gee_project)

    point = ee.Geometry.Point([lon, lat])
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    records: list[dict[str, Any]] = []
    current = start
    while current <= end:
        day_start = current.strftime("%Y-%m-%d")
        day_end = (current + timedelta(days=1)).strftime("%Y-%m-%d")

        col = (
            ee.ImageCollection(IMERG_COLLECTION)
            .filterDate(day_start, day_end)
            .select(IMERG_BAND)
        )

        # Sum half-hourly rates: each image is 30-min, rate in mm/hr
        # Daily total = sum(rate_mm_hr * 0.5h)
        daily_img = col.sum().multiply(0.5)
        val = daily_img.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=IMERG_SCALE,
        ).getInfo()

        rain = val.get(IMERG_BAND)
        records.append({"date": current.normalize(), "rain_imerg": rain})
        current += timedelta(days=1)

    df = pd.DataFrame(records).set_index("date")
    return df


def _process_month(
    fc: ee.FeatureCollection,
    year: int,
    month: int,
) -> list[dict[str, Any]]:
    """Process one month: daily sum images + reduceRegions over station points.

    Returns list of dicts with keys: station_id, date, rain_imerg.
    """
    import calendar

    n_days = calendar.monthrange(year, month)[1]
    records: list[dict[str, Any]] = []

    for day in range(1, n_days + 1):
        day_date = date(year, month, day)
        day_start = day_date.isoformat()
        day_end = (day_date + timedelta(days=1)).isoformat()

        col = (
            ee.ImageCollection(IMERG_COLLECTION)
            .filterDate(day_start, day_end)
            .select(IMERG_BAND)
        )

        # Daily total: sum of 48 half-hourly images, each in mm/hr -> * 0.5
        daily_img = col.sum().multiply(0.5)

        reduced = daily_img.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.first(),
            scale=IMERG_SCALE,
        )

        features = reduced.getInfo()["features"]
        for feat in features:
            props = feat["properties"]
            rain = props.get("first")
            records.append({
                "station_id": props["station_id"],
                "date": pd.Timestamp(day_date),
                "rain_imerg": rain if rain is not None else float("nan"),
            })

    return records


def download_imerg_batch(
    stations_df: pd.DataFrame,
    year: int,
    gee_project: str,
    max_workers: int = 1,
) -> pd.DataFrame:
    """Download IMERG daily precipitation for all station locations.

    Uses reduceRegions for efficiency: one GEE call per day covers all stations.
    Processes one month at a time to avoid GEE timeouts.

    Args:
        stations_df: must have columns station_id, lat, lon
        year: target year
        gee_project: GEE cloud project ID
        max_workers: unused (reserved for future parallelism)

    Returns:
        Long-format DataFrame: station_id, date, rain_imerg
    """
    ee.Initialize(project=gee_project)

    # Build FeatureCollection of station points
    unique_stations = stations_df[["station_id", "lat", "lon"]].drop_duplicates("station_id")
    features = []
    for _, row in unique_stations.iterrows():
        feat = ee.Feature(
            ee.Geometry.Point([row["lon"], row["lat"]]),
            {"station_id": row["station_id"]},
        )
        features.append(feat)
    fc = ee.FeatureCollection(features)

    log.info("IMERG batch: %d stations, year %d", len(unique_stations), year)

    all_records: list[dict[str, Any]] = []
    for month in range(1, 13):
        log.info("Processing IMERG %d-%02d...", year, month)
        try:
            monthly = _process_month(fc, year, month)
            all_records.extend(monthly)
        except Exception as e:
            log.error("Failed IMERG %d-%02d: %s", year, month, e)

    if not all_records:
        return pd.DataFrame(columns=["station_id", "date", "rain_imerg"])

    df = pd.DataFrame(all_records)
    n_valid = df["rain_imerg"].notna().sum()
    log.info("IMERG batch: %d total rows, %d valid", len(df), n_valid)
    return df
