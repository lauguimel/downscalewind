"""
Download ERA5-Land hourly data from Google Earth Engine and compute FWI baseline.

Targets:
  - Perdigao IOP period (2017-05-01 to 2017-06-30)
  - Two points: Perdigao (39.716, -7.740) and Pedrogao Grande (39.93, -8.23)

Uses noon (11-13 UTC mean) observations as standard for FWI computation.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import ee
import numpy as np
import pandas as pd

# Add project root to path for shared imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from shared.fwi import compute_fwi_series  # noqa: E402


# ── Configuration ──────────────────────────────────────────────────────────────

GEE_PROJECT = "ee-guillaumemaitrejean"
ERA5L_COLLECTION = "ECMWF/ERA5_LAND/HOURLY"

START_DATE = "2017-05-01"
END_DATE = "2017-07-01"  # exclusive

POINTS = {
    "perdigao": {"lat": 39.716, "lon": -7.740},
    "pedrogao_grande": {"lat": 39.93, "lon": -8.23},
}

# ERA5-Land band names
BANDS_NOON = [
    "temperature_2m",
    "dewpoint_temperature_2m",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
]
BAND_PRECIP = "total_precipitation_hourly"


def dewpoint_to_rh(t_c: np.ndarray, td_c: np.ndarray) -> np.ndarray:
    """Convert temperature and dewpoint (both in C) to RH (%)."""
    rh = 100.0 * np.exp(17.625 * td_c / (243.04 + td_c)) / np.exp(
        17.625 * t_c / (243.04 + t_c)
    )
    return np.clip(rh, 0.0, 100.0)


def extract_daily_noon(
    collection: ee.ImageCollection,
    point: ee.Geometry.Point,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Extract daily noon (11-13 UTC mean) values at a point from ERA5-Land."""
    dates = pd.date_range(start, end, freq="D", inclusive="left")
    records = []

    for d in dates:
        day_str = d.strftime("%Y-%m-%d")
        # Filter to 11:00 and 12:00 UTC (noon window)
        t0 = f"{day_str}T11:00:00"
        t1 = f"{day_str}T13:00:00"

        noon_col = collection.filterDate(t0, t1).select(BANDS_NOON)
        noon_mean = noon_col.mean()

        # Daily total precipitation: sum of hourly precip over 24h
        day_start = f"{day_str}T00:00:00"
        day_end_str = (d + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00")
        daily_precip = (
            collection.filterDate(day_start, day_end_str)
            .select([BAND_PRECIP])
            .reduce(ee.Reducer.sum())
        )

        # Combine noon weather + daily precip
        combined = noon_mean.addBands(
            daily_precip.rename("total_precip_daily")
        )

        vals = combined.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=11132,  # ~0.1 deg, ERA5-Land native
        )

        records.append({"date": day_str, "values": vals})

    # Batch getInfo to reduce API calls: collect all computations as a list
    print(f"  Fetching {len(records)} days from GEE...")
    rows = []
    for i, rec in enumerate(records):
        try:
            info = rec["values"].getInfo()
        except Exception as e:
            print(f"  Warning: failed for {rec['date']}: {e}")
            info = {}
        rows.append({"date": rec["date"], **info})

        # Small delay every 10 requests to avoid rate limiting
        if (i + 1) % 10 == 0:
            time.sleep(0.5)
            print(f"    ... {i + 1}/{len(records)} days done")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def process_point(name: str, lat: float, lon: float, collection: ee.ImageCollection) -> pd.DataFrame:
    """Extract ERA5-Land data at a point and compute FWI series."""
    print(f"\nProcessing {name} (lat={lat}, lon={lon})...")
    point = ee.Geometry.Point(lon, lat)

    df = extract_daily_noon(collection, point, START_DATE, END_DATE)

    # Unit conversions
    # Temperature: K -> C
    df["T_C"] = df["temperature_2m"] - 273.15
    # Dewpoint: K -> C
    td_c = df["dewpoint_temperature_2m"] - 273.15
    # RH from dewpoint
    df["RH"] = dewpoint_to_rh(df["T_C"].values, td_c.values)
    # Wind speed: m/s -> km/h
    u10 = df["u_component_of_wind_10m"].values
    v10 = df["v_component_of_wind_10m"].values
    df["ws_kmh"] = np.sqrt(u10**2 + v10**2) * 3.6
    # Precipitation: m -> mm (total_precipitation_hourly summed over 24h, in meters)
    df["rain_mm"] = df["total_precip_daily"].fillna(0.0) * 1000.0
    # Clamp negative precip (can happen with rounding)
    df["rain_mm"] = df["rain_mm"].clip(lower=0.0)

    # Month array for FWI
    months = df["date"].dt.month.values

    # Compute FWI
    fwi_out = compute_fwi_series(
        t_c=df["T_C"].values,
        rh=df["RH"].values,
        ws_kmh=df["ws_kmh"].values,
        rain_mm=df["rain_mm"].values,
        months=months,
    )

    # Build output
    result = pd.DataFrame({
        "date": df["date"].dt.strftime("%Y-%m-%d"),
        "T_C": df["T_C"].round(2),
        "RH": df["RH"].round(1),
        "ws_kmh": df["ws_kmh"].round(2),
        "rain_mm": df["rain_mm"].round(2),
        "ffmc": np.round(fwi_out["ffmc"], 2),
        "dmc": np.round(fwi_out["dmc"], 2),
        "dc": np.round(fwi_out["dc"], 2),
        "isi": np.round(fwi_out["isi"], 2),
        "bui": np.round(fwi_out["bui"], 2),
        "fwi": np.round(fwi_out["fwi"], 2),
    })

    return result


def main():
    print("Initializing Google Earth Engine...")
    ee.Initialize(project=GEE_PROJECT)

    collection = ee.ImageCollection(ERA5L_COLLECTION).filterDate(START_DATE, END_DATE)

    output_dir = PROJECT_ROOT / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, coords in POINTS.items():
        df = process_point(name, coords["lat"], coords["lon"], collection)

        out_path = output_dir / f"era5land_fwi_{name}.csv"
        df.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")

        # Print June 17 values
        row = df[df["date"] == "2017-06-17"]
        if not row.empty:
            print(f"\n  === {name.upper()} — 2017-06-17 (Pedrogao Grande fire) ===")
            for col in df.columns:
                print(f"  {col}: {row[col].values[0]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
