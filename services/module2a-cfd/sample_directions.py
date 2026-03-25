"""
sample_directions.py — Select representative ERA5 timestamps per site.

For each site, fetches ERA5 surface wind from Open-Meteo (with file-based
caching), clusters the wind climate via K-means on (speed, direction, shear),
and selects N representative timestamps (medoid per cluster).

Each selected timestamp is a real ERA5 state → CFD uses the full 3D profile.

Usage
-----
    # Single site
    python sample_directions.py \
        --site-lat 39.716 --site-lon -7.74 --site-id site_00000 \
        --n-timestamps 15 --output-dir ../../data/campaign/directions

    # Batch mode (from sites.csv)
    python sample_directions.py \
        --sites-csv ../../data/campaign/sites/sites.csv \
        --n-timestamps 15 --output-dir ../../data/campaign/directions

    # Using local ERA5 Zarr instead of Open-Meteo API
    python sample_directions.py \
        --sites-csv sites.csv --source zarr \
        --era5-zarr ../../data/raw/era5_perdigao.zarr \
        --output-dir ../../data/campaign/directions

Dependencies: numpy, pandas, scikit-learn, matplotlib, requests
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

OPEN_METEO_ERA5_URL = "https://archive-api.open-meteo.com/v1/era5"
HOURLY_VARS = "wind_speed_10m,wind_direction_10m,wind_speed_100m,temperature_2m"


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_era5_openmeteo(
    lat: float, lon: float,
    start_date: str, end_date: str,
    cache_dir: Path,
    delay: float = 0.5,
) -> pd.DataFrame:
    """Fetch ERA5 surface wind from Open-Meteo API, with file-based caching.

    Returns DataFrame with columns:
        timestamp, speed_10m, direction_deg, speed_100m, t2m
    """
    import requests

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"era5_{lat:.3f}_{lon:.3f}.parquet"

    if cache_file.exists():
        logger.debug("Cache hit: %s", cache_file)
        return pd.read_parquet(cache_file)

    # Retry with exponential backoff on 429 Too Many Requests
    max_retries = 5
    for attempt in range(max_retries):
        resp = requests.get(
            OPEN_METEO_ERA5_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": start_date,
                "end_date": end_date,
                "hourly": HOURLY_VARS,
                "timezone": "UTC",
            },
            timeout=120,
        )
        if resp.status_code == 429:
            wait = delay * (2 ** attempt)
            logger.warning("Rate limited, waiting %.0fs (attempt %d/%d)",
                           wait, attempt + 1, max_retries)
            time.sleep(wait)
            continue
        break
    resp.raise_for_status()
    data = resp.json()

    hourly = data["hourly"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        "speed_10m": hourly["wind_speed_10m"],
        "direction_deg": hourly["wind_direction_10m"],
        "speed_100m": hourly["wind_speed_100m"],
        "t2m": hourly["temperature_2m"],
    })
    df = df.dropna(subset=["speed_10m", "direction_deg", "speed_100m"])

    df.to_parquet(cache_file, index=False)
    logger.debug("Cached %d rows → %s", len(df), cache_file)

    time.sleep(delay)  # rate-limit
    return df


def fetch_era5_zarr(
    lat: float, lon: float,
    zarr_path: Path,
) -> pd.DataFrame:
    """Load ERA5 u10/v10 from local Zarr store at nearest grid point.

    Returns DataFrame with same schema as fetch_era5_openmeteo.
    """
    import zarr

    store = zarr.open_group(str(zarr_path), mode="r")

    times_ns = np.array(store["coords/time"][:])
    lats = np.array(store["coords/lat"][:])
    lons = np.array(store["coords/lon"][:])

    # Find nearest grid point
    ilat = int(np.argmin(np.abs(lats - lat)))
    ilon = int(np.argmin(np.abs(lons - lon)))

    u10 = np.array(store["surface/u10"][:, ilat, ilon], dtype=np.float64)
    v10 = np.array(store["surface/v10"][:, ilat, ilon], dtype=np.float64)

    speed_10m = np.sqrt(u10**2 + v10**2)
    # Meteorological convention: direction wind comes FROM
    direction_deg = (270.0 - np.degrees(np.arctan2(v10, u10))) % 360.0

    # Approximate speed_100m from pressure levels if available
    try:
        levels = np.array(store["coords/level"][:])
        z_geopot = np.array(store["pressure/z"][:, :, ilat, ilon])
        u_pres = np.array(store["pressure/u"][:, :, ilat, ilon])
        v_pres = np.array(store["pressure/v"][:, :, ilat, ilon])
        z_m = z_geopot / 9.81  # geopotential → metres

        # Interpolate to 100m AGL (approximate)
        speed_100m = np.full_like(speed_10m, np.nan)
        for t in range(len(speed_10m)):
            z_agl = z_m[t] - z_m[t, -1]  # approx AGL (lowest level ≈ surface)
            u_interp = np.interp(100.0, z_agl[::-1], u_pres[t, ::-1])
            v_interp = np.interp(100.0, z_agl[::-1], v_pres[t, ::-1])
            speed_100m[t] = np.sqrt(u_interp**2 + v_interp**2)
    except (KeyError, IndexError):
        # Fallback: estimate with power law alpha=0.14
        speed_100m = speed_10m * (100.0 / 10.0) ** 0.14

    datetimes = times_ns.astype("datetime64[ns]")

    df = pd.DataFrame({
        "timestamp": datetimes,
        "speed_10m": speed_10m,
        "direction_deg": direction_deg,
        "speed_100m": speed_100m,
        "t2m": np.full_like(speed_10m, np.nan),
    })
    valid = np.isfinite(speed_10m) & np.isfinite(direction_deg)
    return df[valid].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Timestamp selection
# ---------------------------------------------------------------------------

def select_timestamps(
    df: pd.DataFrame,
    n_timestamps: int = 15,
    min_speed: float = 1.0,
) -> pd.DataFrame:
    """Select representative timestamps via K-means clustering.

    Features: speed_10m, dir_sin, dir_cos, shear_exponent.
    Returns subset of df (n_timestamps rows) with added cluster_id column.
    """
    df_valid = df[df["speed_10m"] >= min_speed].copy()

    if len(df_valid) < n_timestamps:
        logger.warning(
            "Only %d valid timestamps (need %d), using all",
            len(df_valid), n_timestamps,
        )
        df_valid["cluster_id"] = range(len(df_valid))
        return df_valid

    # Features
    dir_rad = np.radians(df_valid["direction_deg"].values)
    df_valid["dir_sin"] = np.sin(dir_rad)
    df_valid["dir_cos"] = np.cos(dir_rad)

    speed_ratio = df_valid["speed_100m"].values / np.maximum(
        df_valid["speed_10m"].values, 0.1
    )
    df_valid["shear"] = np.log(np.clip(speed_ratio, 0.1, 10.0)) / np.log(10.0)

    feature_cols = ["speed_10m", "dir_sin", "dir_cos", "shear"]
    X = df_valid[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_clusters = min(n_timestamps, len(df_valid))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    df_valid["cluster_id"] = labels

    # Select medoid per cluster
    selected_indices = []
    for cid in range(n_clusters):
        mask = labels == cid
        cluster_scaled = X_scaled[mask]
        centre = kmeans.cluster_centers_[cid]
        dists = np.linalg.norm(cluster_scaled - centre, axis=1)
        medoid_local = int(np.argmin(dists))
        cluster_idx = df_valid.index[mask]
        selected_indices.append(cluster_idx[medoid_local])

    selected = df_valid.loc[selected_indices].sort_values("timestamp")
    return selected


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_wind_rose(
    df_full: pd.DataFrame,
    df_selected: pd.DataFrame,
    output_path: Path,
    site_id: str = "",
) -> None:
    """Plot wind rose with selected timestamps highlighted."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             subplot_kw={"projection": "polar"})

    # Left: full distribution
    ax = axes[0]
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title("ERA5 wind distribution", pad=15)
    theta = np.radians(df_full["direction_deg"])
    speed = df_full["speed_10m"]
    ax.scatter(theta, speed, s=1, alpha=0.05, c="steelblue")
    ax.set_ylim(0, min(30, speed.quantile(0.99)))
    ax.set_ylabel("Speed [m/s]", labelpad=30)

    # Right: selected timestamps
    ax = axes[1]
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(f"Selected timestamps (n={len(df_selected)})", pad=15)
    theta_sel = np.radians(df_selected["direction_deg"])
    sc = ax.scatter(
        theta_sel, df_selected["speed_10m"],
        c=df_selected["cluster_id"], cmap="tab20",
        s=60, edgecolors="k", linewidths=0.5, zorder=5,
    )
    ax.set_ylim(0, min(30, df_selected["speed_10m"].max() * 1.3))
    ax.set_ylabel("Speed [m/s]", labelpad=30)
    plt.colorbar(sc, ax=ax, label="Cluster", pad=0.1)

    fig.suptitle(
        f"Wind sampling — {site_id}" if site_id else "Wind sampling",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Wind rose → %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_site(
    site_id: str,
    lat: float,
    lon: float,
    n_timestamps: int,
    min_speed: float,
    source: str,
    era5_zarr: Path | None,
    cache_dir: Path,
    start_date: str,
    end_date: str,
    figures_dir: Path | None,
    delay: float = 1.5,
) -> pd.DataFrame | None:
    """Process one site: fetch ERA5, cluster, select timestamps, plot."""
    try:
        if source == "zarr" and era5_zarr is not None:
            df_wind = fetch_era5_zarr(lat, lon, era5_zarr)
        else:
            df_wind = fetch_era5_openmeteo(
                lat, lon, start_date, end_date, cache_dir, delay=delay
            )
    except Exception as e:
        logger.error("Failed to fetch ERA5 for %s: %s", site_id, e)
        return None

    selected = select_timestamps(df_wind, n_timestamps, min_speed)

    # Add site_id
    selected = selected.copy()
    selected.insert(0, "site_id", site_id)

    # Plot
    if figures_dir is not None:
        figures_dir.mkdir(parents=True, exist_ok=True)
        plot_wind_rose(
            df_wind, selected,
            output_path=figures_dir / f"wind_rose_{site_id}.png",
            site_id=site_id,
        )

    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Select representative ERA5 timestamps per site"
    )
    # Single-site mode
    parser.add_argument("--site-lat", type=float, help="Site latitude")
    parser.add_argument("--site-lon", type=float, help="Site longitude")
    parser.add_argument("--site-id", default="site_00000")

    # Batch mode
    parser.add_argument("--sites-csv", type=Path, default=None,
                        help="CSV with columns: site_id, lat, lon")

    # ERA5 source
    parser.add_argument("--source", choices=["api", "zarr"], default="api",
                        help="ERA5 data source: 'api' (Open-Meteo) or 'zarr' (local)")
    parser.add_argument("--era5-zarr", type=Path, default=None,
                        help="Path to ERA5 Zarr store (source=zarr only)")

    # Sampling parameters
    parser.add_argument("--n-timestamps", type=int, default=15)
    parser.add_argument("--min-speed", type=float, default=1.0,
                        help="Min 10m speed filter (m/s)")
    parser.add_argument("--start-date", default="2016-01-01")
    parser.add_argument("--end-date", default="2020-12-31")

    # Output
    parser.add_argument("--output-dir", type=Path,
                        default=Path("../../data/campaign/directions"))
    parser.add_argument("--cache-dir", type=Path,
                        default=Path(".cache/open_meteo_era5"))
    parser.add_argument("--max-sites", type=int, default=None,
                        help="Process only first N sites (testing)")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--delay", type=float, default=1.5,
                        help="Delay between API requests in seconds (default: 1.5)")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = None if args.no_figures else args.output_dir / "figures"

    if args.sites_csv is not None:
        # --- Batch mode ---
        sites = pd.read_csv(args.sites_csv)

        # Handle both column naming conventions
        if "lon_center" in sites.columns:
            sites = sites.rename(columns={
                "lon_center": "lon", "lat_center": "lat"
            })

        if args.max_sites is not None:
            sites = sites.head(args.max_sites)

        logger.info(
            "Batch mode: %d sites, %d timestamps each, source=%s",
            len(sites), args.n_timestamps, args.source,
        )

        all_selected = []
        for i, row in sites.iterrows():
            sid = row["site_id"]
            lat, lon = row["lat"], row["lon"]
            cache_file = args.cache_dir / f"era5_{lat:.3f}_{lon:.3f}.parquet"
            cached = "[cached]" if cache_file.exists() else ""

            logger.info(
                "  [%d/%d] %s (%.3f, %.3f) %s",
                i + 1, len(sites), sid, lat, lon, cached,
            )

            result = process_site(
                sid, lat, lon,
                args.n_timestamps, args.min_speed,
                args.source, args.era5_zarr,
                args.cache_dir, args.start_date, args.end_date,
                figures_dir, delay=args.delay,
            )
            if result is not None:
                all_selected.append(result)

        if not all_selected:
            logger.error("No timestamps selected for any site.")
            return

        combined = pd.concat(all_selected, ignore_index=True)

        # Select output columns
        out_cols = [
            "site_id", "timestamp", "speed_10m", "direction_deg",
            "speed_100m", "shear", "dir_sin", "dir_cos", "cluster_id",
        ]
        out_cols = [c for c in out_cols if c in combined.columns]
        combined = combined[out_cols]

        csv_path = args.output_dir / "all_directions.csv"
        combined.to_csv(csv_path, index=False)

        n_sites_ok = combined["site_id"].nunique()
        logger.info(
            "Selected %d timestamps for %d sites → %s",
            len(combined), n_sites_ok, csv_path,
        )
        print(f"\n{'='*60}")
        print(f"  {n_sites_ok} sites, {len(combined)} total timestamps")
        print(f"  Mean per site: {len(combined)/n_sites_ok:.1f}")
        print(f"  Speed range: {combined['speed_10m'].min():.1f} – "
              f"{combined['speed_10m'].max():.1f} m/s")
        print(f"{'='*60}")

    elif args.site_lat is not None and args.site_lon is not None:
        # --- Single-site mode ---
        result = process_site(
            args.site_id, args.site_lat, args.site_lon,
            args.n_timestamps, args.min_speed,
            args.source, args.era5_zarr,
            args.cache_dir, args.start_date, args.end_date,
            figures_dir, delay=args.delay,
        )
        if result is not None:
            csv_path = args.output_dir / f"directions_{args.site_id}.csv"
            result.to_csv(csv_path, index=False)
            print(f"\nSelected {len(result)} timestamps for {args.site_id}:")
            print(result[["timestamp", "speed_10m", "direction_deg", "shear"]].to_string(index=False))
    else:
        parser.error(
            "Provide --sites-csv (batch) or --site-lat + --site-lon (single)"
        )


if __name__ == "__main__":
    main()
