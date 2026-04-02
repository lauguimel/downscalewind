"""Extract terrain features at station locations via Google Earth Engine SRTM."""

from __future__ import annotations

import logging
from pathlib import Path

import ee
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Cache file for terrain features (terrain doesn't change)
_CACHE_DIR = Path("data/cache")


def extract_terrain_batch(
    stations_df: pd.DataFrame,
    gee_project: str,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Extract terrain features for unique station locations via GEE SRTM.

    Parameters
    ----------
    stations_df : pd.DataFrame
        Must contain columns: station_id, lat, lon (unique stations).
    gee_project : str
        GEE project ID for authentication.
    cache_path : Path, optional
        Path to cache CSV. If exists, load from cache instead of querying GEE.

    Returns
    -------
    pd.DataFrame
        Columns: station_id, elevation, slope, aspect, tpi, elevation_mean_5km.
    """
    if cache_path is None:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = _CACHE_DIR / "terrain_features.csv"

    # Return cached results if available
    if cache_path.exists():
        logger.info("Loading cached terrain features from %s", cache_path)
        return pd.read_csv(cache_path)

    # Initialize GEE
    ee.Initialize(project=gee_project)

    srtm = ee.Image("USGS/SRTMGL1_003")
    slope = ee.Terrain.slope(srtm)
    aspect = ee.Terrain.aspect(srtm)
    terrain_img = srtm.select("elevation").addBands(slope).addBands(aspect)

    # --- Point features (elevation, slope, aspect) via reduceRegions ---
    points = ee.FeatureCollection(
        [
            ee.Feature(
                ee.Geometry.Point([float(row.lon), float(row.lat)]),
                {"station_id": str(row.station_id)},
            )
            for row in stations_df.itertuples()
        ]
    )

    point_results = terrain_img.reduceRegions(
        collection=points,
        reducer=ee.Reducer.first(),
        scale=30,
    ).getInfo()

    # Parse point results into a dict keyed by station_id
    point_data: dict[str, dict] = {}
    for feat in point_results["features"]:
        props = feat["properties"]
        sid = props["station_id"]
        point_data[sid] = {
            "elevation": props.get("elevation"),
            "slope": props.get("slope"),
            "aspect": props.get("aspect"),
        }

    # --- 5km buffer mean elevation (for TPI) via batched reduceRegion ---
    station_list = stations_df.to_dict("records")
    batch_size = 50
    elev_mean_5km: dict[str, float] = {}

    for i in range(0, len(station_list), batch_size):
        batch = station_list[i : i + batch_size]
        logger.info(
            "Processing TPI batch %d/%d (%d stations)",
            i // batch_size + 1,
            (len(station_list) + batch_size - 1) // batch_size,
            len(batch),
        )

        for station in batch:
            sid = str(station["station_id"])
            point = ee.Geometry.Point([float(station["lon"]), float(station["lat"])])
            buffer = point.buffer(5000)  # 5 km buffer

            mean_elev = (
                srtm.select("elevation")
                .reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=buffer,
                    scale=30,
                    maxPixels=1e7,
                )
                .get("elevation")
                .getInfo()
            )
            elev_mean_5km[sid] = mean_elev

    # --- Assemble output DataFrame ---
    rows = []
    for _, row in stations_df.iterrows():
        sid = str(row["station_id"])
        pd_row = point_data.get(sid, {})
        elev = pd_row.get("elevation")
        elev_mean = elev_mean_5km.get(sid)

        tpi = None
        if elev is not None and elev_mean is not None:
            tpi = elev - elev_mean

        rows.append(
            {
                "station_id": sid,
                "elevation": elev,
                "slope": pd_row.get("slope"),
                "aspect": pd_row.get("aspect"),
                "tpi": tpi,
                "elevation_mean_5km": elev_mean,
            }
        )

    result = pd.DataFrame(rows)

    # Cache results
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(cache_path, index=False)
    logger.info("Cached terrain features to %s", cache_path)

    return result
