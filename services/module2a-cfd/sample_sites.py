"""
sample_sites.py — Spatial sampling of terrain patches from SRTM DEM.

Tiles a DEM into overlapping 14×14 km patches (= TBM cylinder diameter),
computes morphological descriptors, clusters them, and selects ~N_total
representative sites covering the full morphological diversity.

Usage
-----
    cd services/module2a-cfd
    python sample_sites.py \
        --dem ../../data/raw/srtm_europe.tif \
        --output-dir ../../data/campaign/sites \
        --n-clusters 20 \
        --n-per-cluster 50

Dependencies: numpy, rasterio, scikit-learn, matplotlib, pandas
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Domain defaults (metres)
PATCH_SIZE_M = 14_000  # cylinder diameter
OVERLAP_FRAC = 0.5
MIN_ELEV_LAND = 5.0  # below this + flat → ocean
MAX_NODATA_FRAC = 0.20
MAX_RELIEF_M = 1500  # exclude extreme alpine terrain


def load_dem(dem_path: Path, bbox: tuple[float, float, float, float] | None = None):
    """Load DEM and optionally crop to bounding box (lon_min, lat_min, lon_max, lat_max)."""
    import rasterio
    from rasterio.windows import from_bounds

    with rasterio.open(dem_path) as src:
        if bbox is not None:
            lon_min, lat_min, lon_max, lat_max = bbox
            window = from_bounds(lon_min, lat_min, lon_max, lat_max, src.transform)
            elev = src.read(1, window=window)
            transform = src.window_transform(window)
        else:
            elev = src.read(1)
            transform = src.transform
        nodata = src.nodata
        crs = src.crs

    return elev, transform, nodata, crs


def compute_pixel_size_m(transform, lat_center: float) -> tuple[float, float]:
    """Approximate pixel size in metres from affine transform at a given latitude."""
    dx_deg = abs(transform.a)
    dy_deg = abs(transform.e)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(np.radians(lat_center))
    return dx_deg * m_per_deg_lon, dy_deg * m_per_deg_lat


def tile_dem(
    elev: np.ndarray,
    transform,
    nodata: float | None,
    patch_size_m: float = PATCH_SIZE_M,
    overlap: float = OVERLAP_FRAC,
    max_nodata_frac: float = MAX_NODATA_FRAC,
    min_elev_land: float = MIN_ELEV_LAND,
) -> list[dict]:
    """Tile the DEM into overlapping patches and compute morphological descriptors."""
    nrows, ncols = elev.shape

    # Approximate pixel size at centre of the DEM
    lat_center = transform.f + transform.e * nrows / 2
    dx_m, dy_m = compute_pixel_size_m(transform, lat_center)

    patch_nx = max(1, int(round(patch_size_m / dx_m)))
    patch_ny = max(1, int(round(patch_size_m / dy_m)))
    step_x = max(1, int(round(patch_nx * (1 - overlap))))
    step_y = max(1, int(round(patch_ny * (1 - overlap))))

    logger.info(
        "Tiling: patch %d×%d px (%.0f×%.0f m), step %d×%d, DEM %d×%d",
        patch_nx, patch_ny, patch_nx * dx_m, patch_ny * dy_m,
        step_x, step_y, ncols, nrows,
    )

    patches = []
    site_id = 0

    for iy in range(0, nrows - patch_ny + 1, step_y):
        for ix in range(0, ncols - patch_nx + 1, step_x):
            tile = elev[iy : iy + patch_ny, ix : ix + patch_nx].astype(np.float64)

            # NoData mask
            if nodata is not None:
                mask = tile == nodata
            else:
                mask = np.isnan(tile)
            nodata_frac = mask.sum() / tile.size
            if nodata_frac > max_nodata_frac:
                continue

            # Replace nodata with NaN for stats
            tile_clean = np.where(mask, np.nan, tile)

            # Ocean filter: mostly low elevation AND flat
            valid = tile_clean[~np.isnan(tile_clean)]
            if len(valid) < 100:
                continue
            median_elev = np.nanmedian(tile_clean)
            std_elev = np.nanstd(tile_clean)
            if median_elev < min_elev_land and std_elev < 10.0:
                continue

            # Centre coordinates (geographic)
            cx_px = ix + patch_nx / 2
            cy_px = iy + patch_ny / 2
            lon_center = transform.c + transform.a * cx_px
            lat_center = transform.f + transform.e * cy_px

            # Morphological descriptors
            desc = compute_descriptors(tile_clean, dx_m, dy_m)
            if desc is None:
                continue

            patches.append({
                "site_id": f"site_{site_id:05d}",
                "lat": round(lat_center, 5),
                "lon": round(lon_center, 5),
                "ix": ix,
                "iy": iy,
                **desc,
            })
            site_id += 1

    logger.info("Generated %d valid patches from %d×%d grid", len(patches), ncols, nrows)
    return patches


def compute_descriptors(
    tile: np.ndarray, dx_m: float, dy_m: float
) -> dict | None:
    """Compute 5 morphological descriptors for a terrain patch.

    Returns None if the patch has insufficient valid data.
    """
    valid = tile[~np.isnan(tile)]
    if len(valid) < 100:
        return None

    std_elev = float(np.nanstd(tile))
    max_relief = float(np.nanmax(tile) - np.nanmin(tile))

    # Slope (degrees) via finite differences
    dy, dx = np.gradient(tile, dy_m, dx_m)
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    mean_slope = float(np.nanmean(slope))

    # TRI: Terrain Ruggedness Index (mean absolute elevation difference in 3×3 neighbourhood)
    tri = _compute_tri(tile)

    # Elongation: ratio of principal axes of elevation variance ellipse
    elongation = _compute_elongation(tile)

    return {
        "std_elev": round(std_elev, 2),
        "mean_slope": round(mean_slope, 3),
        "TRI": round(tri, 2),
        "max_relief": round(max_relief, 1),
        "elongation": round(elongation, 4),
    }


def _compute_tri(tile: np.ndarray) -> float:
    """Terrain Ruggedness Index: mean absolute difference from 3×3 neighbourhood."""
    from scipy.ndimage import uniform_filter

    # Mean of 3×3 neighbourhood
    tile_filled = np.where(np.isnan(tile), np.nanmean(tile), tile)
    mean_3x3 = uniform_filter(tile_filled, size=3, mode="nearest")
    tri_map = np.abs(tile_filled - mean_3x3)
    return float(np.nanmean(tri_map))


def _compute_elongation(tile: np.ndarray) -> float:
    """Ratio of principal axes of the elevation variance ellipse.

    Values near 1.0 = isotropic (bowl/cone).
    Values >> 1.0 = elongated (ridge/valley).
    """
    rows, cols = np.where(~np.isnan(tile))
    if len(rows) < 10:
        return 1.0
    vals = tile[rows, cols]

    # Weighted covariance of (row, col) positions using elevation as weight
    weights = vals - vals.min() + 1.0  # shift to positive
    total_w = weights.sum()
    mean_r = (weights * rows).sum() / total_w
    mean_c = (weights * cols).sum() / total_w

    dr = rows - mean_r
    dc = cols - mean_c
    cov_rr = (weights * dr * dr).sum() / total_w
    cov_cc = (weights * dc * dc).sum() / total_w
    cov_rc = (weights * dr * dc).sum() / total_w

    # Eigenvalues of 2×2 covariance matrix
    trace = cov_rr + cov_cc
    det = cov_rr * cov_cc - cov_rc * cov_rc
    disc = max(0.0, trace**2 / 4 - det)
    lambda1 = trace / 2 + np.sqrt(disc)
    lambda2 = trace / 2 - np.sqrt(disc)

    if lambda2 <= 0:
        return 10.0  # degenerate — very elongated
    return float(np.sqrt(lambda1 / lambda2))


def cluster_and_sample(
    patches: list[dict],
    n_clusters: int = 20,
    n_per_cluster: int = 50,
    max_relief: float = MAX_RELIEF_M,
    random_state: int = 42,
) -> pd.DataFrame:
    """Cluster patches on morphological descriptors and sample representatives."""
    df = pd.DataFrame(patches)
    logger.info("Total patches before filtering: %d", len(df))

    # Exclude extreme alpine terrain
    df = df[df["max_relief"] <= max_relief].copy()
    logger.info("After max_relief <= %.0f filter: %d", max_relief, len(df))

    if len(df) < n_clusters:
        logger.warning("Fewer patches (%d) than clusters (%d), returning all", len(df), n_clusters)
        df["cluster_id"] = 0
        return df

    # Feature matrix
    feature_cols = ["std_elev", "mean_slope", "TRI", "max_relief", "elongation"]
    X = df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means clustering
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df["cluster_id"] = km.fit_predict(X_scaled)

    # Sample: medoid + random per cluster
    selected = []
    rng = np.random.default_rng(random_state)

    for cid in range(n_clusters):
        cluster_df = df[df["cluster_id"] == cid]
        if len(cluster_df) == 0:
            continue

        # Medoid: point closest to cluster centroid
        X_cluster = scaler.transform(cluster_df[feature_cols].values)
        centroid = km.cluster_centers_[cid]
        dists = np.linalg.norm(X_cluster - centroid, axis=1)
        medoid_idx = cluster_df.index[np.argmin(dists)]
        selected.append(medoid_idx)

        # Random members (up to n_per_cluster - 1)
        remaining = cluster_df.index.drop(medoid_idx)
        n_extra = min(n_per_cluster - 1, len(remaining))
        if n_extra > 0:
            extra = rng.choice(remaining, size=n_extra, replace=False)
            selected.extend(extra)

    result = df.loc[selected].copy()
    result = result.reset_index(drop=True)
    result["site_id"] = [f"site_{i:05d}" for i in range(len(result))]
    logger.info("Selected %d sites across %d clusters", len(result), n_clusters)
    return result


def plot_results(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate scatter plot and map of selected sites."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # (a) Scatter: TRI vs std_elev, coloured by cluster
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        df["std_elev"], df["TRI"],
        c=df["cluster_id"], cmap="tab20", s=15, alpha=0.7, edgecolors="none",
    )
    ax.set_xlabel("Elevation std dev [m]")
    ax.set_ylabel("Terrain Ruggedness Index [m]")
    ax.set_title(f"Morphological clustering — {len(df)} sites, {df['cluster_id'].nunique()} clusters")
    plt.colorbar(scatter, ax=ax, label="Cluster ID")
    fig.tight_layout()
    fig.savefig(figures_dir / "morphological_clusters.png", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", figures_dir / "morphological_clusters.png")

    # (b) Map of selected sites
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        df["lon"], df["lat"],
        c=df["cluster_id"], cmap="tab20", s=10, alpha=0.7, edgecolors="none",
    )
    ax.set_xlabel("Longitude [°]")
    ax.set_ylabel("Latitude [°]")
    ax.set_title(f"Selected sites — {len(df)} patches")
    ax.set_aspect("equal")
    plt.colorbar(scatter, ax=ax, label="Cluster ID")
    fig.tight_layout()
    fig.savefig(figures_dir / "site_map.png", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", figures_dir / "site_map.png")


def main():
    parser = argparse.ArgumentParser(
        description="Spatial sampling of terrain patches from SRTM DEM"
    )
    parser.add_argument(
        "--dem", type=Path, required=True,
        help="Path to SRTM DEM GeoTIFF",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("../../data/campaign/sites"),
        help="Output directory for CSV and figures",
    )
    parser.add_argument(
        "--bbox", type=float, nargs=4, default=[-10.0, 40.0, 10.0, 55.0],
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        help="Bounding box [lon_min, lat_min, lon_max, lat_max]",
    )
    parser.add_argument("--patch-size-m", type=float, default=PATCH_SIZE_M)
    parser.add_argument("--overlap", type=float, default=OVERLAP_FRAC)
    parser.add_argument("--n-clusters", type=int, default=20)
    parser.add_argument("--n-per-cluster", type=int, default=50)
    parser.add_argument("--max-relief", type=float, default=MAX_RELIEF_M)
    parser.add_argument("--max-nodata-frac", type=float, default=MAX_NODATA_FRAC)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cache", type=Path, default=None,
        help="Cache file for intermediate tile results (pickle)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load and tile DEM
    cache_path = args.cache or (args.output_dir / "patches_cache.pkl")
    if cache_path.exists():
        logger.info("Loading cached patches from %s", cache_path)
        with open(cache_path, "rb") as f:
            patches = pickle.load(f)
    else:
        logger.info("Loading DEM from %s", args.dem)
        bbox = tuple(args.bbox)
        elev, transform, nodata, crs = load_dem(args.dem, bbox=bbox)
        logger.info("DEM shape: %s, CRS: %s", elev.shape, crs)

        patches = tile_dem(
            elev, transform, nodata,
            patch_size_m=args.patch_size_m,
            overlap=args.overlap,
            max_nodata_frac=args.max_nodata_frac,
        )
        with open(cache_path, "wb") as f:
            pickle.dump(patches, f)
        logger.info("Cached %d patches to %s", len(patches), cache_path)

    # Cluster and sample
    df = cluster_and_sample(
        patches,
        n_clusters=args.n_clusters,
        n_per_cluster=args.n_per_cluster,
        max_relief=args.max_relief,
        random_state=args.seed,
    )

    # Save CSV
    csv_path = args.output_dir / "sites.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved %d sites to %s", len(df), csv_path)

    # Plots
    plot_results(df, args.output_dir)

    # Summary
    print(f"\n{'='*60}")
    print(f"Selected {len(df)} sites across {df['cluster_id'].nunique()} clusters")
    print(f"\nCluster distribution:")
    print(df.groupby("cluster_id").agg(
        count=("site_id", "count"),
        mean_std_elev=("std_elev", "mean"),
        mean_slope=("mean_slope", "mean"),
        mean_TRI=("TRI", "mean"),
    ).to_string())
    print(f"\nOutputs:")
    print(f"  CSV: {csv_path}")
    print(f"  Figures: {args.output_dir / 'figures'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
