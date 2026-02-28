"""
plot_terrain_refinement.py — 4-panel terrain refinement progression

Shows Perdigao DEM at 4 resolutions (10 km, 5 km, 1 km, 500 m) to
illustrate the convergence of terrain representation.

Output: figures/terrain_refinement_4panel.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from services.validation.plot_style import apply_style, save

logger = logging.getLogger(__name__)

RESOLUTIONS_M = [10_000, 5_000, 1_000, 500]
CENTRAL_KM    = 25.0     # central zone width [km]
SITE_LAT      = 39.716
SITE_LON      = -7.740


def load_and_resample(srtm_tif: Path, resolution_m: float) -> tuple:
    """Load SRTM and resample to target resolution over central 25×25 km."""
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.transform import from_bounds
        from rasterio.warp import reproject
    except ImportError:
        raise ImportError("rasterio required: pip install rasterio")

    DEG_LAT = 1.0 / 111_000.0
    DEG_LON = 1.0 / (111_000.0 * np.cos(np.radians(SITE_LAT)))
    half_deg_lat = (CENTRAL_KM * 500) * DEG_LAT
    half_deg_lon = (CENTRAL_KM * 500) * DEG_LON

    west  = SITE_LON - half_deg_lon
    east  = SITE_LON + half_deg_lon
    south = SITE_LAT - half_deg_lat
    north = SITE_LAT + half_deg_lat

    n_col = max(2, int(round(CENTRAL_KM * 1000 / resolution_m)))
    n_row = n_col
    target_transform = from_bounds(west, south, east, north, n_col, n_row)

    with rasterio.open(srtm_tif) as src:
        elevation = np.zeros((n_row, n_col), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=elevation,
            dst_transform=target_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.bilinear,
        )

    x_km = np.linspace(-CENTRAL_KM / 2, CENTRAL_KM / 2, n_col)
    y_km = np.linspace(-CENTRAL_KM / 2, CENTRAL_KM / 2, n_row)
    return x_km, y_km, elevation


def _fake_dem(resolution_m: float) -> tuple:
    """Generate a synthetic 2-ridge DEM for testing without SRTM."""
    n = max(2, int(round(CENTRAL_KM * 1000 / resolution_m)))
    x_km = np.linspace(-CENTRAL_KM / 2, CENTRAL_KM / 2, n)
    y_km = np.linspace(-CENTRAL_KM / 2, CENTRAL_KM / 2, n)
    X, Y = np.meshgrid(x_km, y_km)
    # Two parallel Gaussian ridges oriented NE-SW (~40°)
    theta = np.radians(40)
    xr = X * np.cos(theta) + Y * np.sin(theta)
    yr = -X * np.sin(theta) + Y * np.cos(theta)
    z = (
        300
        + 150 * np.exp(-((xr - 0.7) ** 2) / 0.3 - yr**2 / 5)
        + 150 * np.exp(-((xr + 0.7) ** 2) / 0.3 - yr**2 / 5)
    )
    return x_km, y_km, z.astype(np.float32)


def plot_terrain_refinement(
    srtm_tif: Path | None,
    output_path: Path,
    resolutions_m: list[float] = RESOLUTIONS_M,
) -> None:
    """Create 4-panel terrain refinement figure."""
    apply_style()
    fig, axes = plt.subplots(1, len(resolutions_m), figsize=(14, 4))
    if len(resolutions_m) == 1:
        axes = [axes]

    ls = LightSource(azdeg=315, altdeg=30)

    for ax, res_m in zip(axes, resolutions_m):
        try:
            if srtm_tif is not None and Path(srtm_tif).exists():
                x_km, y_km, elev = load_and_resample(Path(srtm_tif), res_m)
            else:
                x_km, y_km, elev = _fake_dem(res_m)
        except Exception as exc:
            logger.warning("Could not load DEM at %.0f m: %s — using synthetic", res_m, exc)
            x_km, y_km, elev = _fake_dem(res_m)

        X, Y = np.meshgrid(x_km, y_km)

        # Hillshade
        rgb = ls.shade(
            elev, cmap=plt.get_cmap("terrain"),
            vert_exag=2.5, blend_mode="overlay",
            vmin=float(elev.min()), vmax=float(elev.max()),
        )
        ax.imshow(
            rgb,
            extent=[-CENTRAL_KM / 2, CENTRAL_KM / 2,
                    -CENTRAL_KM / 2, CENTRAL_KM / 2],
            origin="lower", aspect="equal",
        )

        # Grid lines (mesh cell edges)
        for xv in x_km[::max(1, len(x_km) // 10)]:
            ax.axvline(xv, color="white", lw=0.3, alpha=0.5)
        for yv in y_km[::max(1, len(y_km) // 10)]:
            ax.axhline(yv, color="white", lw=0.3, alpha=0.5)

        # Tower positions (approximate)
        towers = [
            ("T20",  0.6,  0.8),  # NW ridge
            ("T25",  1.1, -0.1),  # valley
            ("T13",  0.8, -0.5),  # flank
        ]
        for name, tx, ty in towers:
            ax.scatter(tx, ty, marker="^", s=30, c="red", zorder=5, linewidths=0.5,
                       edgecolors="white")

        label = f"{int(res_m):,} m" if res_m < 1000 else f"{res_m/1000:.0f} km"
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Distance E [km]")
        if ax == axes[0]:
            ax.set_ylabel("Distance N [km]")

    fig.suptitle("Terrain representation — Perdigão (25×25 km central zone)",
                 fontsize=12, y=1.01)
    save(fig, str(output_path))
    plt.close(fig)
    logger.info("Saved: %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="4-panel terrain refinement plot")
    parser.add_argument("--srtm",   default=None,
                        help="SRTM GeoTIFF (uses synthetic DEM if absent)")
    parser.add_argument("--output", default="figures/terrain_refinement_4panel.png")
    args = parser.parse_args()
    plot_terrain_refinement(
        srtm_tif=args.srtm,
        output_path=Path(args.output),
    )
