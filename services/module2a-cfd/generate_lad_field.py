"""
generate_lad_field.py — Map land cover raster to OpenFOAM fields

Reads the 3-band GeoTIFF from ingest_landcover.py (z₀, d, LAD) and the
OpenFOAM mesh cell centres to produce:

  0/LAD              — volScalarField [1/m], 3-D leaf area density
  constant/z0Field   — surfaceScalarField [m] on terrain patch faces
                       (for spatially varying wall functions)

The column-average LAD from the raster is distributed vertically:
  LAD(x,y,z) = LAD_column(x,y)  if z_cell < h_canopy(x,y)
  LAD(x,y,z) = 0                otherwise
where h_canopy ≈ 1.5 × d  (from d = 2/3 × h rule used in ingest_landcover.py)

Usage
-----
    python generate_lad_field.py \\
        --case-dir data/cases/perdigao_100m/ \\
        --landcover ../../data/raw/landcover_perdigao.tif \\
        --site-lat 39.716 --site-lon -7.74

References
----------
    Sogachev & Panferov (2006) — canopy source terms for k-epsilon
    Lang et al. (2023) — ETH Global Canopy Height
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Read OpenFOAM boundary face centres (terrain patch)
# ---------------------------------------------------------------------------

def read_boundary_face_centres(case_dir: Path, patch_name: str = "terrain") -> np.ndarray:
    """Read face centres for a specific boundary patch.

    Requires `postProcess -func writeCellCentres` to have been run,
    which writes boundary face centres in 0/Cf (vectorField).
    Falls back to parsing constant/polyMesh/boundary + faceCentres.

    Returns (N_faces, 3) array of face centre coordinates.
    """
    # Try 0/Cf written by writeCellCentres
    cf_path = case_dir / "0" / "Cf"
    if cf_path.exists():
        return _parse_boundary_face_centres(cf_path, patch_name)

    raise FileNotFoundError(
        f"Cannot find face centres. Run `postProcess -func writeCellCentres` in {case_dir}"
    )


def _parse_boundary_face_centres(filepath: Path, patch_name: str) -> np.ndarray:
    """Parse face centres for a boundary patch from the Cf field."""
    text = filepath.read_text()

    # Find the patch block
    pattern = rf'{patch_name}\s*\n\s*\{{[^}}]*\bvalue\b\s+nonuniform\s+List<vector>\s*\n(\d+)\s*\n\('
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        # Try uniform
        pattern_u = rf'{patch_name}\s*\n\s*\{{[^}}]*\bvalue\b\s+uniform\s+\(([^)]+)\)'
        match_u = re.search(pattern_u, text, re.DOTALL)
        if match_u:
            logger.warning("Terrain face centres are uniform — single face?")
            vals = [float(v) for v in match_u.group(1).split()]
            return np.array([vals])
        raise ValueError(f"Cannot parse face centres for patch '{patch_name}' in {filepath}")

    n = int(match.group(1))
    start = match.end()
    end = text.index(')', start)
    vectors = re.findall(
        r'\(\s*([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s*\)',
        text[start:end]
    )
    arr = np.array([[float(x), float(y), float(z)] for x, y, z in vectors[:n]])
    logger.info("Read %d face centres for patch '%s'", len(arr), patch_name)
    return arr


# ---------------------------------------------------------------------------
# Read cell centres (reuse from init_from_era5)
# ---------------------------------------------------------------------------

def read_cell_centres(case_dir: Path) -> np.ndarray:
    """Read cell centres from 0/Cx, 0/Cy, 0/Cz."""
    from init_from_era5 import read_cell_centres as _read_cc
    return _read_cc(case_dir)


# ---------------------------------------------------------------------------
# Raster interpolation
# ---------------------------------------------------------------------------

def sample_raster_at_xy(
    raster_path: Path,
    x_m: np.ndarray,
    y_m: np.ndarray,
    site_lat: float,
    site_lon: float,
    band: int = 1,
) -> np.ndarray:
    """Sample a GeoTIFF band at local (x, y) coordinates [m].

    Converts local metres to lon/lat, then samples the raster using
    nearest-neighbour interpolation.

    Parameters
    ----------
    raster_path : Path to multi-band GeoTIFF (EPSG:4326).
    x_m, y_m : 1-D arrays of local x, y coordinates [m] from site centre.
    site_lat, site_lon : Site centre coordinates [degrees].
    band : Raster band number (1-indexed).

    Returns
    -------
    values : 1-D array of raster values at the query points.
    """
    import rasterio

    DEG_PER_M_LAT = 1.0 / 111_000.0
    DEG_PER_M_LON = 1.0 / (111_000.0 * np.cos(np.radians(site_lat)))

    lons = site_lon + x_m * DEG_PER_M_LON
    lats = site_lat + y_m * DEG_PER_M_LAT

    with rasterio.open(str(raster_path)) as src:
        data = src.read(band)
        nodata = src.nodata if src.nodata is not None else -9999.0

        # Convert lon/lat to row/col
        rows, cols = rasterio.transform.rowcol(src.transform, lons, lats)
        rows = np.clip(np.array(rows), 0, data.shape[0] - 1)
        cols = np.clip(np.array(cols), 0, data.shape[1] - 1)

        values = data[rows, cols].astype(np.float64)
        values[values == nodata] = 0.0

    return values


# ---------------------------------------------------------------------------
# Write OpenFOAM fields
# ---------------------------------------------------------------------------

def write_vol_scalar_field(
    filepath: Path,
    field_name: str,
    data: np.ndarray,
    dimensions: str = "[0 0 0 0 0 0 0]",
) -> None:
    """Write a volScalarField with nonuniform internalField."""
    n = len(data)
    lines = [
        '/*--------------------------------*- C++ -*----------------------------------*\\',
        '  =========                 |',
        '  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox',
        '   \\\\    /   O peration     | Version: 10',
        '    \\\\  /    A nd           | Web:     www.openfoam.org',
        '     \\\\/     M anipulation  |',
        '\\*---------------------------------------------------------------------------*/',
        'FoamFile',
        '{',
        '    version     2.0;',
        '    format      ascii;',
        '    class       volScalarField;',
        '    location    "0";',
        f'    object      {field_name};',
        '}',
        '// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //',
        '',
        f'dimensions      {dimensions};',
        '',
        f'internalField   nonuniform List<scalar>',
        f'{n}',
        '(',
    ]
    for i in range(n):
        lines.append(f'{data[i]:.6e}')
    lines.extend([')', ';', ''])

    # Minimal boundary conditions (zeroGradient everywhere)
    lines.extend([
        'boundaryField',
        '{',
        '    ".*"',
        '    {',
        '        type            zeroGradient;',
        '    }',
        '}',
        '',
        '// ************************************************************************* //',
    ])

    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text('\n'.join(lines))
    logger.info("Wrote %s: %d cells, range [%.4f, %.4f]",
                filepath, n, data.min(), data.max())


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate_lad_field(
    case_dir: Path,
    landcover_tif: Path,
    site_lat: float,
    site_lon: float,
    lai: float = 4.0,
    cd: float = 0.2,
) -> dict:
    """Generate 0/LAD and 0/Cd volScalarFields from land cover raster.

    Supports two input formats:
      - 3-band GeoTIFF (z0, d, LAD) from ingest_landcover.py
      - 1-band GeoTIFF (canopy height [m]) from ETH Global Canopy Height 2020

    For 1-band input, LAD is computed as: LAD = LAI / h_canopy  [1/m]
    (uniform vertical distribution within canopy).

    Parameters
    ----------
    case_dir : OpenFOAM case directory (with mesh + Cx/Cy/Cz generated).
    landcover_tif : GeoTIFF raster (1 or 3 bands).
    site_lat, site_lon : Site centre coordinates [degrees].
    lai : Leaf Area Index (default 4.0, typical eucalyptus).
    cd : Drag coefficient (default 0.2, Sogachev & Panferov 2006).
    """
    import rasterio

    # Read cell centres
    centres = read_cell_centres(case_dir)
    n = len(centres)
    x_m, y_m, z_m = centres[:, 0], centres[:, 1], centres[:, 2]

    # Detect raster format
    with rasterio.open(str(landcover_tif)) as src:
        n_bands = src.count
    logger.info("Raster %s: %d band(s)", landcover_tif.name, n_bands)

    if n_bands >= 3:
        # Legacy 3-band format: z0, d, LAD
        logger.info("Sampling 3-band land cover raster at %d cells", n)
        d_col = sample_raster_at_xy(landcover_tif, x_m, y_m, site_lat, site_lon, band=2)
        lad_col = sample_raster_at_xy(landcover_tif, x_m, y_m, site_lat, site_lon, band=3)
        h_canopy = 1.5 * d_col
    else:
        # ETH Canopy Height: single band = h_canopy [m]
        logger.info("Sampling ETH canopy height raster at %d cells", n)
        h_canopy = sample_raster_at_xy(landcover_tif, x_m, y_m, site_lat, site_lon, band=1)
        h_canopy = np.clip(h_canopy, 0, 50)  # cap at 50m
        # LAD = LAI / h_canopy (uniform within canopy)
        lad_col = np.where(h_canopy > 1.0, lai / h_canopy, 0.0)

    # Terrain height per column: z_terrain(x,y) = min z over cells at (x,y)
    # Approximate: find nearest terrain face z
    from scipy.interpolate import NearestNDInterpolator
    from generate_z0_field import read_face_centres, read_boundary_info

    terrain_patch = "terrain"
    try:
        fc = read_face_centres(case_dir, terrain_patch)
        terrain_interp = NearestNDInterpolator(fc[:, :2], fc[:, 2])
        z_terrain = terrain_interp(x_m, y_m)
    except Exception:
        # Fallback: use minimum z in local column
        logger.warning("Could not read terrain faces, using column-min z")
        z_terrain = np.full(n, z_m.min())

    z_agl = z_m - z_terrain

    # 3-D LAD: non-zero only within canopy (z_agl < h_canopy)
    lad_3d = np.where(
        (z_agl < h_canopy) & (z_agl >= 0) & (lad_col > 0),
        lad_col,
        0.0,
    )

    n_canopy = np.count_nonzero(lad_3d > 0)
    logger.info(
        "LAD field: %d/%d cells in canopy (%.1f%%), max LAD=%.3f 1/m, max h=%.1fm",
        n_canopy, n, 100.0 * n_canopy / n,
        lad_3d.max() if n_canopy else 0.0,
        h_canopy.max(),
    )

    # Write 0/LAD
    write_vol_scalar_field(
        case_dir / "0" / "LAD", "LAD", lad_3d,
        dimensions="[0 -1 0 0 0 0 0]",
    )

    # Write 0/Cd (non-zero only where LAD > 0)
    cd_field = np.where(lad_3d > 0, cd, 0.0)
    write_vol_scalar_field(
        case_dir / "0" / "Cd", "Cd", cd_field,
        dimensions="[0 0 0 0 0 0 0]",
    )

    stats = {
        "n_cells": n,
        "n_canopy_cells": int(n_canopy),
        "canopy_fraction": float(n_canopy / n),
        "lad_max": float(lad_3d.max()),
        "h_canopy_max": float(h_canopy.max()),
        "h_canopy_mean_nonzero": float(h_canopy[h_canopy > 1].mean()) if (h_canopy > 1).any() else 0,
    }
    logger.info("Canopy field generation complete: %s", stats)
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate LAD volScalarField from land cover raster"
    )
    parser.add_argument("--case-dir", required=True, help="OpenFOAM case directory")
    parser.add_argument("--landcover", required=True, help="Land cover GeoTIFF (3 bands: z0, d, LAD)")
    parser.add_argument("--site-lat", type=float, required=True, help="Site centre latitude [deg]")
    parser.add_argument("--site-lon", type=float, required=True, help="Site centre longitude [deg]")
    args = parser.parse_args()

    generate_lad_field(
        case_dir=Path(args.case_dir),
        landcover_tif=Path(args.landcover),
        site_lat=args.site_lat,
        site_lon=args.site_lon,
    )
