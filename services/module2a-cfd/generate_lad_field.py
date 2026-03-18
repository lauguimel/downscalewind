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
) -> dict:
    """Generate 0/LAD volScalarField from land cover raster.

    Parameters
    ----------
    case_dir : OpenFOAM case directory (with mesh generated).
    landcover_tif : 3-band GeoTIFF from ingest_landcover.py
                    (band 1: z0, band 2: d, band 3: LAD).
    site_lat, site_lon : Site centre coordinates [degrees].

    Returns
    -------
    dict with stats: n_canopy_cells, lad_max, z0_range, etc.
    """
    # Read cell centres
    centres = read_cell_centres(case_dir)
    n = len(centres)
    x_m = centres[:, 0]
    y_m = centres[:, 1]
    z_m = centres[:, 2]

    # Sample raster bands at cell (x, y) positions
    logger.info("Sampling land cover raster at %d cell centres", n)
    z0_col = sample_raster_at_xy(landcover_tif, x_m, y_m, site_lat, site_lon, band=1)
    d_col = sample_raster_at_xy(landcover_tif, x_m, y_m, site_lat, site_lon, band=2)
    lad_col = sample_raster_at_xy(landcover_tif, x_m, y_m, site_lat, site_lon, band=3)

    # Estimate canopy height from displacement height: h = 1.5 * d
    h_canopy = 1.5 * d_col

    # 3-D LAD: LAD > 0 only within the canopy (z_cell < h_canopy)
    lad_3d = np.where(
        (z_m < h_canopy) & (lad_col > 0),
        lad_col,
        0.0,
    )

    n_canopy = np.count_nonzero(lad_3d > 0)
    logger.info(
        "LAD field: %d/%d cells within canopy (%.1f%%), max LAD=%.3f 1/m",
        n_canopy, n, 100.0 * n_canopy / n, lad_3d.max() if n_canopy > 0 else 0.0,
    )

    # Write 0/LAD
    write_vol_scalar_field(
        case_dir / "0" / "LAD",
        "LAD",
        lad_3d,
        dimensions="[0 -1 0 0 0 0 0]",  # [1/m]
    )

    # Write 0/Cd (uniform drag coefficient, required by atmPlantCanopyUSource)
    cd_value = 0.2  # Sogachev & Panferov (2006) default
    cd_field = np.where(lad_3d > 0, cd_value, 0.0)
    write_vol_scalar_field(
        case_dir / "0" / "Cd",
        "Cd",
        cd_field,
        dimensions="[0 0 0 0 0 0 0]",  # dimensionless
    )

    stats = {
        "n_cells": n,
        "n_canopy_cells": int(n_canopy),
        "lad_max": float(lad_3d.max()),
        "z0_range": [float(z0_col.min()), float(z0_col.max())],
        "h_canopy_max": float(h_canopy.max()),
    }

    logger.info("LAD field generation complete: %s", stats)
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
