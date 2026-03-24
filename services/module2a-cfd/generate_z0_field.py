"""
generate_z0_field.py — Generate spatially-varying z0 from ESA WorldCover 2021

Reads the WorldCover GeoTIFF, reprojects to local coordinates (matching
the OpenFOAM mesh), reclassifies land cover to aerodynamic roughness z0,
and writes z0 as a constant/boundaryData/terrain/0/z0 file for use by
atmNutkWallFunction and atmEpsilonWallFunction.

The z0 lookup follows the "integrated roughness" approach (no explicit canopy):
  - Tree cover → z0 = 0.5 m  (integrates canopy drag into roughness)
  - Shrubland  → z0 = 0.1 m
  - Grassland  → z0 = 0.03 m
  - Cropland   → z0 = 0.05 m
  - Built-up   → z0 = 1.0 m
  - Bare        → z0 = 0.005 m
  - Water       → z0 = 0.0002 m

Usage
-----
    python generate_z0_field.py \\
        --case-dir data/cases/poc_tbm_25ts/case_ts00 \\
        --worldcover data/raw/worldcover_perdigao.tif \\
        --site-lat 39.716 --site-lon -7.740
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ESA WorldCover 2021 classes → aerodynamic roughness z0 [m]
# "Integrated roughness" approach: tree effect is in z0, NOT in explicit canopy.
# References: Wieringa (1992), Davenport classification, Silva et al. (2007)
Z0_LOOKUP = {
    10: 0.5,     # Tree cover (integrated canopy drag)
    20: 0.10,    # Shrubland
    30: 0.03,    # Grassland
    40: 0.05,    # Cropland
    50: 1.0,     # Built-up
    60: 0.005,   # Bare / sparse vegetation
    70: 0.001,   # Snow and ice
    80: 0.0002,  # Permanent water bodies
    90: 0.10,    # Herbaceous wetland
    95: 0.5,     # Mangroves
    100: 0.01,   # Moss and lichen
}

# Fallback z0 for unknown classes
Z0_DEFAULT = 0.05


def read_boundary_info(case_dir: Path) -> dict:
    """Parse constant/polyMesh/boundary for patch startFace/nFaces."""
    boundary_path = case_dir / "constant" / "polyMesh" / "boundary"
    text = boundary_path.read_text()
    match = re.search(r'^\s*(\d+)\s*\(', text, re.MULTILINE)
    if not match:
        raise ValueError(f"Cannot parse boundary: {boundary_path}")
    block = text[match.end():]
    patches = {}
    for m in re.finditer(r'(\w+)\s*\{([^}]+)\}', block):
        name = m.group(1)
        body = m.group(2)
        nf = re.search(r'nFaces\s+(\d+)', body)
        sf = re.search(r'startFace\s+(\d+)', body)
        if nf and sf:
            patches[name] = {"nFaces": int(nf.group(1)), "startFace": int(sf.group(1))}
    return patches


def read_face_centres(case_dir: Path, patch_name: str) -> np.ndarray:
    """Compute face centres for a specific patch from polyMesh."""
    poly = case_dir / "constant" / "polyMesh"

    # Points
    points_text = (poly / "points").read_text()
    match = re.search(r'^\s*(\d+)\s*\(', points_text, re.MULTILINE)
    n_pts = int(match.group(1))
    block = points_text[match.end():]
    coords = re.findall(r'\(\s*([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s*\)', block)
    points = np.array([[float(x), float(y), float(z)] for x, y, z in coords[:n_pts]])

    # Faces
    faces_text = (poly / "faces").read_text()
    match = re.search(r'^\s*(\d+)\s*\(', faces_text, re.MULTILINE)
    n_faces = int(match.group(1))
    block = faces_text[match.end():]
    face_entries = re.findall(r'\d+\(([^)]+)\)', block)
    faces = [[int(v) for v in entry.split()] for entry in face_entries[:n_faces]]

    # Patch info
    patches = read_boundary_info(case_dir)
    info = patches[patch_name]
    start, n = info["startFace"], info["nFaces"]

    # Compute face centres
    centres = np.zeros((n, 3))
    for i in range(n):
        verts = faces[start + i]
        centres[i] = points[verts].mean(axis=0)

    return centres


def sample_worldcover_at_faces(
    worldcover_tif: Path,
    face_centres: np.ndarray,
    site_lat: float,
    site_lon: float,
) -> np.ndarray:
    """Sample WorldCover class at each face centre (in local coords).

    Returns array of land cover class IDs (uint8).
    """
    import rasterio
    from pyproj import Transformer

    # Local coords (mesh) → UTM → WGS84
    # Mesh origin is at (site_lon, site_lat) in UTM zone 29N
    transformer_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32629", always_xy=True)
    x0_utm, y0_utm = transformer_to_utm.transform(site_lon, site_lat)

    # Face centres (local) → UTM
    face_x_utm = face_centres[:, 0] + x0_utm
    face_y_utm = face_centres[:, 1] + y0_utm

    # UTM → WGS84 for raster sampling
    transformer_to_wgs = Transformer.from_crs("EPSG:32629", "EPSG:4326", always_xy=True)
    face_lon, face_lat = transformer_to_wgs.transform(face_x_utm, face_y_utm)

    # Sample raster
    with rasterio.open(worldcover_tif) as src:
        # Convert lon/lat to raster row/col
        rows, cols = rasterio.transform.rowcol(src.transform, face_lon, face_lat)
        rows = np.clip(rows, 0, src.height - 1)
        cols = np.clip(cols, 0, src.width - 1)

        band = src.read(1)
        classes = band[rows, cols]

    return classes.astype(np.uint8)


def classes_to_z0(classes: np.ndarray) -> np.ndarray:
    """Convert WorldCover class IDs to z0 values."""
    z0 = np.full(len(classes), Z0_DEFAULT, dtype=np.float64)
    for cls, val in Z0_LOOKUP.items():
        z0[classes == cls] = val
    return z0


def write_z0_boundary_data(
    case_dir: Path,
    face_centres: np.ndarray,
    z0_values: np.ndarray,
    patch_name: str = "terrain",
) -> Path:
    """Write z0 as constant/boundaryData/<patch>/points + 0/z0."""
    bd_dir = case_dir / "constant" / "boundaryData" / patch_name

    # Points file
    pts_file = bd_dir / "points"
    pts_file.parent.mkdir(parents=True, exist_ok=True)
    n = len(face_centres)
    lines = [f'{n}', '(']
    for i in range(n):
        lines.append(f'({face_centres[i, 0]:.6f} {face_centres[i, 1]:.6f} {face_centres[i, 2]:.6f})')
    lines.append(')')
    pts_file.write_text('\n'.join(lines))

    # z0 field file
    z0_file = bd_dir / "0" / "z0"
    z0_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [f'{n}', '(']
    for i in range(n):
        lines.append(f'{z0_values[i]:.6e}')
    lines.append(')')
    z0_file.write_text('\n'.join(lines))

    logger.info("Wrote z0 boundaryData: %d faces, z0 range [%.4f, %.4f] m",
                n, z0_values.min(), z0_values.max())

    return bd_dir


def generate_z0_field(
    case_dir: Path,
    worldcover_tif: Path,
    site_lat: float,
    site_lon: float,
    patch_name: str = "terrain",
) -> np.ndarray:
    """Full pipeline: WorldCover → z0 boundaryData on terrain patch."""
    logger.info("Reading terrain face centres from %s", case_dir)
    face_centres = read_face_centres(case_dir, patch_name)
    logger.info("Terrain patch: %d faces, z range [%.1f, %.1f] m",
                len(face_centres), face_centres[:, 2].min(), face_centres[:, 2].max())

    logger.info("Sampling WorldCover at face centres...")
    classes = sample_worldcover_at_faces(worldcover_tif, face_centres, site_lat, site_lon)

    # Stats
    unique, counts = np.unique(classes, return_counts=True)
    for cls, cnt in zip(unique, counts):
        pct = 100 * cnt / len(classes)
        z0_val = Z0_LOOKUP.get(cls, Z0_DEFAULT)
        logger.info("  Class %3d: %5d faces (%5.1f%%) → z0 = %.4f m", cls, cnt, pct, z0_val)

    z0 = classes_to_z0(classes)
    write_z0_boundary_data(case_dir, face_centres, z0, patch_name)

    return z0


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser(description="Generate z0 field from WorldCover")
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--worldcover", required=True, type=Path)
    parser.add_argument("--site-lat", type=float, required=True)
    parser.add_argument("--site-lon", type=float, required=True)
    parser.add_argument("--patch", default="terrain")
    args = parser.parse_args()

    generate_z0_field(args.case_dir, args.worldcover, args.site_lat, args.site_lon, args.patch)


if __name__ == "__main__":
    main()
