"""
generate_mesh.py — SRTM DEM → OpenFOAM mesh (blockMesh + snappyHexMesh)

Architecture: nested domain with a refined central zone
-------------------------------------------------------
  context_cells=1 : single domain, 25×25 km (pipeline test, blockMesh only)
  context_cells=3 : 3×3 super-mailles, total 75×75 km (Perdigão + 25 km buffer)
  context_cells=5 : 5×5 super-mailles, total 125×125 km (larger context)

Only the central 25×25 km zone is refined to the target resolution.
Outer zones remain coarse (~super-maille width) to absorb boundary effects.

Usage
-----
    python generate_mesh.py \
        --site perdigao \
        --resolution-m 1000 \
        --context-cells 3 \
        --output data/cases/perdigao_1000m_3x3/

    python generate_mesh.py \
        --site perdigao \
        --resolution-m 1000 \
        --context-cells 1 \
        --output data/cases/perdigao_1000m_pipeline_test/

References
----------
  Neunaber et al. (WES 2023) — OpenFOAM ABL at Perdigão, resolution 12.5 m, ESI v2012
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import yaml
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CENTRAL_ZONE_KM    = 25.0       # km — width of the refined central zone
SUPER_MAILLE_KM    = CENTRAL_ZONE_KM
DOMAIN_HEIGHT_M    = 3000.0     # reduced from 7 km — sufficient for ABL
AR_TARGET          = 5.0        # target aspect ratio (dx/dz at ground)
STL_FILENAME       = "terrain.stl"

# snappyHexMesh defaults
N_SNAP_LAYERS      = 3
DEFAULT_REFINE_LEVELS = 2       # blockMesh is 2^N coarser than target; snappy refines near terrain

# ABL distance thresholds for snappyHexMesh distance refinement [m]
ABL_SURFACE_M      = 500.0     # top of surface layer — finest refinement
ABL_MIXING_M       = 1500.0    # top of mixing layer  — intermediate refinement


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def compute_domain_geometry(
    resolution_m: float,
    context_cells: int,
    n_z: int | None = None,
    n_refine_levels: int = DEFAULT_REFINE_LEVELS,
) -> dict:
    """Compute all domain dimensions and cell counts.

    The blockMesh creates a **coarse** base mesh; snappyHexMesh then refines
    near the terrain surface using distance-based refinement.  This gives
    fine horizontal resolution near the ground and coarse resolution aloft.

    Parameters
    ----------
    resolution_m:
        Target horizontal resolution in the central zone [m].
    context_cells:
        Number of super-mailles on each side (1 = central only, 3 = 3×3, 5 = 5×5).
    n_z:
        Number of vertical cells.  If None, computed adaptively.
    n_refine_levels:
        Number of snappyHexMesh refinement levels.  blockMesh base cell =
        resolution_m * 2^n_refine_levels.

    Returns
    -------
    dict with geometry parameters ready to be passed to Jinja2 templates.
    """
    central_m  = CENTRAL_ZONE_KM * 1000.0
    total_m    = context_cells * SUPER_MAILLE_KM * 1000.0

    # Base cell size for blockMesh (coarse); snappy refines to resolution_m
    base_cell_m = resolution_m * (2 ** n_refine_levels)

    # Cells in the central zone (at base resolution)
    n_central  = max(1, int(round(central_m / base_cell_m)))

    if context_cells == 1:
        n_x = n_central
        n_y = n_central
        grading_x = "1"
        grading_y = "1"
    else:
        # The outer zone (half the buffer on each side)
        outer_km   = (context_cells - 1) / 2.0 * SUPER_MAILLE_KM
        outer_m    = outer_km * 1000.0
        # Outer cells are 2× coarser than base (already coarse)
        n_outer_cells = max(3, int(round(outer_m / (base_cell_m * 2))))

        total_xy_cells = 2 * n_outer_cells + n_central
        n_x = total_xy_cells
        n_y = total_xy_cells

        coarse_cell_m = outer_m / n_outer_cells
        g_ratio       = coarse_cell_m / base_cell_m

        left_frac  = outer_m / total_m
        mid_frac   = central_m / total_m
        right_frac = left_frac

        grading_x = (
            f"( ({left_frac:.6f} {n_outer_cells} {g_ratio:.4f}) "
            f"  ({mid_frac:.6f} {n_central}     1           ) "
            f"  ({right_frac:.6f} {n_outer_cells} {1/g_ratio:.4f}) )"
        )
        grading_y = grading_x

    # Adaptive vertical sizing based on base cell
    # After snappy refinement near terrain, effective dz_ground = dz_base / 2^n_refine
    dz_ground_base = base_cell_m / AR_TARGET
    G_TARGET = AR_TARGET  # grading ratio: dz_top / dz_ground
    if n_z is None:
        # Linear approximation: H ≈ n * dz_ground * (1 + G) / 2
        n_z = max(10, int(round(2.0 * DOMAIN_HEIGHT_M / (dz_ground_base * (1.0 + G_TARGET)))))

    # Compute actual grading from H, n_z, dz_ground
    grading_z = max(1.0, 2.0 * DOMAIN_HEIGHT_M / (n_z * dz_ground_base) - 1.0)
    grading_z = min(grading_z, 20.0)  # clamp

    # Effective resolution near terrain after snappy refinement
    dz_ground_eff = dz_ground_base / (2 ** n_refine_levels)

    logger.info(
        "blockMesh: base_cell=%.0fm, %d×%d×%d cells | "
        "snappy: %d refine levels → %.0fm target near terrain (dz_eff=%.0fm)",
        base_cell_m, n_x, n_y, n_z,
        n_refine_levels, resolution_m, dz_ground_eff,
    )

    return {
        "total_x_m":        total_m,
        "total_y_m":        total_m,
        "total_z_m":        DOMAIN_HEIGHT_M,
        "n_x":              n_x,
        "n_y":              n_y,
        "n_z":              n_z,
        "grading_x":        grading_x,
        "grading_y":        grading_y,
        "grading_z":        f"{grading_z:.4f}",
        "central_m":        central_m,
        "resolution_m":     resolution_m,
        "base_cell_m":      base_cell_m,
        "n_refine_levels":  n_refine_levels,
        "context_cells":    context_cells,
    }


def compute_refine_distances(n_refine_levels: int) -> list[tuple[float, int]]:
    """Compute snappyHexMesh distance-based refinement levels.

    Returns a list of (distance_m, level) tuples for the terrain surface,
    ordered from innermost (highest level) to outermost (lowest level).
    """
    if n_refine_levels <= 0:
        return []
    if n_refine_levels == 1:
        return [(ABL_MIXING_M, 1)]
    if n_refine_levels == 2:
        return [(ABL_SURFACE_M, 2), (ABL_MIXING_M, 1)]
    # n_refine_levels >= 3: add an inner shell at 200m
    distances = [(200.0, n_refine_levels)]
    distances.append((ABL_SURFACE_M, n_refine_levels - 1))
    distances.append((ABL_MIXING_M, max(1, n_refine_levels - 2)))
    return distances


# ---------------------------------------------------------------------------
# DEM → terrain STL
# ---------------------------------------------------------------------------

def dem_to_stl(
    srtm_tif: Path,
    out_stl: Path,
    bounds_lonlat: tuple[float, float, float, float],
    resolution_m: float,
    site_lat: float,
    site_lon: float,
) -> None:
    """Resample SRTM DEM to CFD resolution and export as STL.

    Parameters
    ----------
    srtm_tif:
        Path to SRTM GeoTIFF (30 m, full resolution).
    out_stl:
        Output STL file path.
    bounds_lonlat:
        (west, south, east, north) bounding box [°].
    resolution_m:
        Target resolution for resampling [m].
    site_lat, site_lon:
        Site centre coordinates [°] — used to project lon/lat to local x/y [m].
    """
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.transform import from_bounds
        from rasterio.warp import reproject
    except ImportError as exc:
        raise ImportError("rasterio is required for DEM processing") from exc

    try:
        from stl import mesh as stl_mesh  # numpy-stl
    except ImportError as exc:
        raise ImportError("numpy-stl is required: pip install numpy-stl") from exc

    west, south, east, north = bounds_lonlat
    DEG_PER_M_LAT = 1.0 / 111_000.0
    DEG_PER_M_LON = 1.0 / (111_000.0 * np.cos(np.radians(site_lat)))

    width_deg  = east - west
    height_deg = north - south
    width_m    = width_deg / DEG_PER_M_LON
    height_m   = height_deg / DEG_PER_M_LAT

    n_col = max(2, int(round(width_m / resolution_m)))
    n_row = max(2, int(round(height_m / resolution_m)))

    logger.info(
        "Resampling DEM to %.0f m → %d×%d cells over %.1f×%.1f km",
        resolution_m, n_col, n_row, width_m / 1000, height_m / 1000,
    )

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

    # Handle nodata
    elevation = np.where(np.isfinite(elevation), elevation, np.nanmean(elevation))

    # Build x, y grids in local metres (origin = site centre)
    lons = np.linspace(west, east, n_col)
    lats = np.linspace(south, north, n_row)
    x_m  = (lons - site_lon) / DEG_PER_M_LON    # east positive
    y_m  = (lats - site_lat) / DEG_PER_M_LAT    # north positive

    X, Y = np.meshgrid(x_m, y_m)
    Z = elevation

    # Triangulate: each quad → 2 triangles
    n_tri = 2 * (n_row - 1) * (n_col - 1)
    terrain = stl_mesh.Mesh(np.zeros(n_tri, dtype=stl_mesh.Mesh.dtype))

    idx = 0
    for i in range(n_row - 1):
        for j in range(n_col - 1):
            # Lower-left triangle
            terrain.vectors[idx][0] = [X[i,   j],   Y[i,   j],   Z[i,   j]]
            terrain.vectors[idx][1] = [X[i+1, j],   Y[i+1, j],   Z[i+1, j]]
            terrain.vectors[idx][2] = [X[i+1, j+1], Y[i+1, j+1], Z[i+1, j+1]]
            idx += 1
            # Upper-right triangle
            terrain.vectors[idx][0] = [X[i,   j],   Y[i,   j],   Z[i,   j]]
            terrain.vectors[idx][1] = [X[i+1, j+1], Y[i+1, j+1], Z[i+1, j+1]]
            terrain.vectors[idx][2] = [X[i,   j+1], Y[i,   j+1], Z[i,   j+1]]
            idx += 1

    out_stl.parent.mkdir(parents=True, exist_ok=True)
    terrain.save(str(out_stl))
    logger.info("Terrain STL saved: %s (%d triangles)", out_stl, n_tri)


# ---------------------------------------------------------------------------
# Jinja2 template rendering
# ---------------------------------------------------------------------------

def render_templates(
    template_dir: Path,
    output_dir: Path,
    context: dict,
) -> None:
    """Render all .j2 templates in template_dir → output_dir.

    Parameters
    ----------
    template_dir:
        Directory containing Jinja2 templates (.j2 files), mirroring the
        OpenFOAM case directory structure (system/, constant/, 0/).
    output_dir:
        OpenFOAM case root directory to write rendered files into.
    context:
        Jinja2 rendering context (variables passed to templates).
    """
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        keep_trailing_newline=True,
    )

    for tmpl_path in sorted(template_dir.rglob("*.j2")):
        rel = tmpl_path.relative_to(template_dir)
        # Strip .j2 suffix
        out_rel = rel.with_suffix("")
        out_file = output_dir / out_rel

        tmpl = env.get_template(str(rel))
        rendered = tmpl.render(**context)

        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(rendered)
        logger.debug("Rendered: %s → %s", rel, out_rel)

    # Copy static files (no .j2 extension)
    for static_path in sorted(template_dir.rglob("*")):
        if static_path.suffix == ".j2" or static_path.is_dir():
            continue
        rel = static_path.relative_to(template_dir)
        out_file = output_dir / rel
        out_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(static_path, out_file)
        logger.debug("Copied static: %s", rel)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_mesh(
    site_cfg: dict,
    resolution_m: float,
    context_cells: int,
    output_dir: Path,
    srtm_tif: Path | None = None,
    template_dir: Path | None = None,
    inflow_json: Path | None = None,
    n_z: int | None = None,
    n_refine_levels: int = DEFAULT_REFINE_LEVELS,
) -> dict:
    """Generate an OpenFOAM case directory with mesh and boundary condition files.

    Parameters
    ----------
    site_cfg:
        Parsed perdigao.yaml content.
    resolution_m:
        Target horizontal resolution in central zone [m].
    context_cells:
        1, 3, or 5 (number of super-mailles on each side of central zone).
    output_dir:
        Where to write the OpenFOAM case.
    srtm_tif:
        SRTM GeoTIFF (30 m).  If None, flat terrain is used.
    template_dir:
        Jinja2 template root.  Defaults to
        services/module2a-cfd/templates/openfoam/.
    inflow_json:
        Optional path to inflow profile JSON (from prepare_inflow.py).
        Provides u_hub, u_star, z0_eff, flowDir_x, flowDir_y for BC templates.
    n_z:
        Number of vertical layers.
    n_refine_levels:
        Number of snappyHexMesh refinement levels (default 2).

    Returns
    -------
    dict with geometry info (useful for logging / convergence study).
    """
    if template_dir is None:
        template_dir = Path(__file__).parent / "templates" / "openfoam"

    site = site_cfg["site"]
    site_lat = site["coordinates"]["latitude"]
    site_lon = site["coordinates"]["longitude"]

    # ---- geometry -------------------------------------------------------
    geom = compute_domain_geometry(resolution_m, context_cells, n_z, n_refine_levels)
    logger.info(
        "Domain: %.0f×%.0f km, base=%.0fm → target=%.0fm, context=%d×%d, "
        "blockMesh=%d×%d×%d cells",
        geom["total_x_m"] / 1000, geom["total_y_m"] / 1000,
        geom["base_cell_m"], resolution_m,
        context_cells, context_cells,
        geom["n_x"], geom["n_y"], geom["n_z"],
    )

    # ---- STL terrain -------------------------------------------------------
    output_dir = Path(output_dir)
    trisurf_dir = output_dir / "constant" / "triSurface"
    stl_path    = trisurf_dir / STL_FILENAME

    half_m = geom["central_m"] / 2.0
    DEG_LAT = 1.0 / 111_000.0
    DEG_LON = 1.0 / (111_000.0 * np.cos(np.radians(site_lat)))

    if srtm_tif is not None and Path(srtm_tif).exists():
        bounds = (
            site_lon - half_m * DEG_LON,
            site_lat - half_m * DEG_LAT,
            site_lon + half_m * DEG_LON,
            site_lat + half_m * DEG_LAT,
        )
        dem_to_stl(
            srtm_tif=Path(srtm_tif),
            out_stl=stl_path,
            bounds_lonlat=bounds,
            resolution_m=resolution_m,
            site_lat=site_lat,
            site_lon=site_lon,
        )
    else:
        logger.warning(
            "No SRTM raster provided or found — generating flat terrain STL"
        )
        _write_flat_terrain_stl(stl_path, geom["total_x_m"], geom["total_y_m"])

    # ---- inflow params -------------------------------------------------------
    inflow = {
        "u_hub":     10.0,
        "u_star":    0.5,
        "z0_eff":    0.05,
        "z0":        0.05,
        "z_ref":     80.0,
        "kappa":     0.41,
        "flowDir_x": 1.0,
        "flowDir_y": 0.0,
        "wind_dir":  270.0,
        "T_ref":     288.15,
        "L_mo":      None,
    }
    if inflow_json is not None and Path(inflow_json).exists():
        import json
        with open(inflow_json) as f:
            inflow.update(json.load(f))

    # ---- Inlet/outlet face assignment based on wind direction -----------------
    import math as _math

    wind_dir = float(inflow.get("wind_dir", 270.0))  # degrees FROM North
    wind_rad = _math.radians(wind_dir)
    # Met convention: wind_dir = direction wind comes FROM
    # sin(wind_dir) > 0 → wind from E sector → east face = inlet
    # sin(wind_dir) < 0 → wind from W sector → west face = inlet
    # cos(wind_dir) > 0 → wind from N sector → north face = inlet
    # cos(wind_dir) < 0 → wind from S sector → south face = inlet
    inlet_faces = []
    outlet_faces = []
    if _math.sin(wind_rad) > 0.1:
        inlet_faces.append("east");   outlet_faces.append("west")
    elif _math.sin(wind_rad) < -0.1:
        inlet_faces.append("west");   outlet_faces.append("east")
    else:
        inlet_faces += ["west", "east"]

    if _math.cos(wind_rad) > 0.1:
        inlet_faces.append("north");  outlet_faces.append("south")
    elif _math.cos(wind_rad) < -0.1:
        inlet_faces.append("south");  outlet_faces.append("north")
    else:
        inlet_faces += ["south", "north"]

    logger.info(
        "Wind dir %.1f° → inlet faces: %s, outlet faces: %s",
        wind_dir, inlet_faces, outlet_faces,
    )

    # ---- Jinja2 context -------------------------------------------------------
    n_cores = 8  # default parallel decomposition
    central_m = geom["central_m"]
    half_x = geom["total_x_m"] / 2
    half_y = geom["total_y_m"] / 2

    # snappyHexMesh: distance-based refinement near terrain
    refine_level = geom["n_refine_levels"]
    refine_distances = compute_refine_distances(refine_level)

    logger.info(
        "snappy refinement: level=%d, distances=%s",
        refine_level, refine_distances,
    )

    jinja_ctx = {
        "domain": {
            "total_x_m":   geom["total_x_m"],
            "total_y_m":   geom["total_y_m"],
            "total_z_m":   geom["total_z_m"],
            "n_x":         geom["n_x"],
            "n_y":         geom["n_y"],
            "n_z":         geom["n_z"],
            "grading_x":   geom["grading_x"],
            "grading_y":   geom["grading_y"],
            "grading_z":   geom["grading_z"],
            "half_x":      half_x,
            "half_y":      half_y,
            "x_min":       0.0,
            "x_max":       geom["total_x_m"],
            "y_min":       0.0,
            "y_max":       geom["total_y_m"],
            "z_min":       0.0,
            "z_max":       geom["total_z_m"],
        },
        "mesh": {
            "nx":        geom["n_x"],
            "ny":        geom["n_y"],
            "nz":        geom["n_z"],
            "x_grading": geom["grading_x"],
            "y_grading": geom["grading_y"],
            "z_grading": geom["grading_z"],
        },
        "terrain": {
            "stl_file":       STL_FILENAME,
            "central_m":      central_m,
            "resolution_m":   resolution_m,
            "n_snap_layers":  N_SNAP_LAYERS,
        },
        "refine_level":     refine_level,
        "refine_distances": refine_distances,
        "n_layers":         N_SNAP_LAYERS,
        "inflow": inflow,
        "inlet_faces":  inlet_faces,
        "outlet_faces": outlet_faces,
        "physics": {
            "T_ref_K":  float(inflow.get("T_ref", 300.0)),
            "p_ref_Pa": 0.0,
            "rho_ref":  1.225,
        },
        "solver": {
            "n_iter":         500,
            "write_interval": 100,
            "n_cores":        n_cores,
        },
        "parallel": {
            "n_cores": n_cores,
        },
    }

    # ---- render templates -------------------------------------------------------
    logger.info("Rendering OpenFOAM templates → %s", output_dir)
    render_templates(template_dir, output_dir, jinja_ctx)

    logger.info("Case directory ready: %s", output_dir)
    return geom


def _write_flat_terrain_stl(out_stl: Path, total_x: float, total_y: float) -> None:
    """Write a minimal flat terrain STL (2 triangles) for pipeline testing."""
    try:
        from stl import mesh as stl_mesh
    except ImportError:
        # Fallback: write ASCII STL manually
        out_stl.parent.mkdir(parents=True, exist_ok=True)
        hx, hy = total_x / 2, total_y / 2
        content = (
            "solid terrain\n"
            f" facet normal 0 0 1\n"
            f"  outer loop\n"
            f"   vertex {-hx} {-hy} 0\n"
            f"   vertex  {hx} {-hy} 0\n"
            f"   vertex  {hx}  {hy} 0\n"
            f"  endloop\n"
            f" endfacet\n"
            f" facet normal 0 0 1\n"
            f"  outer loop\n"
            f"   vertex {-hx} {-hy} 0\n"
            f"   vertex  {hx}  {hy} 0\n"
            f"   vertex {-hx}  {hy} 0\n"
            f"  endloop\n"
            f" endfacet\n"
            "endsolid terrain\n"
        )
        out_stl.write_text(content)
        logger.info("Flat terrain STL (ASCII) written: %s", out_stl)
        return

    hx, hy = total_x / 2, total_y / 2
    terrain = stl_mesh.Mesh(np.zeros(2, dtype=stl_mesh.Mesh.dtype))
    terrain.vectors[0] = [[-hx, -hy, 0], [hx, -hy, 0], [hx, hy, 0]]
    terrain.vectors[1] = [[-hx, -hy, 0], [hx,  hy, 0], [-hx, hy, 0]]
    out_stl.parent.mkdir(parents=True, exist_ok=True)
    terrain.save(str(out_stl))
    logger.info("Flat terrain STL written: %s", out_stl)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate OpenFOAM case directory from SRTM DEM"
    )
    parser.add_argument("--site",           default="perdigao",
                        help="Site name (configs/sites/<site>.yaml)")
    parser.add_argument("--resolution-m",   type=float, required=True,
                        help="Target horizontal resolution [m] in central zone")
    parser.add_argument("--context-cells",  type=int, default=3, choices=[1, 3, 5],
                        help="Super-maille context: 1=pipeline test, 3=75km, 5=125km")
    parser.add_argument("--n-z",            type=int, default=None,
                        help="Number of vertical layers (auto if not set)")
    parser.add_argument("--n-refine-levels", type=int, default=DEFAULT_REFINE_LEVELS,
                        help=f"snappyHexMesh refinement levels (default {DEFAULT_REFINE_LEVELS})")
    parser.add_argument("--srtm",           default=None,
                        help="SRTM GeoTIFF path (default: data/raw/srtm_<site>_30m.tif)")
    parser.add_argument("--inflow-json",    default=None,
                        help="Inflow profile JSON (from prepare_inflow.py)")
    parser.add_argument("--output",         required=True,
                        help="Output OpenFOAM case directory")
    args = parser.parse_args()

    # Load site config
    cfg_path = (
        Path(__file__).parents[2]
        / "configs" / "sites" / f"{args.site}.yaml"
    )
    with open(cfg_path) as f:
        site_cfg = yaml.safe_load(f)

    # Default SRTM path
    srtm_tif = args.srtm
    if srtm_tif is None:
        srtm_tif = (
            Path(__file__).parents[2]
            / "data" / "raw" / f"srtm_{args.site}_30m.tif"
        )
        if not Path(srtm_tif).exists():
            logger.warning("SRTM not found at %s — will use flat terrain", srtm_tif)
            srtm_tif = None

    geom = generate_mesh(
        site_cfg=site_cfg,
        resolution_m=args.resolution_m,
        context_cells=args.context_cells,
        output_dir=Path(args.output),
        srtm_tif=srtm_tif,
        inflow_json=args.inflow_json,
        n_z=args.n_z,
        n_refine_levels=args.n_refine_levels,
    )

    print(f"Case generated: {args.output}")
    print(f"  target res  : {args.resolution_m:.0f} m")
    print(f"  base cell   : {geom['base_cell_m']:.0f} m (blockMesh)")
    print(f"  refine      : {geom['n_refine_levels']} levels (snappyHexMesh distance)")
    print(f"  context     : {args.context_cells}×{args.context_cells} super-mailles")
    print(f"  domain      : {geom['total_x_m']/1000:.0f}×{geom['total_y_m']/1000:.0f}×{geom['total_z_m']/1000:.0f} km")
    print(f"  blockMesh   : {geom['n_x']}×{geom['n_y']}×{geom['n_z']} = {geom['n_x']*geom['n_y']*geom['n_z']:,} cells")
