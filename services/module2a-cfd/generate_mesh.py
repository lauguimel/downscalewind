"""
generate_mesh.py — SRTM DEM → OpenFOAM mesh (cfMesh cartesianMesh)

Architecture: nested domain with a refined central zone
-------------------------------------------------------
  context_cells=1 : single domain, 25×25 km (pipeline test)
  context_cells=3 : 3×3 super-mailles, total 75×75 km (Perdigão + 25 km buffer)
  context_cells=5 : 5×5 super-mailles, total 125×125 km (larger context)

Only the central 25×25 km zone is refined to the target resolution.
Outer zones remain coarse (2-4× resolution) to absorb boundary effects.

Meshing: cfMesh cartesianMesh (octree, 2:1 transitions guaranteed).
Replaces the previous blockMesh + snappyHexMesh pipeline.

Usage
-----
    python generate_mesh.py \
        --site perdigao \
        --resolution-m 1000 \
        --context-cells 3 \
        --output data/cases/perdigao_1000m_3x3/

    python generate_mesh.py \
        --site perdigao \
        --resolution-m 500 \
        --context-cells 1 \
        --domain-km 10 \
        --output data/cases/perdigao_500m_test/

References
----------
  Neunaber et al. (WES 2023) — OpenFOAM ABL at Perdigão, resolution 12.5 m, OF2412 ESI
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
DEFAULT_DOMAIN_KM  = 25.0       # km — default width of the refined central zone
DOMAIN_HEIGHT_M    = 3000.0     # m above highest terrain point
STL_FILENAME       = "terrain.stl"

# cfMesh boundary layer defaults
N_BOUNDARY_LAYERS      = 5
BL_EXPANSION_RATIO     = 1.2
BL_FIRST_LAYER_M       = 10.0   # max first layer thickness [m]

# cfMesh terrain surface refinement
TERRAIN_REFINE_LEVELS  = 2       # additional octree levels on terrain surface
TERRAIN_REFINE_THICKNESS_M = 500.0  # distance from terrain to refine [m]

# Near-terrain refinement zone height [m]
NEAR_TERRAIN_HEIGHT_M  = 1000.0  # fine box covering surface layer


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def compute_domain_geometry(
    resolution_m: float,
    context_cells: int,
    domain_km: float = DEFAULT_DOMAIN_KM,
) -> dict:
    """Compute domain dimensions and cfMesh cell sizes.

    cfMesh uses maxCellSize (outer zone) + objectRefinements (boxes with
    explicit cellSize) to control resolution.  No blockMesh or grading needed.

    Parameters
    ----------
    resolution_m:
        Target horizontal resolution in the central zone [m].
    context_cells:
        Number of super-mailles on each side (1 = central only, 3 = 3×3, 5 = 5×5).
    domain_km:
        Width of the central (refined) zone in km (default 25).

    Returns
    -------
    dict with geometry parameters for Jinja2 templates.
    """
    central_m = domain_km * 1000.0
    total_m   = context_cells * domain_km * 1000.0

    # cfMesh cell sizes
    if context_cells == 1:
        # Single zone: maxCellSize = target resolution
        max_cell_size = resolution_m
    else:
        # Outer zone is coarser: 2-4× resolution depending on context
        max_cell_size = resolution_m * min(4.0, context_cells)

    target_cell_size = resolution_m
    # Near-terrain: half the target resolution for better terrain conformity
    fine_cell_size = resolution_m / 2.0

    logger.info(
        "cfMesh: max_cell=%.0fm, target=%.0fm, fine=%.0fm | "
        "domain=%.0f×%.0f km, context=%d×%d",
        max_cell_size, target_cell_size, fine_cell_size,
        total_m / 1000, total_m / 1000,
        context_cells, context_cells,
    )

    return {
        "total_x_m":         total_m,
        "total_y_m":         total_m,
        "total_z_m":         DOMAIN_HEIGHT_M,
        "central_m":         central_m,
        "resolution_m":      resolution_m,
        "max_cell_size":     max_cell_size,
        "target_cell_size":  target_cell_size,
        "fine_cell_size":    fine_cell_size,
        "context_cells":     context_cells,
    }


def compute_cfmesh_refinements(
    geom: dict,
    terrain_z_max: float,
) -> list[dict]:
    """Compute objectRefinement boxes for cfMesh.

    Creates two refinement zones:
    1. centralZone: covers the central domain at target resolution
    2. nearTerrain: covers the lowest ~1000m at half target resolution

    For context_cells=1, only nearTerrain is needed (whole domain is already
    at target resolution via maxCellSize).

    Parameters
    ----------
    geom:
        Output of compute_domain_geometry().
    terrain_z_max:
        Maximum terrain elevation [m].

    Returns
    -------
    List of objectRefinement dicts for the meshDict.j2 template.
    """
    refinements = []
    central_m = geom["central_m"]
    domain_z_max = geom["total_z_m"]

    # Central zone box: target resolution across the central domain
    if geom["context_cells"] > 1:
        refinements.append({
            "name": "centralZone",
            "cell_size": geom["target_cell_size"],
            "cx": 0.0,
            "cy": 0.0,
            "cz": domain_z_max / 2.0,
            "lx": central_m,
            "ly": central_m,
            "lz": domain_z_max,
        })

    # Near-terrain box: fine resolution in the surface layer
    near_terrain_top = min(
        terrain_z_max + NEAR_TERRAIN_HEIGHT_M,
        domain_z_max,
    )
    near_terrain_cz = near_terrain_top / 2.0
    refinements.append({
        "name": "nearTerrain",
        "cell_size": geom["fine_cell_size"],
        "cx": 0.0,
        "cy": 0.0,
        "cz": near_terrain_cz,
        "lx": central_m,
        "ly": central_m,
        "lz": near_terrain_top,
    })

    return refinements


def compute_octagonal_refinements(
    domain_km: float,
    domain_z_max: float,
    fine_cell_size: float = 30,
) -> list[dict]:
    """Compute 3-ring objectRefinement zones for octagonal/cylindrical domains.

    Ring structure (progressive refinement from maxCellSize):
      mesoZone   : 500 m cells, ~57% of domain diameter (transition zone)
      fineZone   : 200 m cells, 2.4×2.4 km (GNN output cube + margin)
      nearTerrain: fine_cell_size cells, 2.4×2.4×0.5 km (surface layer)

    The ratio between adjacent zones is kept ≤ 2.5:1 for smooth cfMesh transitions.

    Parameters
    ----------
    domain_km:
        Domain diameter [km].
    domain_z_max:
        Domain height [m].
    fine_cell_size:
        Finest cell size [m] (default 100, use 50 for production).

    Returns
    -------
    List of objectRefinement dicts for meshDict.j2 template.
    """
    diameter_m = domain_km * 1000.0

    # mesoZone: 500m cells, covers ~57% of domain diameter (capped at 80%)
    meso_extent = min(0.57 * diameter_m, 0.8 * diameter_m)
    # Fine/near-terrain: fixed 2.4 km (GNN output cube 1×1 km + 0.7 km margin)
    fine_extent = min(2400.0, 0.5 * diameter_m)

    return [
        {
            "name": "mesoZone",
            "cell_size": 500,
            "cx": 0.0, "cy": 0.0, "cz": domain_z_max / 2.0,
            "lx": meso_extent, "ly": meso_extent, "lz": domain_z_max,
        },
        {
            "name": "fineZone",
            "cell_size": 200,
            "cx": 0.0, "cy": 0.0, "cz": 500.0,
            "lx": fine_extent, "ly": fine_extent, "lz": 1000.0,
        },
        {
            "name": "nearTerrain",
            "cell_size": fine_cell_size,
            "cx": 0.0, "cy": 0.0, "cz": 250.0,
            "lx": fine_extent, "ly": fine_extent, "lz": 500.0,
        },
    ]


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
    level_terrain: bool = False,
    domain_radius_m: float | None = None,
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

    # Optional terrain leveling: taper Z → 0 at domain boundary
    if level_terrain and domain_radius_m is not None:
        Z = _level_terrain(Z, X, Y, 0.0, 0.0, domain_radius_m)
        logger.info(
            "Terrain leveling applied: tanh blend over radius %.0f m", domain_radius_m
        )

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

    return {"z_min": float(Z.min()), "z_max": float(Z.max())}


def _level_terrain(
    Z: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    center_x: float,
    center_y: float,
    domain_radius_m: float,
) -> np.ndarray:
    """Apply tanh terrain leveling: smoothly reduce terrain height to 0 at boundary.

    Parameters
    ----------
    Z : terrain elevation array
    X, Y : coordinate arrays (same shape as Z)
    center_x, center_y : domain centre in projected coords [m]
    domain_radius_m : domain radius — terrain → 0 at this distance

    Returns
    -------
    Z_leveled : same shape as Z
    """
    r = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    blend = np.tanh((1.6 * r / domain_radius_m) ** 8)
    return Z * (1.0 - blend)


def make_octagon_stl(
    center_x: float,
    center_y: float,
    radius_m: float,
    height_m: float,
    n_sides: int = 8,
) -> str:
    """Generate a regular polygon prism STL in ASCII multi-solid format.

    Parameters
    ----------
    center_x, center_y : domain centre in projected coordinates [m]
    radius_m : circumradius of the polygon [m]
    height_m : prism height [m]
    n_sides : number of sides (default 8 → octagon)

    Returns
    -------
    ASCII STL string with two solids: 'lateral' and 'top'.
    """
    angles = [2.0 * np.pi * i / n_sides for i in range(n_sides)]
    # Corner vertices at bottom and top
    pts_bot = np.array(
        [[center_x + radius_m * np.cos(a), center_y + radius_m * np.sin(a), 0.0]
         for a in angles]
    )
    pts_top = np.array(
        [[center_x + radius_m * np.cos(a), center_y + radius_m * np.sin(a), height_m]
         for a in angles]
    )
    top_center = np.array([center_x, center_y, height_m])

    lines: list[str] = []

    def _write_facet(out: list[str], normal: np.ndarray, v0, v1, v2) -> None:
        n = normal / (np.linalg.norm(normal) + 1e-300)
        out.append(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}")
        out.append("    outer loop")
        for v in (v0, v1, v2):
            out.append(f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}")
        out.append("    endloop")
        out.append("  endfacet")

    # ---- lateral panels ----
    lines.append("solid lateral")
    for i in range(n_sides):
        bl = pts_bot[i]
        br = pts_bot[(i + 1) % n_sides]
        tl = pts_top[i]
        tr = pts_top[(i + 1) % n_sides]
        # Outward normal: midpoint of panel projected radially
        mid_angle = 2.0 * np.pi * (i + 0.5) / n_sides
        normal = np.array([np.cos(mid_angle), np.sin(mid_angle), 0.0])
        # Triangle 1: BL, BR, TR
        _write_facet(lines, normal, bl, br, tr)
        # Triangle 2: BL, TR, TL
        _write_facet(lines, normal, bl, tr, tl)
    lines.append("endsolid lateral")

    # ---- top cap ----
    lines.append("solid top")
    top_normal = np.array([0.0, 0.0, 1.0])
    for i in range(n_sides):
        tl = pts_top[i]
        tr = pts_top[(i + 1) % n_sides]
        _write_facet(lines, top_normal, top_center, tl, tr)
    lines.append("endsolid top")

    return "\n".join(lines) + "\n"


def build_octagon_domain_stl(
    terrain_stl: Path,
    octagon_stl_content: str,
    out_stl: Path,
) -> None:
    """Build a closed STL surface for cfMesh from terrain + octagon prism.

    Combines into a single ASCII multi-solid file:
    - terrain STL (bottom surface, solid "terrain")
    - octagon prism (lateral walls + top cap, solids "lateral" and "top")

    cfMesh requires a closed surfaceFile to determine inside/outside via
    ray-casting.  Without the terrain as the bottom face, the octagon is open
    and cartesianMesh produces zero cells.
    """
    try:
        from stl import mesh as stl_mesh
    except ImportError as exc:
        raise ImportError("numpy-stl is required: pip install numpy-stl") from exc

    terrain = stl_mesh.Mesh.from_file(str(terrain_stl))

    out_stl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_stl, "w") as f:
        # Terrain solid (bottom)
        f.write("solid terrain\n")
        for tri in terrain.vectors:
            v0, v1, v2 = tri
            e1 = v1 - v0
            e2 = v2 - v0
            n = np.cross(e1, e2)
            norm = np.linalg.norm(n)
            if norm > 0:
                n = n / norm
            f.write(f"  facet normal {n[0]} {n[1]} {n[2]}\n")
            f.write("    outer loop\n")
            for v in tri:
                f.write(f"      vertex {v[0]} {v[1]} {v[2]}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write("endsolid terrain\n")

        # Octagon solids (lateral + top) — already ASCII
        f.write(octagon_stl_content)

    logger.info(
        "Combined octagon domain STL: %s (%d terrain tri + octagon lateral+top)",
        out_stl, len(terrain.vectors),
    )


def build_domain_fms(
    terrain_stl: Path,
    out_fms: Path,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    z_min: float, z_max: float,
) -> None:
    """Build a closed FMS surface for cfMesh from terrain STL + bounding box.

    Creates a closed surface by combining:
    - The terrain STL (bottom surface, patch "terrain")
    - 5 planar faces: xMin, xMax, yMin, yMax, zMax

    Uses ASCII STL multi-solid format which cfMesh reads natively.
    Each solid name becomes a patch in the mesh.

    This replaces surfaceGenerateBoundingBox which has issues with negative
    coordinates being parsed as flags by the OF argument parser.
    """
    try:
        from stl import mesh as stl_mesh
    except ImportError as exc:
        raise ImportError("numpy-stl is required: pip install numpy-stl") from exc

    # Read terrain STL
    terrain = stl_mesh.Mesh.from_file(str(terrain_stl))

    # Build 5 bounding box faces (2 triangles each)
    # Normals point INWARD (towards domain interior) for cfMesh convention
    box_faces = {
        "xMin": [  # x = x_min, normal +x
            [[x_min, y_min, z_min], [x_min, y_min, z_max], [x_min, y_max, z_max]],
            [[x_min, y_min, z_min], [x_min, y_max, z_max], [x_min, y_max, z_min]],
        ],
        "xMax": [  # x = x_max, normal -x
            [[x_max, y_min, z_min], [x_max, y_max, z_min], [x_max, y_max, z_max]],
            [[x_max, y_min, z_min], [x_max, y_max, z_max], [x_max, y_min, z_max]],
        ],
        "yMin": [  # y = y_min, normal +y
            [[x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_min, z_max]],
            [[x_min, y_min, z_min], [x_max, y_min, z_max], [x_min, y_min, z_max]],
        ],
        "yMax": [  # y = y_max, normal -y
            [[x_min, y_max, z_min], [x_min, y_max, z_max], [x_max, y_max, z_max]],
            [[x_min, y_max, z_min], [x_max, y_max, z_max], [x_max, y_max, z_min]],
        ],
        "zMax": [  # z = z_max, normal -z
            [[x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max]],
            [[x_min, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max]],
        ],
    }

    # Write combined ASCII STL with named solids
    out_fms.parent.mkdir(parents=True, exist_ok=True)
    n_box_tri = 0
    with open(out_fms, "w") as f:
        # Terrain solid
        f.write("solid terrain\n")
        for tri in terrain.vectors:
            # Compute normal
            v0, v1, v2 = tri
            e1 = v1 - v0
            e2 = v2 - v0
            n = np.cross(e1, e2)
            norm = np.linalg.norm(n)
            if norm > 0:
                n = n / norm
            f.write(f"  facet normal {n[0]} {n[1]} {n[2]}\n")
            f.write("    outer loop\n")
            for v in tri:
                f.write(f"      vertex {v[0]} {v[1]} {v[2]}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write("endsolid terrain\n")

        # Box face solids
        for solid_name, tris in box_faces.items():
            f.write(f"solid {solid_name}\n")
            for tri in tris:
                v0, v1, v2 = np.array(tri[0]), np.array(tri[1]), np.array(tri[2])
                e1 = v1 - v0
                e2 = v2 - v0
                n = np.cross(e1, e2)
                norm = np.linalg.norm(n)
                if norm > 0:
                    n = n / norm
                f.write(f"  facet normal {n[0]} {n[1]} {n[2]}\n")
                f.write("    outer loop\n")
                for v in tri:
                    f.write(f"      vertex {v[0]} {v[1]} {v[2]}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")
                n_box_tri += 1
            f.write(f"endsolid {solid_name}\n")

    n_total = len(terrain.vectors) + n_box_tri
    logger.info(
        "Domain FMS saved: %s (%d terrain + %d box = %d triangles, 6 patches)",
        out_fms, len(terrain.vectors), n_box_tri, n_total,
    )


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
        # Preserve execute permission for shell scripts (Allrun, Allcontinue, etc.)
        if tmpl_path.stat().st_mode & 0o111:
            out_file.chmod(out_file.stat().st_mode | 0o111)
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
    domain_km: float = DEFAULT_DOMAIN_KM,
    solver_name: str = "simpleFoam",
    thermal: bool = False,
    coriolis: bool = True,
    canopy_enabled: bool = False,
    domain_type: str = "box",
    **kwargs,
) -> dict:
    """Generate an OpenFOAM case directory with cfMesh mesh and BC files.

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
    domain_km:
        Width of the central (refined) zone in km (default 25).

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
    geom = compute_domain_geometry(resolution_m, context_cells, domain_km)
    logger.info(
        "Domain: %.0f×%.0f km, target=%.0fm, context=%d×%d",
        geom["total_x_m"] / 1000, geom["total_y_m"] / 1000,
        resolution_m,
        context_cells, context_cells,
    )

    # ---- STL terrain -------------------------------------------------------
    output_dir = Path(output_dir)
    trisurf_dir = output_dir / "constant" / "triSurface"
    stl_path    = trisurf_dir / STL_FILENAME

    half_m = geom["central_m"] / 2.0
    DEG_LAT = 1.0 / 111_000.0
    DEG_LON = 1.0 / (111_000.0 * np.cos(np.radians(site_lat)))

    is_cylinder = domain_type == "cylinder"
    octagon_radius_m = domain_km * 1000.0 / 2.0 if is_cylinder else None

    terrain_z_max = 0.0
    if srtm_tif is not None and Path(srtm_tif).exists():
        bounds = (
            site_lon - half_m * DEG_LON,
            site_lat - half_m * DEG_LAT,
            site_lon + half_m * DEG_LON,
            site_lat + half_m * DEG_LAT,
        )
        # For cylindrical domains, STL must match finest refinement,
        # not maxCellSize which controls the coarse outer ring.
        stl_resolution = min(30, resolution_m) if is_cylinder else resolution_m
        terrain_stats = dem_to_stl(
            srtm_tif=Path(srtm_tif),
            out_stl=stl_path,
            bounds_lonlat=bounds,
            resolution_m=stl_resolution,
            site_lat=site_lat,
            site_lon=site_lon,
            level_terrain=is_cylinder,
            domain_radius_m=octagon_radius_m,
        )
        terrain_z_max = terrain_stats["z_max"]
    else:
        logger.warning(
            "No SRTM raster provided or found — generating flat terrain STL"
        )
        _write_flat_terrain_stl(stl_path, geom["total_x_m"], geom["total_y_m"])

    # Adjust domain height: DOMAIN_HEIGHT_M above highest terrain point
    domain_z_max = terrain_z_max + DOMAIN_HEIGHT_M
    geom["total_z_m"] = domain_z_max

    logger.info("Terrain z_max=%.0fm → domain z_max=%.0fm (%.0fm above terrain)",
                terrain_z_max, domain_z_max, DOMAIN_HEIGHT_M)

    # ---- Build closed FMS surface for cfMesh --------------------------------
    half_x = geom["total_x_m"] / 2
    half_y = geom["total_y_m"] / 2

    if is_cylinder:
        # Build closed surface: terrain (bottom) + octagon (lateral + top)
        octagon_stl_content = make_octagon_stl(
            center_x=0.0,
            center_y=0.0,
            radius_m=octagon_radius_m,
            height_m=domain_z_max,
        )
        combined_stl_path = trisurf_dir / "domain_octagon.stl"
        build_octagon_domain_stl(stl_path, octagon_stl_content, combined_stl_path)
        logger.info(
            "Octagonal domain STL saved: %s (radius=%.0f m, height=%.0f m)",
            combined_stl_path, octagon_radius_m, domain_z_max,
        )
    else:
        # Box domain: build closed FMS from terrain + bounding box faces
        fms_path = trisurf_dir / "domain.stl"
        build_domain_fms(
            terrain_stl=stl_path,
            out_fms=fms_path,
            x_min=-half_x, x_max=half_x,
            y_min=-half_y, y_max=half_y,
            z_min=0.0, z_max=domain_z_max,
        )

    # ---- cfMesh refinement zones -------------------------------------------
    if is_cylinder:
        cfmesh_refinements = compute_octagonal_refinements(domain_km, domain_z_max)
    else:
        cfmesh_refinements = compute_cfmesh_refinements(geom, terrain_z_max)
    logger.info(
        "cfMesh refinements: %s",
        ", ".join(f"{r['name']}={r['cell_size']:.0f}m" for r in cfmesh_refinements),
    )

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

    # ---- Fit T(z) polynomial from inflow T_profile (for setExpressionFields) ---
    domain_top = geom["total_z_m"]
    T_poly_coeffs = None
    if "T_profile" in inflow and "z_levels" in inflow:
        _z = np.array(inflow["z_levels"])
        _T = np.array(inflow["T_profile"])
        # Extend profile to domain top with constant value (avoid poly extrapolation)
        if _z[-1] < domain_top:
            z_ext = np.linspace(_z[-1], domain_top, 10)
            _z = np.concatenate([_z, z_ext[1:]])
            _T = np.concatenate([_T, np.full(len(z_ext) - 1, _T[-1])])
        T_poly_coeffs = np.polyfit(_z, _T, 3)[::-1].tolist()
        logger.info("T(z) polynomial fit: T(0)=%.2f K, dT/dz=%.2f K/km",
                     T_poly_coeffs[0], T_poly_coeffs[1] * 1000)

    # ---- Fit U(z) polynomial from inflow u_profile (for setExpressionFields) ---
    U_poly_coeffs = None
    if "u_profile" in inflow and "z_levels" in inflow:
        _z = np.array(inflow["z_levels"])
        _u = np.array(inflow["u_profile"])
        if _z[-1] < domain_top:
            z_ext = np.linspace(_z[-1], domain_top, 10)
            _z = np.concatenate([_z, z_ext[1:]])
            _u = np.concatenate([_u, np.full(len(z_ext) - 1, _u[-1])])
        U_poly_coeffs = np.polyfit(_z, _u, 5)[::-1].tolist()
        logger.info("U(z) polynomial fit: U(10m)=%.2f m/s, U(100m)=%.2f m/s",
                     np.polyval(np.polyfit(_z, _u, 5), 10),
                     np.polyval(np.polyfit(_z, _u, 5), 100))

    # ---- p_rgh at domain top (fixedValue BC — synoptic pressure anchor) --------
    # p_rgh = p/rho0 - g*z  [m2/s2, kinematic]
    _G = 9.81
    _RHO0 = 1.225
    _p_rgh_top = 0.0
    if "p_profile" in inflow and "z_levels" in inflow:
        _p_top_Pa = float(np.interp(domain_top,
                                    np.array(inflow["z_levels"]),
                                    np.array(inflow["p_profile"])))
        _p_rgh_top = _p_top_Pa / _RHO0 + _G * domain_top
        logger.info("p_rgh_top = %.2f m2/s2 (ERA5 p=%.0f Pa at z_top=%.0f m)",
                    _p_rgh_top, _p_top_Pa, domain_top)

    # ---- Lmax: turbulent length scale limiter (Venkatraman KE-Lim) ---------------
    # Lmax = 0.00027 * U_g / fc  (Blackadar 1962, Venkatraman 2023 Table 3)
    # fc = 2 * Omega * sin(lat), U_g = geostrophic wind ≈ ERA5 wind at domain top
    _OMEGA = 7.2921e-5
    _fc = 2.0 * _OMEGA * np.sin(np.radians(site_lat))
    _u_hub = float(inflow.get("u_hub", 10.0))
    if "u_profile" in inflow and "z_levels" in inflow:
        _U_geo = float(np.interp(domain_top,
                                 np.array(inflow["z_levels"]),
                                 np.array(inflow["u_profile"])))
    else:
        _U_geo = _u_hub
    _l_max_calc = 0.00027 * _U_geo / abs(_fc)
    logger.info("Lmax limiter: U_geo=%.1f m/s, fc=%.2e s⁻¹ → Lmax=%.1f m (Blackadar)",
                _U_geo, _fc, _l_max_calc)

    # ---- Robin BC: inletOutlet on all lateral faces ----------------------------
    wind_dir = float(inflow.get("wind_dir", 270.0))
    logger.info("Wind dir %.1f° — Robin BC on all lateral faces", wind_dir)

    # ---- Jinja2 context -------------------------------------------------------
    n_cores = 8
    central_m = geom["central_m"]

    # ---- tower positions in local coords (for sampleDict) -------------------------
    towers = []
    key_towers = (
        site.get("terrain", {}).get("key_towers", [])
        or site_cfg.get("measurement", {}).get("masts", {}).get("key_towers", [])
    )
    for tw in key_towers:
        tw_lat = tw["lat"]
        tw_lon = tw["lon"]
        tw_x = (tw_lon - site_lon) * 111_000.0 * np.cos(np.radians(site_lat))
        tw_y = (tw_lat - site_lat) * 111_000.0
        tw_z = tw.get("elevation_m", 0.0)
        towers.append({
            "id": tw["id"],
            "x": round(tw_x, 1),
            "y": round(tw_y, 1),
            "z_ground": round(tw_z, 1),
        })
    if towers:
        logger.info("Tower positions (local coords): %s",
                     ", ".join(f"{t['id']}=({t['x']},{t['y']},{t['z_ground']})" for t in towers))

    jinja_ctx = {
        "domain": {
            "total_x_m":   geom["total_x_m"],
            "total_y_m":   geom["total_y_m"],
            "total_z_m":   geom["total_z_m"],
            "x_min":       -half_x,
            "x_max":       half_x,
            "y_min":       -half_y,
            "y_max":       half_y,
            "z_min":       0.0,
            "z_max":       geom["total_z_m"],
            "octagonal":   domain_type == "cylinder",
            "radius_m":    domain_km * 1000.0 / 2.0 if domain_type == "cylinder" else None,
        },
        "mesh": {
            "max_cell_size":          geom["max_cell_size"],
            "target_cell_size":       geom["target_cell_size"],
            "fine_cell_size":         geom["fine_cell_size"],
            "terrain_refine_levels":  TERRAIN_REFINE_LEVELS,
            "terrain_refine_thickness": TERRAIN_REFINE_THICKNESS_M,
            "n_boundary_layers":      N_BOUNDARY_LAYERS,
            "bl_expansion":           BL_EXPANSION_RATIO,
            "bl_first_layer":         BL_FIRST_LAYER_M,
        },
        "cfmesh_refinements": cfmesh_refinements,
        "terrain": {
            "stl_file":       STL_FILENAME,
            "central_m":      central_m,
            "resolution_m":   resolution_m,
        },
        "inflow": inflow,
        "site": {
            "latitude":  site_lat,
            "longitude": site_lon,
        },
        "physics": {
            "T_ref_K":    float(inflow.get("T_ref", 300.0)),
            "p_ref_Pa":   0.0,
            "rho_ref":    1.225,
            "p_rgh_top":  _p_rgh_top,
            "coriolis":   coriolis,
            "T_poly_coeffs": T_poly_coeffs,
            "U_poly_coeffs": U_poly_coeffs,
            # Ambient turbulence sources (BBSF k-ε Lim, Venkatraman 2023 Table 3)
            "k_amb":           kwargs.get("k_amb", 0.001),
            "epsilon_amb":     kwargs.get("epsilon_amb", 7.208e-08),
            "l_max":           kwargs.get("l_max", min(_l_max_calc, 62.14)),
            "use_lmax_limiter": kwargs.get("use_lmax_limiter", thermal),
        },
        "solver": {
            "name":           solver_name,
            "n_iter":         kwargs.get("n_iter", 1000),
            "write_interval": kwargs.get("write_interval", 100),
            "n_cores":        n_cores,
            "thermal":        thermal,
            "boussinesq":     kwargs.get("boussinesq", False),
        },
        "canopy": {
            "enabled": canopy_enabled,
            "Cd": 0.2,
            "C4_eps": 0.9,
        },
        "parallel": {
            "n_cores": n_cores,
        },
        "towers": towers,
    }

    # ---- render templates -------------------------------------------------------
    logger.info("Rendering OpenFOAM templates → %s", output_dir)
    render_templates(template_dir, output_dir, jinja_ctx)

    # ---- Remove solver-incompatible constant files --------------------------------
    if solver_name in ("simpleFoam", "buoyantBoussinesqSimpleFoam"):
        thermo = output_dir / "constant" / "thermophysicalProperties"
        if thermo.exists():
            thermo.unlink()
            logger.info("Removed thermophysicalProperties (not used by %s)", solver_name)

    # ---- turbulenceProperties copy for ESI v2406 compatibility ---------------------
    mt_file = output_dir / "constant" / "momentumTransport"
    tp_copy = output_dir / "constant" / "turbulenceProperties"
    if mt_file.exists() and not tp_copy.exists():
        shutil.copy2(mt_file, tp_copy)
        logger.info("Copied momentumTransport -> turbulenceProperties (decomposePar compat)")

    # ---- create empty .foam file for ParaView -------------------------------------
    foam_file = output_dir / "case.foam"
    foam_file.touch()

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
        description="Generate OpenFOAM case directory from SRTM DEM (cfMesh)"
    )
    parser.add_argument("--site",           default="perdigao",
                        help="Site name (configs/sites/<site>.yaml)")
    parser.add_argument("--resolution-m",   type=float, required=True,
                        help="Target horizontal resolution [m] in central zone")
    parser.add_argument("--context-cells",  type=int, default=3, choices=[1, 3, 5],
                        help="Super-maille context: 1=pipeline test, 3=75km, 5=125km")
    parser.add_argument("--domain-km",      type=float, default=DEFAULT_DOMAIN_KM,
                        help=f"Central zone width in km (default {DEFAULT_DOMAIN_KM})")
    parser.add_argument("--srtm",           default=None,
                        help="SRTM GeoTIFF path (default: data/raw/srtm_<site>_30m.tif)")
    parser.add_argument("--inflow-json",    default=None,
                        help="Inflow profile JSON (from prepare_inflow.py)")
    parser.add_argument("--solver",         default="simpleFoam",
                        help="Solver name (simpleFoam or buoyantBoussinesqSimpleFoam)")
    parser.add_argument("--thermal",        action="store_true",
                        help="Enable thermal coupling (buoyantBoussinesqSimpleFoam)")
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
        domain_km=args.domain_km,
        solver_name=args.solver,
        thermal=args.thermal,
    )

    print(f"Case generated: {args.output}")
    print(f"  target res  : {args.resolution_m:.0f} m")
    print(f"  max cell    : {geom['max_cell_size']:.0f} m (cfMesh outer)")
    print(f"  fine cell   : {geom['fine_cell_size']:.0f} m (near terrain)")
    print(f"  context     : {args.context_cells}×{args.context_cells} super-mailles")
    print(f"  domain      : {geom['total_x_m']/1000:.0f}×{geom['total_y_m']/1000:.0f}×{geom['total_z_m']/1000:.0f} km")
