#!/usr/bin/env python3
"""
setup_uluru_cfd.py — Prepare Uluru CFD cases for website 3D wind demo.

Generates STL from COP-DEM, creates synthetic inflow profiles (log-law,
15 m/s neutral), renders OpenFOAM templates, and writes TBM mesh configs
for 2 resolutions × 2 wind directions = 4 CFD cases. Also generates the
coarse (5 km) analytical CSV. Outputs a bash script for UGA execution.

Usage:
    conda run -n downscalewind python notebooks/setup_uluru_cfd.py

Output:
    data/website/uluru/cfd/
        terrain.stl
        mesh_500m/  mesh_100m/       (TBM mesh dirs)
        case_500m_cross/  ...along/  (OF case dirs)
        case_100m_cross/  ...along/
        run_uluru.sh                 (UGA execution script)
    data/website/uluru/
        terrain_5km.csv  wind_cross_5km.csv  wind_along_5km.csv
"""
from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "services" / "module2a-cfd" / "templates" / "openfoam"
DEM_TIF = ROOT / "data" / "website" / "uluru" / "copdem_uluru_30m.tif"
OUTDIR = ROOT / "data" / "website" / "uluru"
CFD_DIR = OUTDIR / "cfd"

# ── Site constants ────────────────────────────────────────────────────────────

ULURU_LAT = -25.3444
ULURU_LON = 131.0369

# Domain geometry (metres, local coords centred on Uluru)
CYLINDER_RADIUS = 10_000       # 20 km total diameter
INNER_SIZE = 4500              # inner structured block side [m]
DOMAIN_HEIGHT = 5000           # [m]
STL_RESOLUTION = 50            # STL resolution [m] (2 pts per finest cell)
STL_DOMAIN = 2 * CYLINDER_RADIUS + 2000  # 22 km STL extent

# Solver
N_ITER = 500
N_CORES = 24
WRITE_INTERVAL = 500

# Physics
Z0 = 0.01          # desert roughness [m]
T_REF = 303.0      # 30°C (austral summer)
WIND_SPEED = 15.0   # m/s

# Docker images (UGA)
OF_IMAGE = "microfluidica/openfoam:latest"
TBM_IMAGE = "terrainblockmesher:of24"

# ── Mesh configs ──────────────────────────────────────────────────────────────

MESH_CONFIGS = {
    "500m": {
        "inner_size_m": INNER_SIZE,
        "inner_blocks": 3,
        "cells_per_block_xy": 3,
        "cylinder_radius_m": CYLINDER_RADIUS,
        "cylinder_sections": 8,
        "radial_cells": 10,
        "radial_grading": 15,
        "height_m": DOMAIN_HEIGHT,
        "cells_z": 30,
        "grading_z": 10,
        "max_dist_proj": 15000,
        "blend_distance_m": 3000,
        "p_above_z": 10000,
    },
    "100m": {
        "inner_size_m": INNER_SIZE,
        "inner_blocks": 15,
        "cells_per_block_xy": 3,
        "cylinder_radius_m": CYLINDER_RADIUS,
        "cylinder_sections": 8,
        "radial_cells": 20,
        "radial_grading": 20,
        "height_m": DOMAIN_HEIGHT,
        "cells_z": 50,
        "grading_z": 15,
        "max_dist_proj": 15000,
        "blend_distance_m": 3000,
        "p_above_z": 10000,
    },
}

# Wind conditions: (name, direction_from_deg)
# cross = from south (180°) → perpendicular to Uluru's long axis
# along = from west  (270°) → along Uluru's long axis
WIND_CONDITIONS = [
    ("cross", 180.0),
    ("along", 270.0),
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. STL generation from local COP-DEM
# ══════════════════════════════════════════════════════════════════════════════

def generate_stl(dem_tif: Path, output_stl: Path) -> float:
    """Generate terrain STL from local COP-DEM GeoTIFF.

    Returns terrain_z_min (for TBM p_corner).
    """
    import rasterio
    from pyproj import Transformer
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, reproject

    # UTM zone for Uluru: zone 52S
    utm_zone = int((ULURU_LON + 180) / 6) + 1
    utm_epsg = f"EPSG:327{utm_zone:02d}"

    to_utm = Transformer.from_crs("EPSG:4326", utm_epsg, always_xy=True)
    to_wgs = Transformer.from_crs(utm_epsg, "EPSG:4326", always_xy=True)

    x0, y0 = to_utm.transform(ULURU_LON, ULURU_LAT)
    half = STL_DOMAIN / 2
    nx = ny = int(STL_DOMAIN / STL_RESOLUTION)

    # Read DEM and reproject to UTM at target resolution
    with rasterio.open(dem_tif) as src:
        dst_transform = from_bounds(x0 - half, y0 - half, x0 + half, y0 + half, nx, ny)
        dst_array = np.zeros((ny, nx), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=utm_epsg,
            resampling=Resampling.bilinear,
        )

    # Clean nodata
    valid = dst_array > -500
    if not valid.any():
        raise ValueError("No valid terrain data")
    z_min_valid = float(dst_array[valid].min())
    dst_array[~valid] = z_min_valid

    terrain_z_min = z_min_valid - 50

    # Write ASCII STL in local coords (centre = 0,0)
    xs = np.linspace(-half, half, nx)
    ys = np.linspace(-half, half, ny)

    output_stl.parent.mkdir(parents=True, exist_ok=True)
    ntri = 0
    with open(output_stl, "w") as f:
        f.write("solid terrain\n")
        for j in range(ny - 1):
            for i in range(nx - 1):
                pts = [
                    (xs[i], ys[j], float(dst_array[j, i])),
                    (xs[i + 1], ys[j], float(dst_array[j, i + 1])),
                    (xs[i], ys[j + 1], float(dst_array[j + 1, i])),
                    (xs[i + 1], ys[j + 1], float(dst_array[j + 1, i + 1])),
                ]
                for tri in [(pts[0], pts[1], pts[2]), (pts[1], pts[3], pts[2])]:
                    f.write("  facet normal 0 0 1\n    outer loop\n")
                    for v in tri:
                        f.write(f"      vertex {v[0]:.1f} {v[1]:.1f} {v[2]:.1f}\n")
                    f.write("    endloop\n  endfacet\n")
                    ntri += 1
        f.write("endsolid terrain\n")

    size_mb = output_stl.stat().st_size / 1e6
    print(f"  STL: {ntri} triangles, {size_mb:.1f} MB, "
          f"z=[{dst_array.min():.0f}, {dst_array.max():.0f}] m")
    return terrain_z_min


# ══════════════════════════════════════════════════════════════════════════════
# 2. Synthetic inflow profile
# ══════════════════════════════════════════════════════════════════════════════

def create_synthetic_inflow(wind_dir_deg: float) -> dict:
    """Create synthetic neutral log-law inflow profile for Uluru demo."""
    kappa = 0.41
    z_ref = 100.0

    u_star = WIND_SPEED * kappa / math.log(z_ref / Z0)

    # Z levels: log near ground, linear in ABL, sparse above
    z_log = np.geomspace(1, 100, 30)
    z_lin = np.linspace(140, 2000, 50)
    z_upper = np.array([3000, 4000, 5000])
    z_levels = np.concatenate([z_log, z_lin, z_upper])

    # Log-law profile, capped above BL top
    u_profile = (u_star / kappa) * np.log(z_levels / Z0)
    # Smooth cap above 1000m (geostrophic)
    u_geo = (u_star / kappa) * math.log(1000.0 / Z0)
    mask = z_levels > 1000
    u_profile[mask] = u_geo + 0.001 * (z_levels[mask] - 1000)  # slight increase

    # Wind direction (meteorological: FROM)
    dir_rad = math.radians(wind_dir_deg)
    fd_x = -math.sin(dir_rad)   # towards east
    fd_y = -math.cos(dir_rad)   # towards north

    # Temperature: ISA lapse rate from T_ref at ground
    T_profile = T_REF - 0.0065 * z_levels

    # Pressure: hydrostatic (barometric formula)
    g, R, lapse = 9.81, 287.05, 0.0065
    p_profile = 101325.0 * (T_profile / T_REF) ** (g / (R * lapse))

    return {
        "u_hub": float(WIND_SPEED),
        "u_star": float(u_star),
        "z0_eff": Z0,
        "L_mo": None,
        "T_ref": T_REF,
        "flowDir_x": fd_x,
        "flowDir_y": fd_y,
        "Ri_b": 0.0,
        "z_levels": z_levels.tolist(),
        "u_profile": u_profile.tolist(),
        "ux_profile": (u_profile * fd_x).tolist(),
        "uy_profile": (u_profile * fd_y).tolist(),
        "T_profile": T_profile.tolist(),
        "p_profile": p_profile.tolist(),
        "z0": Z0,
        "kappa": kappa,
        "d": 0.0,
        "z_ref": z_ref,
        "wind_dir": wind_dir_deg,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. TBM mesh directory setup
# ══════════════════════════════════════════════════════════════════════════════

def setup_mesh_dir(mesh_dir: Path, mesh_cfg: dict, stl_path: Path,
                   terrain_z_min: float) -> None:
    """Write TBM dict + minimal system files for meshing."""
    m = mesh_cfg
    half = m["inner_size_m"] / 2
    n_blocks = m["inner_blocks"]
    cpb = m["cells_per_block_xy"]

    dict_content = f"""\
FoamFile {{ version 2.0; format ascii; class dictionary; object terrainBlockMesherDict; }}

stl {{ terrain.stl {{ type triSurfaceMesh; }} }};

writeBlockMeshDict  true;
writePolyMesh       true;

blockManager
{{
    coordinates
    {{
        origin      (0 0 0);
        baseVectors (( 1 0 0 )( 0 1 0 )( 0 0 1 ));
    }}

    p_corner    ({-half:.0f} {-half:.0f} {terrain_z_min:.0f});
    dimensions  ({m["inner_size_m"]:.0f} {m["inner_size_m"]:.0f} {m["height_m"]:.0f});

    p_above     (0 0 {m["p_above_z"]});

    blocks      ({n_blocks} {n_blocks} 1);
    cells       ({cpb} {cpb} {m["cells_z"]});

    maxDistProj {m["max_dist_proj"]};

    gradingFactors  ( 1 1 {m["grading_z"]} );

    patch_name_west     west;
    patch_name_east     east;
    patch_name_north    north;
    patch_name_south    south;
    patch_name_sky      top;
    patch_name_ground   terrain;

    patch_type_west     patch;
    patch_type_east     patch;
    patch_type_north    patch;
    patch_type_south    patch;
    patch_type_sky      patch;
    patch_type_ground   wall;

    outerCylinder
    {{
        centrePoint                 (0 0 {terrain_z_min:.0f});
        radius                      {m["cylinder_radius_m"]};
        radialGrading               {m["radial_grading"]};
        radialBlockCells            {m["radial_cells"]};
        firstSectionStartDirection  (-1 1 0);
        numberOfSections            {m["cylinder_sections"]};

        blendingFunction
        {{
            type    distance;
            dMin    0;
            dMax    {m["blend_distance_m"]};

            transitionFunction
            {{
                type    linear;
            }}
        }}
    }}
}}

checkMesh           false;
checkMeshNoTopology false;
checkMeshAllGeometry false;
checkMeshAllTopology false;
"""

    system = mesh_dir / "system"
    system.mkdir(parents=True, exist_ok=True)
    (system / "terrainBlockMesherDict").write_text(dict_content)

    # Minimal OF system files for TBM
    (system / "controlDict").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\n"
        "application terrainBlockMesher; startFrom latestTime; startTime 0; "
        "stopAt endTime; endTime 0; deltaT 1; writeControl timeStep; "
        "writeInterval 1; writeFormat ascii; writePrecision 10; "
        "writeCompression uncompressed; timeFormat general; timePrecision 6;\n"
    )
    (system / "fvSchemes").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }\n"
        "ddtSchemes { default steadyState; }\n"
        "gradSchemes { default Gauss linear; }\n"
        "divSchemes { default none; }\n"
        "laplacianSchemes { default Gauss linear corrected; }\n"
        "interpolationSchemes { default linear; }\n"
        "snGradSchemes { default corrected; }\n"
    )
    (system / "fvSolution").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\n"
    )

    # STL will be mounted via Docker volume on UGA (not copied)
    inner_res = m["inner_size_m"] / (n_blocks * cpb)
    total_cells_inner = n_blocks**2 * cpb**2 * m["cells_z"]
    print(f"  Mesh dir: {mesh_dir.name} "
          f"({inner_res:.0f}m inner, ~{total_cells_inner//1000}k inner cells)")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Case directory setup (render templates)
# ══════════════════════════════════════════════════════════════════════════════

def setup_case(case_dir: Path, inflow: dict, mesh_cfg: dict) -> None:
    """Render all Jinja2 templates for one OpenFOAM case."""
    from jinja2 import Environment, FileSystemLoader

    case_dir.mkdir(parents=True, exist_ok=True)

    # Write inflow.json
    with open(case_dir / "inflow.json", "w") as f:
        json.dump(inflow, f, indent=2)

    # STL not needed in case dirs (only mesh dirs use it for TBM)
    # Solver reads terrain from polyMesh, not STL

    # Jinja2 context (matches run_multisite_campaign.py)
    n_sections = mesh_cfg.get("cylinder_sections", 8)
    lateral_patches = [f"section_{i}" for i in range(n_sections)]

    ctx = {
        "domain": {
            "octagonal": True,
            "lateral_patches": lateral_patches,
            "radius_m": mesh_cfg["cylinder_radius_m"],
            "z_max": mesh_cfg["height_m"],
        },
        "solver": {
            "name": "simpleFoam",
            "n_iter": N_ITER,
            "n_cores": N_CORES,
            "write_interval": WRITE_INTERVAL,
            "transport_T": True,
            "transport_q": False,   # not needed for demo
        },
        "physics": {
            "coriolis": False,  # simpler for demo
            "T_ref_K": inflow["T_ref"],
        },
        "canopy": {"enabled": False},
        "inflow": inflow,
        "site": {"latitude": ULURU_LAT, "longitude": ULURU_LON},
    }

    # Render all .j2 templates
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)),
                      keep_trailing_newline=True)
    skip = {"meshDict.j2", "terrainBlockMesherDict.j2"}

    for tmpl_path in sorted(TEMPLATE_DIR.rglob("*.j2")):
        if tmpl_path.name in skip:
            continue
        rel = tmpl_path.relative_to(TEMPLATE_DIR)
        out_file = case_dir / rel.with_suffix("")
        tmpl = env.get_template(str(rel))
        rendered = tmpl.render(**ctx)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(rendered)

    print(f"  Case: {case_dir.name} "
          f"(wind_dir={inflow['wind_dir']:.0f}°, {WIND_SPEED} m/s)")


# ══════════════════════════════════════════════════════════════════════════════
# 5. UGA run script
# ══════════════════════════════════════════════════════════════════════════════

def write_uga_script(script_path: Path, terrain_z_min: float) -> None:
    """Generate bash script for UGA execution."""
    # Write a helper solve script (avoids quoting hell in Docker bash -c)
    solve_helper = script_path.parent / "solve_case.sh"
    solve_helper.write_text("""\
#!/usr/bin/env bash
# solve_case.sh — called inside Docker container
set -e
cd /home/ofuser/run

# OF ESI compat: turbulenceProperties alias
if [ -f constant/momentumTransport ] && [ ! -f constant/turbulenceProperties ]; then
    cp constant/momentumTransport constant/turbulenceProperties
fi

N_CORES=$1
UID_GID=$2

foamDictionary system/decomposeParDict -entry numberOfSubdomains -set "$N_CORES"
decomposePar -force > log.decomposePar 2>&1

# Symlink boundaryData into processor dirs
if [ -d constant/boundaryData ]; then
    for d in processor*/; do
        ln -sf ../../constant/boundaryData "$d/constant/"
    done
fi

mpirun -np "$N_CORES" simpleFoam -parallel > log.simpleFoam 2>&1
chown -R "$UID_GID" /home/ofuser/run
""")
    solve_helper.chmod(0o755)

    # Main UGA script (uses plain string template for readability)
    script_content = (
        '#!/usr/bin/env bash\n'
        '# run_uluru.sh — Run Uluru CFD demo on UGA\n'
        '# Upload this entire cfd/ directory to UGA, then:\n'
        '#   cd /home/guillaume/dsw/uluru_demo && bash run_uluru.sh\n'
        'set -euo pipefail\n'
        '\n'
        'SCRIPTS="/home/guillaume/dsw/scripts"\n'
        'PYTHON="/home/guillaume/miniconda3/bin/python"\n'
        f'OF_IMG="{OF_IMAGE}"\n'
        f'TBM_IMG="{TBM_IMAGE}"\n'
        f'N_CORES={N_CORES}\n'
        f'N_ITER={N_ITER}\n'
        'UID_GID="$(id -u):$(id -g)"\n'
        'STL="$(pwd)/terrain.stl"\n'
        '\n'
        'echo "=== Uluru CFD Demo ==="\n'
        'echo "Started: $(date)"\n'
        '\n'
        '# ── 1. TBM mesh generation ──────────────────────────────────────\n'
        'for MESH in mesh_500m mesh_100m; do\n'
        '    echo ""\n'
        '    echo "--- Meshing: $MESH ---"\n'
        '    if [ -f "$MESH/constant/polyMesh/points" ]; then\n'
        '        echo "  polyMesh already exists, skipping"\n'
        '    else\n'
        '        mkdir -p "$MESH/constant/triSurface"\n'
        '        docker run --rm \\\n'
        '            -v "$(pwd)/$MESH:/home/ofuser/run" \\\n'
        '            -v "$STL:/home/ofuser/run/constant/triSurface/terrain.stl:ro" \\\n'
        '            -w /home/ofuser/run \\\n'
        '            "$TBM_IMG" \\\n'
        '            bash -c "terrainBlockMesher; chown -R $UID_GID /home/ofuser/run"\n'
        '        echo "  Mesh done"\n'
        '    fi\n'
        '\n'
        '    # writeCellCentres\n'
        '    if [ ! -f "$MESH/0/Cx" ]; then\n'
        '        mkdir -p "$MESH/0"\n'
        "        cat > \"$MESH/0/p\" << 'PEOF'\n"
        'FoamFile { version 2.0; format ascii; class volScalarField; object p; }\n'
        'dimensions [0 2 -2 0 0 0 0];\n'
        'internalField uniform 0;\n'
        'boundaryField { ".*" { type zeroGradient; } }\n'
        'PEOF\n'
        '        docker run --rm \\\n'
        '            -v "$(pwd)/$MESH:/home/ofuser/run" \\\n'
        '            -w /home/ofuser/run \\\n'
        '            "$OF_IMG" \\\n'
        '            bash -c "postProcess -func writeCellCentres -time 0; '
        'chown -R $UID_GID /home/ofuser/run/0"\n'
        '        echo "  writeCellCentres done"\n'
        '    fi\n'
        'done\n'
        '\n'
        '# ── 2. Per-case: copy mesh → init → solve ───────────────────────\n'
        'for CASE in case_500m_cross case_500m_along case_100m_cross case_100m_along; do\n'
        '    echo ""\n'
        '    echo "=== Case: $CASE ==="\n'
        '\n'
        '    # Determine mesh dir\n'
        '    if [[ "$CASE" == case_500m* ]]; then MESH=mesh_500m; else MESH=mesh_100m; fi\n'
        '\n'
        '    # Check if already solved\n'
        '    if [ -f "$CASE/$N_ITER/U" ]; then\n'
        '        echo "  Already solved, skipping"\n'
        '        continue\n'
        '    fi\n'
        '\n'
        '    # Copy shared mesh + cell centres\n'
        '    if [ ! -d "$CASE/constant/polyMesh" ]; then\n'
        '        echo "  Copying mesh from $MESH..."\n'
        '        mkdir -p "$CASE/constant"\n'
        '        cp -r "$MESH/constant/polyMesh" "$CASE/constant/"\n'
        '    fi\n'
        '    for F in Cx Cy Cz; do\n'
        '        if [ -f "$MESH/0/$F" ] && [ ! -f "$CASE/0/$F" ]; then\n'
        '            cp "$MESH/0/$F" "$CASE/0/"\n'
        '        fi\n'
        '    done\n'
        '\n'
        '    # Init from inflow\n'
        '    echo "  Initializing fields..."\n'
        '    $PYTHON "$SCRIPTS/init_from_era5.py" \\\n'
        '        --case-dir "$CASE" \\\n'
        '        --inflow "$CASE/inflow.json"\n'
        '\n'
        '    # Solve: decompose → mpirun → reconstruct\n'
        '    echo "  Solving ($N_CORES cores, $N_ITER iter)..."\n'
        '    docker run --rm \\\n'
        '        -v "$(pwd)/$CASE:/home/ofuser/run" \\\n'
        '        -v "$(pwd)/solve_case.sh:/home/ofuser/solve_case.sh:ro" \\\n'
        '        -w /home/ofuser/run \\\n'
        '        "$OF_IMG" \\\n'
        '        bash /home/ofuser/solve_case.sh "$N_CORES" "$UID_GID"\n'
        '\n'
        '    # Reconstruct\n'
        '    echo "  Reconstructing..."\n'
        '    $PYTHON "$SCRIPTS/reconstruct_fields.py" \\\n'
        '        --case-dir "$CASE" \\\n'
        '        --time latest \\\n'
        '        --write-foam \\\n'
        '        --fields U T k epsilon nut p\n'
        '\n'
        '    # Clean processor dirs\n'
        '    rm -rf "$CASE"/processor*\n'
        '    echo "  Done: $CASE"\n'
        'done\n'
        '\n'
        'echo ""\n'
        'echo "=== All cases complete ==="\n'
        'echo "Finished: $(date)"\n'
        'echo "Results in: $(pwd)/case_*/500/"\n'
    )
    script_path.write_text(script_content)
    script_path.chmod(0o755)
    print(f"  UGA script: {script_path}")
    print(f"  Solve helper: {solve_helper}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Coarse (5 km) analytical CSV
# ══════════════════════════════════════════════════════════════════════════════

def generate_coarse_csv() -> None:
    """Generate 5 km terrain + wind CSVs (ERA5-like: flat terrain, uniform wind)."""
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import gaussian_filter
    import rasterio

    print("\n--- Generating 5 km analytical CSVs ---")

    with rasterio.open(DEM_TIF) as src:
        data = src.read(1)
        tf = src.transform
    ncols, nrows = data.shape[1], data.shape[0]
    lons = np.array([tf.c + (j + 0.5) * tf.a for j in range(ncols)])
    lats = np.array([tf.f + (i + 0.5) * tf.e for i in range(nrows)])

    deg_lat = 111_320
    deg_lon = deg_lat * math.cos(math.radians(abs(ULURU_LAT)))
    half_lat = 10 / 111.32
    half_lon = 10 / (111.32 * math.cos(math.radians(abs(ULURU_LAT))))
    step = 5000

    lon_t = np.arange(ULURU_LON - half_lon, ULURU_LON + half_lon + 0.001, step / deg_lon)
    lat_t = np.arange(ULURU_LAT + half_lat, ULURU_LAT - half_lat - 0.001, -step / deg_lat)

    interp = RegularGridInterpolator(
        (lats[::-1], lons), data[::-1],
        method="linear", bounds_error=False, fill_value=None,
    )
    lon_g, lat_g = np.meshgrid(lon_t, lat_t)
    z = interp((lat_g, lon_g))
    # Heavy smoothing (ERA5-like: 25 km effective)
    sigma_px = max(1, 25000 / step)
    z_smooth = gaussian_filter(z, sigma=sigma_px)

    # Terrain CSV
    with open(OUTDIR / "terrain_5km.csv", "w") as f:
        f.write("lon,lat,z\n")
        for i in range(len(lat_t)):
            for j in range(len(lon_t)):
                f.write(f"{lon_t[j]:.6f},{lat_t[i]:.6f},{z_smooth[i, j]:.1f}\n")

    # Wind CSVs (uniform — ERA5 can't see Uluru)
    for cond_name, wind_dir in WIND_CONDITIONS:
        dir_r = math.radians(wind_dir)
        u0 = -WIND_SPEED * math.sin(dir_r)
        v0 = -WIND_SPEED * math.cos(dir_r)
        with open(OUTDIR / f"wind_{cond_name}_5km.csv", "w") as f:
            f.write("lon,lat,u,v,w\n")
            for i in range(len(lat_t)):
                for j in range(len(lon_t)):
                    f.write(f"{lon_t[j]:.6f},{lat_t[i]:.6f},{u0:.2f},{v0:.2f},0.00\n")

    print(f"  terrain_5km.csv: {len(lat_t)}x{len(lon_t)} = {len(lat_t)*len(lon_t)} pts")
    print(f"  wind_cross_5km.csv, wind_along_5km.csv")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Setup Uluru CFD demo")
    print("=" * 60)

    # 1. Generate STL
    stl_path = CFD_DIR / "terrain.stl"
    print("\n--- 1. STL generation ---")
    if stl_path.exists():
        print(f"  STL already exists: {stl_path}")
        # Read terrain_z_min from STL (scan for min z)
        z_min = float("inf")
        with open(stl_path) as f:
            for line in f:
                if line.strip().startswith("vertex"):
                    z = float(line.split()[3])
                    z_min = min(z_min, z)
        terrain_z_min = z_min - 50
    else:
        terrain_z_min = generate_stl(DEM_TIF, stl_path)
    print(f"  terrain_z_min = {terrain_z_min:.0f} m")

    # 2. Mesh directories
    print("\n--- 2. TBM mesh directories ---")
    for res_label, mesh_cfg in MESH_CONFIGS.items():
        mesh_dir = CFD_DIR / f"mesh_{res_label}"
        if (mesh_dir / "system" / "terrainBlockMesherDict").exists():
            print(f"  mesh_{res_label}: already set up")
        else:
            setup_mesh_dir(mesh_dir, mesh_cfg, stl_path, terrain_z_min)

    # 3. Case directories
    print("\n--- 3. Case directories ---")
    for res_label, mesh_cfg in MESH_CONFIGS.items():
        for cond_name, wind_dir in WIND_CONDITIONS:
            case_name = f"case_{res_label}_{cond_name}"
            case_dir = CFD_DIR / case_name
            if (case_dir / "inflow.json").exists():
                print(f"  {case_name}: already set up")
                continue
            inflow = create_synthetic_inflow(wind_dir)
            setup_case(case_dir, inflow, mesh_cfg)

    # 4. UGA script
    print("\n--- 4. UGA run script ---")
    write_uga_script(CFD_DIR / "run_uluru.sh", terrain_z_min)

    # 5. Coarse analytical CSV
    generate_coarse_csv()

    # Summary
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print(f"\nFiles in: {CFD_DIR}/")
    print("\nNext steps:")
    print(f"  1. scp -r {CFD_DIR}/ guillaume@uga:/home/guillaume/dsw/uluru_demo/")
    print("  2. ssh uga")
    print("  3. cd /home/guillaume/dsw/uluru_demo && bash run_uluru.sh")
    print("  4. scp back the case_*/500/ directories")
    print("  5. Run export script to generate website CSVs")

    total_mb = sum(f.stat().st_size for f in CFD_DIR.rglob("*") if f.is_file()) / 1e6
    print(f"\nTotal size: {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
