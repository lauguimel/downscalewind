"""
init_from_era5.py — Initialise OpenFOAM fields from ERA5 interpolation

Replaces potentialFoam: interpolates ERA5 wind (u, v), k, epsilon, and
temperature T to every cell centre AND boundary face centre in the mesh.

Pipeline position:
    cartesianMesh → checkMesh → **init_from_era5** → simpleFoam

The script:
  1. Reads cell centres from the OpenFOAM case
  2. Reads boundary face centres from constant/polyMesh/{points,faces,boundary}
  3. For each cell/face at height z:
     - Interpolates ERA5 speed, T vertically (linear on z_levels/u_profile)
     - Computes U = speed × (flowDir_x, flowDir_y, 0)
     - Computes k = u*²/√Cmu  and  ε(z) = Cmu^0.75·k^1.5/(κ·max(z, 2·z0))
  4. Writes internalField as nonuniform List
  5. Writes constant/boundaryData/<patch>/points + 0/<field> for MappedFile BCs
  6. Detects solver (simpleFoam vs BBSF) for correct p_rgh formulation

Usage
-----
    python init_from_era5.py \\
        --case-dir  data/cases/perdigao_500m_1x1/ \\
        --inflow    data/processed/inflow/2017-05-11T06_00.json
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

KAPPA = 0.41
CMU   = 0.09

# Patches on lateral faces that receive MappedFile boundaryData.
# Top face now uses zeroGradient U + fixedValue p_rgh (no boundaryData needed).
# Wall functions, noSlip, zeroGradient, fixedFluxPressure etc. are left unchanged.
BOUNDARY_DATA_PATCHES = {"west", "east", "south", "north"}


def detect_lateral_patches(boundary_faces: dict) -> set[str]:
    """Detect which patches should receive lateral inflow boundary data.

    For cylindrical (octagonal) domains, a single ``lateral`` patch is used.
    For box domains, the four cardinal patches are used.

    Parameters
    ----------
    boundary_faces : dict
        Mapping of patch name → patch data (e.g., the boundary field dict
        from an OpenFOAM ``0/U`` file or the inflow profile's patch list).

    Returns
    -------
    set[str]
        Set of patch names to write boundaryData for.

    Examples
    --------
    >>> detect_lateral_patches({"lateral": {}, "top": {}, "terrain": {}})
    {'lateral'}
    >>> detect_lateral_patches({"west": {}, "east": {}, "south": {}, "north": {}, "top": {}})
    {'west', 'east', 'south', 'north'}
    """
    if "lateral" in boundary_faces:
        return {"lateral"}
    # terrainBlockMesher cylindrical domain: section_0 .. section_N
    sections = {k for k in boundary_faces if k.startswith("section_")}
    if sections:
        return sections
    return {"west", "east", "south", "north"}

# Legacy: patch types for old inletOutlet workflow (kept for backward compat)
PATCHABLE_BC_TYPES = {"inletOutlet", "outletInlet"}

# Reference values for p_rgh computation
G_ACC  = 9.81    # m/s²
RHO0   = 1.225   # kg/m³ (reference density for Boussinesq)


# ---------------------------------------------------------------------------
# Read OpenFOAM cell centres
# ---------------------------------------------------------------------------

def read_cell_centres(case_dir: Path) -> np.ndarray:
    """Read cell centre coordinates from an OpenFOAM case.

    Tries (in order):
      1. Parse 0/C{x,y,z} written by `postProcess -func writeCellCentres`
      2. Parse constant/polyMesh/C (if it exists)

    Returns
    -------
    centres : (N, 3) array of (x, y, z) cell centre coordinates [m].
    """
    cx_path = case_dir / "0" / "Cx"
    cy_path = case_dir / "0" / "Cy"
    cz_path = case_dir / "0" / "Cz"
    if cx_path.exists() and cy_path.exists() and cz_path.exists():
        cx = _parse_of_scalar_field(cx_path)
        cy = _parse_of_scalar_field(cy_path)
        cz = _parse_of_scalar_field(cz_path)
        return np.column_stack([cx, cy, cz])

    c_path = case_dir / "constant" / "polyMesh" / "C"
    if c_path.exists():
        return _parse_of_vector_field(c_path)

    raise FileNotFoundError(
        f"Cannot find cell centres in {case_dir}. "
        "Run `postProcess -func writeCellCentres` first."
    )


# ---------------------------------------------------------------------------
# Read OpenFOAM boundary face centres from mesh files
# ---------------------------------------------------------------------------

def read_boundary_info(case_dir: Path) -> dict[str, dict]:
    """Parse constant/polyMesh/boundary to get patch names, start faces, nFaces.

    Returns
    -------
    dict mapping patch_name → {"nFaces": int, "startFace": int}
    """
    boundary_path = case_dir / "constant" / "polyMesh" / "boundary"
    if not boundary_path.exists():
        raise FileNotFoundError(f"Cannot find {boundary_path}")

    text = boundary_path.read_text()

    # Find the top-level list: N ( ... )
    match = re.search(r'^\s*(\d+)\s*\(', text, re.MULTILINE)
    if not match:
        raise ValueError(f"Cannot parse boundary file: {boundary_path}")

    block = text[match.end():]

    patches = {}
    # Match each patch entry: name { ... nFaces N; startFace M; ... }
    for m in re.finditer(
        r'(\w+)\s*\{([^}]+)\}', block
    ):
        name = m.group(1)
        body = m.group(2)
        nf = re.search(r'nFaces\s+(\d+)', body)
        sf = re.search(r'startFace\s+(\d+)', body)
        if nf and sf:
            patches[name] = {
                "nFaces": int(nf.group(1)),
                "startFace": int(sf.group(1)),
            }

    return patches


def read_boundary_face_centres(case_dir: Path) -> dict[str, np.ndarray]:
    """Compute face centres for each boundary patch from the mesh.

    Reads constant/polyMesh/{points, faces, boundary} and computes
    face centres as the mean of vertex coordinates for each face.

    Returns
    -------
    dict mapping patch_name → (nFaces, 3) array of face centre coordinates [m].
    """
    poly = case_dir / "constant" / "polyMesh"

    # --- Read points ---
    points_text = (poly / "points").read_text()
    match = re.search(r'^\s*(\d+)\s*\(', points_text, re.MULTILINE)
    if not match:
        raise ValueError("Cannot parse points file")
    n_points = int(match.group(1))
    block = points_text[match.end():]
    coords = re.findall(
        r'\(\s*([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s*\)', block
    )
    points = np.array([[float(x), float(y), float(z)] for x, y, z in coords[:n_points]])

    # --- Read faces ---
    faces_text = (poly / "faces").read_text()
    match = re.search(r'^\s*(\d+)\s*\(', faces_text, re.MULTILINE)
    if not match:
        raise ValueError("Cannot parse faces file")
    n_faces = int(match.group(1))
    block = faces_text[match.end():]
    # Each face: N(v0 v1 v2 ...) — parse all
    face_entries = re.findall(r'\d+\(([^)]+)\)', block)
    faces = []
    for entry in face_entries[:n_faces]:
        verts = [int(v) for v in entry.split()]
        faces.append(verts)

    # --- Read boundary patches ---
    patches = read_boundary_info(case_dir)

    # --- Compute face centres for each boundary patch ---
    result = {}
    for patch_name, info in patches.items():
        start = info["startFace"]
        n = info["nFaces"]
        centres = np.zeros((n, 3))
        for i in range(n):
            face_idx = start + i
            if face_idx < len(faces):
                verts = faces[face_idx]
                centres[i] = points[verts].mean(axis=0)
        result[patch_name] = centres
        if n > 0:
            logger.debug("Patch %s: %d faces, z range [%.1f, %.1f] m",
                         patch_name, n, centres[:, 2].min(), centres[:, 2].max())

    return result


# ---------------------------------------------------------------------------
# Parse OpenFOAM fields
# ---------------------------------------------------------------------------

def _parse_of_scalar_field(filepath: Path) -> np.ndarray:
    """Parse an OpenFOAM volScalarField into a 1-D numpy array."""
    text = filepath.read_text()
    match = re.search(r'internalField\s+nonuniform\s+List<scalar>\s*\n(\d+)\s*\n\(', text)
    if match:
        n = int(match.group(1))
        start = match.end()
        end = text.index(')', start)
        values = text[start:end].split()
        return np.array([float(v) for v in values[:n]])

    match = re.search(r'internalField\s+uniform\s+([\d.eE+\-]+)', text)
    if match:
        logger.warning("Scalar field %s is uniform — cannot determine N", filepath)
        return np.array([float(match.group(1))])

    raise ValueError(f"Cannot parse scalar field: {filepath}")


def _parse_of_vector_field(filepath: Path) -> np.ndarray:
    """Parse an OpenFOAM volVectorField into an (N, 3) numpy array."""
    text = filepath.read_text()
    match = re.search(r'internalField\s+nonuniform\s+List<vector>\s*\n(\d+)\s*\n\(', text)
    if not match:
        raise ValueError(f"Cannot parse vector field: {filepath}")

    n = int(match.group(1))
    start = match.end()
    end = text.index(')', start)
    vectors = re.findall(r'\(\s*([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s*\)',
                         text[start:end])
    arr = np.array([[float(x), float(y), float(z)] for x, y, z in vectors[:n]])
    return arr


# ---------------------------------------------------------------------------
# ERA5 profile interpolation
# ---------------------------------------------------------------------------

def _build_interpolators(inflow: dict):
    """Build speed, temperature, and pressure interpolators from inflow JSON.

    Returns (speed_interp, T_interp, p_interp, fd_x, fd_y, u_star, z0, ux_interp, uy_interp).
    """
    from scipy.interpolate import interp1d

    z_levels = np.array(inflow["z_levels"])
    u_profile = np.array(inflow["u_profile"])

    speed_interp = interp1d(
        z_levels, u_profile,
        kind="linear", bounds_error=False,
        fill_value=(u_profile[0], u_profile[-1]),
    )

    # Component profiles ux(z), uy(z) — wind direction varies with height
    ux_interp = None
    uy_interp = None
    if "ux_profile" in inflow and "uy_profile" in inflow:
        ux_profile = np.array(inflow["ux_profile"])
        uy_profile = np.array(inflow["uy_profile"])
        ux_interp = interp1d(
            z_levels, ux_profile,
            kind="linear", bounds_error=False,
            fill_value=(ux_profile[0], ux_profile[-1]),
        )
        uy_interp = interp1d(
            z_levels, uy_profile,
            kind="linear", bounds_error=False,
            fill_value=(uy_profile[0], uy_profile[-1]),
        )

    T_interp = None
    T_profile = inflow.get("T_profile")
    if T_profile is not None and len(T_profile) == len(z_levels):
        T_profile = np.array(T_profile)
        T_interp = interp1d(
            z_levels, T_profile,
            kind="linear", bounds_error=False,
            fill_value=(T_profile[0], T_profile[-1]),
        )

    p_interp = None
    p_profile = inflow.get("p_profile")
    if p_profile is not None and len(p_profile) == len(z_levels):
        p_profile = np.array(p_profile)
        p_interp = interp1d(
            z_levels, p_profile,
            kind="linear", bounds_error=False,
            fill_value=(p_profile[0], p_profile[-1]),
        )

    fd_x = float(inflow["flowDir_x"])
    fd_y = float(inflow["flowDir_y"])
    u_star = float(inflow["u_star"])
    z0 = float(inflow.get("z0", inflow.get("z0_eff", 0.05)))

    return speed_interp, T_interp, p_interp, fd_x, fd_y, u_star, z0, ux_interp, uy_interp


def interpolate_profiles_at_z(
    z: np.ndarray,
    speed_interp,
    T_interp,
    p_interp,
    fd_x: float,
    fd_y: float,
    u_star: float,
    z0: float,
    T_ref: float = 300.0,
    is_bbsf: bool = False,
    ux_interp=None,
    uy_interp=None,
) -> dict[str, np.ndarray]:
    """Compute U, k, epsilon, T, p_rgh at given heights z.

    Parameters
    ----------
    z : (N,) heights above datum [m].
    is_bbsf : if True, compute p_rgh in static form (Pa) for BBSF.
    ux_interp, uy_interp : optional interpolators for height-varying wind components.
        If provided, wind direction varies with height (ERA5 Ekman spiral).
        If None, falls back to speed × (fd_x, fd_y) uniform direction.

    Returns
    -------
    dict with 'U' (N,3), 'k' (N,), 'epsilon' (N,), 'T' (N,), 'p_rgh' (N,).
    """
    n = len(z)
    z = np.maximum(z, 0.1)

    U = np.zeros((n, 3))
    if ux_interp is not None and uy_interp is not None:
        # Height-varying wind direction from ERA5 components
        U[:, 0] = ux_interp(z)
        U[:, 1] = uy_interp(z)
    else:
        # Fallback: uniform direction (old behaviour)
        speed = np.maximum(speed_interp(z), 0.0)
        U[:, 0] = speed * fd_x
        U[:, 1] = speed * fd_y

    k = np.full(n, u_star**2 / CMU**0.5)
    # Epsilon: UNIFORM value consistent with epsilonWallFunction.
    # The wall function computes eps_wall = Cmu^0.75 * k^1.5 / (kappa * y_wall).
    # Using log-law eps(z) creates a violent mismatch at iter 2 when the wall
    # function overrides → eps drops 40× → eps/k ratio crashes → Pε explodes.
    # y_wall ≈ first BL cell height (cfMesh maxFirstLayerThickness).
    _y_wall = 10.0  # m — matches BL_FIRST_LAYER_M in generate_mesh.py
    _k_scalar = float(k[0])
    _eps_uniform = CMU**0.75 * _k_scalar**1.5 / (KAPPA * _y_wall)
    epsilon = np.full(n, _eps_uniform)
    logger.info("epsilon init = %.6e m²/s³ (uniform, y_wall=%.1f m, k=%.4f)",
                _eps_uniform, _y_wall, _k_scalar)

    if T_interp is not None:
        T = T_interp(z)
    else:
        T = np.full(n, T_ref)

    # p_rgh is a Lagrange multiplier ≈ 0 in Boussinesq.
    # Do NOT initialise with ERA5 pressure (creates non-Boussinesq gradient).
    p_rgh = np.zeros(n)

    return {"U": U, "k": k, "epsilon": epsilon, "T": T, "p_rgh": p_rgh}


# ---------------------------------------------------------------------------
# Patch internalField (existing logic, improved regex)
# ---------------------------------------------------------------------------

def _patch_internal_field_vector(filepath: Path, data: np.ndarray) -> None:
    """Replace internalField in an existing OF volVectorField with nonuniform data."""
    text = filepath.read_text()
    n = len(data)

    values = '\n'.join(f'({data[i, 0]:.6f} {data[i, 1]:.6f} {data[i, 2]:.6f})'
                       for i in range(n))
    replacement = f'internalField   nonuniform List<vector>\n{n}\n(\n{values}\n)\n;'

    text = re.sub(
        r'internalField\s+(?:uniform\s+\([^)]+\)|nonuniform\s+List<vector>\s*\n\d+\s*\n\(.*?\)\n)\s*;',
        replacement, text, count=1, flags=re.DOTALL,
    )
    filepath.write_text(text)
    logger.info("Patched internalField %s: %d cells", filepath.name, n)


def _patch_internal_field_scalar(filepath: Path, data: np.ndarray) -> None:
    """Replace internalField in an existing OF volScalarField with nonuniform data."""
    text = filepath.read_text()
    n = len(data)

    values = '\n'.join(f'{data[i]:.6e}' for i in range(n))
    replacement = f'internalField   nonuniform List<scalar>\n{n}\n(\n{values}\n)\n;'

    text = re.sub(
        r'internalField\s+(?:uniform\s+[\d.eE+\-]+|nonuniform\s+List<scalar>\s*\n\d+\s*\n\(.*?\)\n)\s*;',
        replacement, text, count=1, flags=re.DOTALL,
    )
    filepath.write_text(text)
    logger.info("Patched internalField %s: %d cells", filepath.name, n)


# ---------------------------------------------------------------------------
# Patch boundaryField (inletValue + value → nonuniform)
# ---------------------------------------------------------------------------

def _format_nonuniform_vector(data: np.ndarray) -> str:
    """Format a nonuniform List<vector> string."""
    n = len(data)
    values = '\n'.join(f'({data[i, 0]:.6f} {data[i, 1]:.6f} {data[i, 2]:.6f})'
                       for i in range(n))
    return f'nonuniform List<vector>\n{n}\n(\n{values}\n)'


def _format_nonuniform_scalar(data: np.ndarray) -> str:
    """Format a nonuniform List<scalar> string."""
    n = len(data)
    values = '\n'.join(f'{data[i]:.6e}' for i in range(n))
    return f'nonuniform List<scalar>\n{n}\n(\n{values}\n)'


def _patch_boundary_values(
    filepath: Path,
    patch_data: dict[str, np.ndarray],
    field_type: str,
) -> None:
    """Patch inletValue and value in boundaryField for inletOutlet patches.

    Parameters
    ----------
    filepath : Path to the OF field file (e.g. 0/U).
    patch_data : dict mapping patch_name → array of per-face values.
    field_type : "vector" or "scalar".
    """
    text = filepath.read_text()
    format_fn = _format_nonuniform_vector if field_type == "vector" else _format_nonuniform_scalar

    patched_count = 0
    for patch_name, data in patch_data.items():
        # Find the patch block in boundaryField
        # Pattern: patch_name { ... type inletOutlet; ... inletValue uniform ...; ... value uniform ...; }
        patch_pattern = re.compile(
            rf'(\b{re.escape(patch_name)}\s*\{{)(.*?)(\}})',
            re.DOTALL,
        )
        match = patch_pattern.search(text)
        if not match:
            logger.debug("Patch %s not found in %s — skipping", patch_name, filepath.name)
            continue

        block = match.group(2)

        # Check BC type — only patch inletOutlet / outletInlet
        type_match = re.search(r'type\s+(\w+)', block)
        if not type_match or type_match.group(1) not in PATCHABLE_BC_TYPES:
            logger.debug("Patch %s has type %s — skipping",
                        patch_name, type_match.group(1) if type_match else "unknown")
            continue

        bc_type = type_match.group(1)
        nonuniform_str = format_fn(data)

        # Determine the keyword: inletValue for inletOutlet, outletValue for outletInlet
        val_keyword = "outletValue" if bc_type == "outletInlet" else "inletValue"

        # Patch inletValue/outletValue + value
        if field_type == "vector":
            block = re.sub(
                rf'{val_keyword}\s+uniform\s+\([^)]+\)',
                f'{val_keyword}      {nonuniform_str}',
                block, count=1,
            )
            block = re.sub(
                r'(\n\s+value)\s+uniform\s+\([^)]+\)',
                rf'\1           {nonuniform_str}',
                block, count=1,
            )
        else:
            block = re.sub(
                rf'{val_keyword}\s+uniform\s+[\d.eE+\-]+',
                f'{val_keyword}      {nonuniform_str}',
                block, count=1,
            )
            block = re.sub(
                r'(\n\s+value)\s+uniform\s+[\d.eE+\-]+',
                rf'\1           {nonuniform_str}',
                block, count=1,
            )

        text = text[:match.start(2)] + block + text[match.end(2):]
        patched_count += 1

    filepath.write_text(text)
    logger.info("Patched boundaryField %s: %d patches", filepath.name, patched_count)


# ---------------------------------------------------------------------------
# Solver detection
# ---------------------------------------------------------------------------

def detect_solver(case_dir: Path) -> str:
    """Read application name from system/controlDict."""
    cd_path = case_dir / "system" / "controlDict"
    if not cd_path.exists():
        return "simpleFoam"
    text = cd_path.read_text()
    match = re.search(r'application\s+(\w+)', text)
    return match.group(1) if match else "simpleFoam"


# ---------------------------------------------------------------------------
# Write constant/boundaryData for MappedFile BCs
# ---------------------------------------------------------------------------

def _write_of_points(filepath: Path, points: np.ndarray) -> None:
    """Write an OpenFOAM points file for MappedFile (raw format, no FoamFile header)."""
    n = len(points)
    lines = [f'{n}', '(']
    for i in range(n):
        lines.append(f'({points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f})')
    lines.append(')')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text('\n'.join(lines))


def _write_of_mapped_vector(filepath: Path, data: np.ndarray) -> None:
    """Write a vector field for MappedFile boundaryData (raw format)."""
    n = len(data)
    lines = [f'{n}', '(']
    for i in range(n):
        lines.append(f'({data[i, 0]:.6f} {data[i, 1]:.6f} {data[i, 2]:.6f})')
    lines.append(')')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text('\n'.join(lines))


def _write_of_mapped_scalar(filepath: Path, data: np.ndarray) -> None:
    """Write a scalar field for MappedFile boundaryData (raw format)."""
    n = len(data)
    lines = [f'{n}', '(']
    for i in range(n):
        lines.append(f'{data[i]:.6e}')
    lines.append(')')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text('\n'.join(lines))


def write_boundary_data(
    case_dir: Path,
    boundary_faces: dict[str, np.ndarray],
    patch_fields: dict[str, dict[str, np.ndarray]],
) -> None:
    """Write constant/boundaryData/<patch>/points and 0/<field> for each patch.

    Parameters
    ----------
    case_dir : OpenFOAM case directory.
    boundary_faces : dict mapping patch_name → (nFaces, 3) face centres.
    patch_fields : dict mapping patch_name → {field_name: data_array}.
    """
    bd_root = case_dir / "constant" / "boundaryData"

    for patch_name, fields in patch_fields.items():
        if patch_name not in boundary_faces:
            continue
        face_centres = boundary_faces[patch_name]
        if len(face_centres) == 0:
            continue

        patch_dir = bd_root / patch_name

        # Write points
        _write_of_points(patch_dir / "points", face_centres)

        # Write each field at time=0
        for field_name, data in fields.items():
            time_dir = patch_dir / "0"
            if data.ndim == 2 and data.shape[1] == 3:
                _write_of_mapped_vector(time_dir / field_name, data)
            else:
                _write_of_mapped_scalar(time_dir / field_name, data)

        logger.info("Wrote boundaryData for patch %s: %d faces, fields %s",
                     patch_name, len(face_centres), list(fields.keys()))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def init_from_era5(
    case_dir: Path | str,
    inflow_json: Path | str,
    neutral_T_init: bool = False,
) -> None:
    """Initialise OpenFOAM fields from ERA5 interpolation.

    - internalField → nonuniform List (per-cell interpolation)
    - constant/boundaryData/<patch>/ → MappedFile data (per-face profiles)

    Parameters
    ----------
    case_dir : OpenFOAM case directory (must have mesh already generated).
    inflow_json : Path to inflow profile JSON (from prepare_inflow.py).
    neutral_T_init : if True, skip T internalField patch (keep template uniform
        T_ref). BCs still get stratified ERA5 profile via boundaryData.
        Use for BBSF: avoids large buoyancy at iter 1 that crashes k-epsilon.
    """
    case_dir = Path(case_dir)

    with open(inflow_json) as f:
        inflow = json.load(f)

    T_ref_surface = float(inflow.get("T_ref", 300.0))

    # Detect solver for p_rgh formulation
    solver = detect_solver(case_dir)
    is_bbsf = "boussinesq" in solver.lower()
    logger.info("Solver: %s (BBSF=%s)", solver, is_bbsf)

    # Build interpolators once
    speed_interp, T_interp, p_interp, fd_x, fd_y, u_star, z0, ux_interp, uy_interp = _build_interpolators(inflow)

    # Compute T_ref as volume-average of ERA5 T profile over domain height.
    # ERA5 levels are regularly spaced in pressure → uniform weight for average.
    # Using T at surface (T_ref_surface) causes a systematic negative buoyancy
    # force above ground (T(z) < T_ref everywhere → air "too cold" → Uz < 0).
    T_ref = T_ref_surface  # fallback
    if T_interp is not None and "z_levels" in inflow:
        z_era5 = np.array(inflow["z_levels"])
        T_era5 = np.array(inflow["T_profile"])
        # Domain top: read from mesh bounding box or estimate from controlDict
        # For now, use the max z_level within 5000 m (typical domain height)
        z_domain_top = 5000.0  # will be refined below with actual cell centres
        mask = z_era5 <= z_domain_top
        if mask.sum() > 2:
            T_ref = float(np.trapz(T_era5[mask], z_era5[mask]) / (z_era5[mask][-1] - z_era5[mask][0]))
        logger.info("T_ref: surface=%.2f K → volume-average(0-%dm)=%.2f K (Δ=%.1f K)",
                    T_ref_surface, z_domain_top, T_ref, T_ref_surface - T_ref)

    # ---- Internal field (cell centres) ----
    logger.info("Reading cell centres from %s", case_dir)
    centres = read_cell_centres(case_dir)
    logger.info("Found %d cell centres", len(centres))

    # Refine T_ref with actual domain height from cell centres
    if T_interp is not None and "z_levels" in inflow:
        z_domain_top_actual = float(centres[:, 2].max())
        z_era5 = np.array(inflow["z_levels"])
        T_era5 = np.array(inflow["T_profile"])
        mask = z_era5 <= z_domain_top_actual
        if mask.sum() > 2:
            T_ref = float(np.trapz(T_era5[mask], z_era5[mask]) / (z_era5[mask][-1] - z_era5[mask][0]))
        logger.info("T_ref refined with z_top=%.0f m: T_ref=%.2f K", z_domain_top_actual, T_ref)

        # Update transportProperties with corrected T_ref
        tp_path = case_dir / "constant" / "transportProperties"
        if tp_path.exists():
            tp_text = tp_path.read_text()
            import re as _re
            tp_text = _re.sub(
                r'(TRef\s+\[.*?\]\s+)[\d.]+',
                lambda m: m.group(1) + f"{T_ref:.2f}",
                tp_text,
            )
            # Also update beta = 1/T_ref
            beta_new = 1.0 / T_ref
            tp_text = _re.sub(
                r'(beta\s+\[.*?\]\s+)[\d.e+-]+',
                lambda m: m.group(1) + f"{beta_new:.6e}",
                tp_text,
            )
            tp_path.write_text(tp_text)
            logger.info("Updated transportProperties: TRef=%.2f K, beta=%.6e K⁻¹", T_ref, beta_new)

    cell_fields = interpolate_profiles_at_z(
        centres[:, 2], speed_interp, T_interp, p_interp,
        fd_x, fd_y, u_star, z0, T_ref, is_bbsf=is_bbsf,
        ux_interp=ux_interp, uy_interp=uy_interp,
    )

    u_path = case_dir / "0" / "U"
    k_path = case_dir / "0" / "k"
    epsilon_path = case_dir / "0" / "epsilon"
    t_path = case_dir / "0" / "T"
    p_path = case_dir / "0" / "p_rgh"

    _patch_internal_field_vector(u_path, cell_fields["U"])
    _patch_internal_field_scalar(k_path, cell_fields["k"])
    _patch_internal_field_scalar(epsilon_path, cell_fields["epsilon"])
    if t_path.exists():
        if neutral_T_init:
            # Keep template uniform T_ref — BBSF neutral spin-up strategy.
            # Stratified T from ERA5 activates full buoyancy at iter 1 → k-ε crash.
            # BCs (boundaryData) still receive stratified profile below.
            logger.info("neutral_T_init=True: skipping T internalField patch (keeping uniform T_ref)")
        else:
            _patch_internal_field_scalar(t_path, cell_fields["T"])
    if p_path.exists():
        _patch_internal_field_scalar(p_path, cell_fields["p_rgh"])

    # ---- Boundary data for MappedFile BCs ----
    logger.info("Reading boundary face centres from %s", case_dir)
    boundary_faces = read_boundary_face_centres(case_dir)

    # Auto-detect lateral patches (cylindrical → "lateral"; box → cardinal four)
    lateral_patches = detect_lateral_patches(boundary_faces)

    # Compute profiles at each patch's face centres and write boundaryData
    patch_fields: dict[str, dict[str, np.ndarray]] = {}

    for patch_name, face_centres in boundary_faces.items():
        if patch_name not in lateral_patches:
            continue
        if len(face_centres) == 0:
            continue

        pf = interpolate_profiles_at_z(
            face_centres[:, 2], speed_interp, T_interp, p_interp,
            fd_x, fd_y, u_star, z0, T_ref, is_bbsf=is_bbsf,
            ux_interp=ux_interp, uy_interp=uy_interp,
        )

        fields = {
            "U": pf["U"],
            "k": pf["k"],
            "epsilon": pf["epsilon"],
            "p_rgh": pf["p_rgh"],
        }
        if t_path.exists():
            fields["T"] = pf["T"]

        patch_fields[patch_name] = fields

    write_boundary_data(case_dir, boundary_faces, patch_fields)

    logger.info("ERA5 initialisation complete for %s (internal + boundaryData)", case_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Initialise OpenFOAM fields from ERA5 interpolation"
    )
    parser.add_argument("--case-dir", required=True, help="OpenFOAM case directory")
    parser.add_argument("--inflow", required=True, help="Inflow profile JSON")
    parser.add_argument("--neutral-T-init", action="store_true",
                        help="Keep T internalField as uniform T_ref (BBSF neutral spin-up)")
    args = parser.parse_args()

    init_from_era5(
        case_dir=Path(args.case_dir),
        inflow_json=Path(args.inflow),
        neutral_T_init=args.neutral_T_init,
    )
