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

# numpy 2.x renamed trapz → trapezoid
_trapz = getattr(np, "trapezoid", None) or np.trapz

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

    Only reads the boundary faces (not all 6M+ internal faces) for speed.

    Returns
    -------
    dict mapping patch_name → (nFaces, 3) array of face centre coordinates [m].
    """
    poly = case_dir / "constant" / "polyMesh"

    # --- Read points (binary-safe: parse the coordinate block only) ---
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

    # --- Read boundary patches ---
    patches = read_boundary_info(case_dir)

    # Only need boundary faces: skip 6M+ internal faces entirely.
    min_start = min(p["startFace"] for p in patches.values())
    max_end = max(p["startFace"] + p["nFaces"] for p in patches.values())
    n_boundary_faces = max_end - min_start

    faces_path = poly / "faces"
    boundary_faces = []
    face_idx = 0
    with open(faces_path) as f:
        # Skip header until we find the count line "N\n(\n"
        in_list = False
        for line in f:
            stripped = line.strip()
            if not in_list:
                if stripped == "(":
                    in_list = True
                continue
            if stripped == ")":
                break
            if face_idx < min_start:
                face_idx += 1
                continue
            if face_idx >= max_end:
                break
            # Parse face: "4(v0 v1 v2 v3)" or "4 (v0 v1 v2 v3)"
            m = re.match(r'\d+\(([^)]+)\)', stripped)
            if m:
                boundary_faces.append([int(v) for v in m.group(1).split()])
            face_idx += 1

    logger.debug("Read %d boundary faces (skipped %d internal)", len(boundary_faces), min_start)

    result = {}
    for patch_name, info in patches.items():
        start = info["startFace"] - min_start
        n = info["nFaces"]
        centres = np.zeros((n, 3))
        for i in range(n):
            idx = start + i
            if idx < len(boundary_faces):
                verts = boundary_faces[idx]
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

    q_interp = None
    q_profile = inflow.get("q_profile")
    if q_profile is not None and len(q_profile) == len(z_levels):
        q_profile = np.array(q_profile)
        q_interp = interp1d(
            z_levels, q_profile,
            kind="linear", bounds_error=False,
            fill_value=(q_profile[0], q_profile[-1]),
        )

    fd_x = float(inflow["flowDir_x"])
    fd_y = float(inflow["flowDir_y"])
    u_star = float(inflow["u_star"])
    z0 = float(inflow.get("z0", inflow.get("z0_eff", 0.05)))

    return speed_interp, T_interp, p_interp, fd_x, fd_y, u_star, z0, ux_interp, uy_interp, q_interp


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
    q_interp=None,
) -> dict[str, np.ndarray]:
    """Compute U, k, epsilon, T, q, p_rgh at given heights z.

    Parameters
    ----------
    z : (N,) heights above datum [m].
    is_bbsf : if True, compute p_rgh in static form (Pa) for BBSF.
    ux_interp, uy_interp : optional interpolators for height-varying wind components.
        If provided, wind direction varies with height (ERA5 Ekman spiral).
        If None, falls back to speed × (fd_x, fd_y) uniform direction.
    q_interp : optional interpolator for specific humidity q(z).

    Returns
    -------
    dict with 'U' (N,3), 'k' (N,), 'epsilon' (N,), 'T' (N,), 'q' (N,), 'p_rgh' (N,).
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

    # Specific humidity q(z)
    if q_interp is not None:
        q = np.maximum(q_interp(z), 0.0)  # q ≥ 0 always
    else:
        q = None

    # p_rgh is a Lagrange multiplier ≈ 0 in Boussinesq.
    # Do NOT initialise with ERA5 pressure (creates non-Boussinesq gradient).
    p_rgh = np.zeros(n)

    result = {"U": U, "k": k, "epsilon": epsilon, "T": T, "p_rgh": p_rgh}
    if q is not None:
        result["q"] = q
    return result


# ---------------------------------------------------------------------------
# 3D lateral BC: bilinear(lat, lon) + PCHIP(z) + log-law blend
# ---------------------------------------------------------------------------

# Blend window (must match prepare_inflow.py Z_BLEND_LOW / Z_BLEND_HIGH)
Z_BLEND_LOW  = 50.0    # m — below: pure log-law
Z_BLEND_HIGH = 250.0   # m — above: pure ERA5
G_GEO        = 9.80665 # m/s² — gravity used to build z_geo = geopotential / g


def _e_sat_pa(T_K: np.ndarray) -> np.ndarray:
    """Saturation vapour pressure over water [Pa], Bolton 1980 formula."""
    Tc = T_K - 273.15
    return 611.2 * np.exp(17.67 * Tc / (Tc + 243.5))


def _d2m_to_q(d2m_K: float, p_surf_Pa: float = 101325.0) -> float:
    """Convert 2-m dew-point temperature to specific humidity.

    q = 0.622 · e_sat(Td) / (p - 0.378 · e_sat(Td))
    """
    e = float(_e_sat_pa(np.asarray(d2m_K)))
    return 0.622 * e / max(p_surf_Pa - 0.378 * e, 1.0)


def _build_corner_pchip(era5_grid: dict, field: str, p_surf_Pa: float = 101325.0):
    """Build an (N, N) array of PchipInterpolator instances for a field.

    Anchors surface values per corner when relevant:
        - T : prepend (z=2m, t2m)
        - q : prepend (z=2m, q_from_d2m)
    u, v, p use raw pressure-level columns (log-law blend handles near-surface).

    Parameters
    ----------
    era5_grid : dict
        Block produced by ``extract_era5_grid_block`` in prepare_inflow.
    field : str
        One of ``u, v, T, q``.
    p_surf_Pa : float
        Surface pressure [Pa], used only to convert d2m → q.

    Returns
    -------
    interps : ndarray of PchipInterpolator, shape (N, N)
    """
    from scipy.interpolate import PchipInterpolator

    z_geo = np.asarray(era5_grid["z_geo"])   # (N, N, L)
    data  = np.asarray(era5_grid[field])     # (N, N, L)
    N = z_geo.shape[0]
    interps = np.empty((N, N), dtype=object)

    for i in range(N):
        for j in range(N):
            z_col = np.asarray(z_geo[i, j, :], dtype=float)
            v_col = np.asarray(data[i, j, :], dtype=float)
            order = np.argsort(z_col)
            z_s = z_col[order]
            v_s = v_col[order]

            # Anchor surface for T and q
            if field == "T" and "t2m" in era5_grid:
                t2m_val = float(np.asarray(era5_grid["t2m"])[i, j])
                z_s = np.concatenate([[2.0], z_s])
                v_s = np.concatenate([[t2m_val], v_s])
            elif field == "q" and "d2m" in era5_grid:
                d2m_val = float(np.asarray(era5_grid["d2m"])[i, j])
                q2m = _d2m_to_q(d2m_val, p_surf_Pa)
                z_s = np.concatenate([[2.0], z_s])
                v_s = np.concatenate([[q2m], v_s])

            # Enforce strict monotonic increase (PCHIP requirement)
            for m in range(1, len(z_s)):
                if z_s[m] <= z_s[m - 1]:
                    z_s[m] = z_s[m - 1] + 1e-3

            interps[i, j] = PchipInterpolator(z_s, v_s, extrapolate=True)

    return interps


def interpolate_profiles_at_xyz(
    face_centres: np.ndarray,
    era5_grid: dict,
    site_lat: float,
    site_lon: float,
    u_star: float,
    z0: float,
    fd_x: float,
    fd_y: float,
    T_ref: float = 300.0,
    is_bbsf: bool = False,
    p_surf_Pa: float = 101325.0,
    site_ground_elev_m: float | None = None,
) -> dict[str, np.ndarray]:
    """Per-face BC: bilinear(lat, lon) + PCHIP(z) + log-law blend under 250 m.

    Wind strategy (see feedback BC 3D — 2026-04-21):
      - Below Z_BLEND_LOW: pure log-law with site-scalar u_star, z0, oriented
        by site-scalar (fd_x, fd_y).
      - Above Z_BLEND_HIGH: pure ERA5 u, v components (direction varies in
        space → Ekman spiral preserved).
      - In the blend window: smootherstep weighting, component-wise on ux/uy.
    T, q use raw ERA5 PCHIP with t2m/d2m surface anchor per corner.

    Parameters
    ----------
    face_centres : (N, 3) array
        Boundary face centres in mesh-local coords. Site is at (x=0, y=0).
    era5_grid : dict
        3×3 block from ``extract_era5_grid_block`` embedded in inflow JSON.
    site_lat, site_lon : float
        Site coordinates (origin of the local frame).
    u_star, z0 : float
        Scalar (site-level) friction velocity [m/s] and roughness [m].
    fd_x, fd_y : float
        Unit vector of the *site-level* inflow direction.
    T_ref, is_bbsf, p_surf_Pa : as in ``interpolate_profiles_at_z``.

    Returns
    -------
    dict with keys 'U' (N,3), 'k' (N,), 'epsilon' (N,), 'T' (N,), 'q' (N,)?, 'p_rgh' (N,).
    """
    lats = np.asarray(era5_grid["lats"], dtype=float)   # (M,) ascending
    lons = np.asarray(era5_grid["lons"], dtype=float)
    M = len(lats)
    if M < 2 or len(lons) < 2:
        raise ValueError("era5_grid must have at least 2×2 corners")
    has_q = "q" in era5_grid and era5_grid["q"] is not None

    interps = {
        "u": _build_corner_pchip(era5_grid, "u", p_surf_Pa),
        "v": _build_corner_pchip(era5_grid, "v", p_surf_Pa),
        "T": _build_corner_pchip(era5_grid, "T", p_surf_Pa),
    }
    if has_q:
        interps["q"] = _build_corner_pchip(era5_grid, "q", p_surf_Pa)

    # (x, y) → (lat, lon) via first-order flat-earth approximation
    DEG_PER_M_LAT = 1.0 / 111_000.0
    DEG_PER_M_LON = 1.0 / (111_000.0 * float(np.cos(np.radians(site_lat))))

    face_lat = site_lat + face_centres[:, 1] * DEG_PER_M_LAT
    face_lon = site_lon + face_centres[:, 0] * DEG_PER_M_LON
    face_z   = np.maximum(face_centres[:, 2], 0.1)
    N_faces  = len(face_centres)

    # Enclosing 2×2 sub-cell in the lat/lon grid
    def cell_index(grid: np.ndarray, x: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(grid, x) - 1
        return np.clip(idx, 0, len(grid) - 2)

    i_lat = cell_index(lats, face_lat)  # (N_faces,)
    j_lon = cell_index(lons, face_lon)

    face_lat_c = np.clip(face_lat, lats[0], lats[-1])
    face_lon_c = np.clip(face_lon, lons[0], lons[-1])

    w_lat = (face_lat_c - lats[i_lat]) / (lats[i_lat + 1] - lats[i_lat])
    w_lon = (face_lon_c - lons[j_lon]) / (lons[j_lon + 1] - lons[j_lon])

    # Evaluate every corner's PCHIP at every face_z, then bilinear-combine.
    # (M, M, N_faces) is ~9 × N_faces floats — cheap.
    n_faces_idx = np.arange(N_faces)

    def combine(field: str) -> np.ndarray:
        all_corners = np.empty((M, M, N_faces), dtype=float)
        for i in range(M):
            for j in range(M):
                all_corners[i, j, :] = interps[field][i, j](face_z)
        a00 = all_corners[i_lat,     j_lon,     n_faces_idx]
        a01 = all_corners[i_lat,     j_lon + 1, n_faces_idx]
        a10 = all_corners[i_lat + 1, j_lon,     n_faces_idx]
        a11 = all_corners[i_lat + 1, j_lon + 1, n_faces_idx]
        return ((1 - w_lat) * (1 - w_lon) * a00
                + (1 - w_lat) * w_lon       * a01
                + w_lat       * (1 - w_lon) * a10
                + w_lat       * w_lon       * a11)

    ux_era5 = combine("u")
    uy_era5 = combine("v")
    T_face  = combine("T")
    q_face  = combine("q") if has_q else None

    # Log-law + blend are defined in AGL (z above ground), not MSL.
    # The cylinder lateral surface has a vertical STACK of faces at each angular
    # position (the mesh has cells_z faces per column). Local ground at a given
    # (x, y) is the minimum z of THAT column. This matters on steep terrain
    # where the cylinder floor varies by hundreds of meters between angles.
    # Prior versions used face_z (MSL) directly which injected a spurious
    # ~5 m/s log-law contribution at the first lateral row and caused a
    # non-monotonic |U| "saut" in the blend zone.
    xy_key = np.round(face_centres[:, :2] / 5.0).astype(np.int64)
    # Group face indices by (x, y) bin to find the per-column z_min
    z_min_col = np.full(N_faces, np.inf)
    # np.unique with return_inverse gives a compact column index
    _, inv = np.unique(xy_key, axis=0, return_inverse=True)
    n_cols = int(inv.max()) + 1
    z_by_col = np.full(n_cols, np.inf)
    for k in range(N_faces):
        if face_centres[k, 2] < z_by_col[inv[k]]:
            z_by_col[inv[k]] = face_centres[k, 2]
    z_min_col = z_by_col[inv]
    # Fallback to supplied site_ground_elev_m if column grouping failed
    if site_ground_elev_m is not None:
        # If too few columns were detected (e.g. degenerate), fall back
        if n_cols < 4:
            z_min_col = np.full(N_faces, float(site_ground_elev_m))
    z_agl = np.maximum(face_z - z_min_col, 0.1)

    # Log-law (site-scalar u_star, z0) blended component-wise with ERA5
    speed_log = np.maximum((u_star / KAPPA) * np.log((z_agl + z0) / z0), 0.0)

    t = np.clip((z_agl - Z_BLEND_LOW) / (Z_BLEND_HIGH - Z_BLEND_LOW), 0.0, 1.0)
    alpha = t * t * t * (t * (t * 6.0 - 15.0) + 10.0)  # smootherstep

    U = np.zeros((N_faces, 3))
    U[:, 0] = (1.0 - alpha) * speed_log * fd_x + alpha * ux_era5
    U[:, 1] = (1.0 - alpha) * speed_log * fd_y + alpha * uy_era5

    # k uniform; epsilon consistent with epsilonWallFunction (matches _z version)
    k_val = u_star ** 2 / CMU ** 0.5
    k = np.full(N_faces, k_val)
    _y_wall = 10.0
    eps_val = CMU ** 0.75 * k_val ** 1.5 / (KAPPA * _y_wall)
    epsilon = np.full(N_faces, eps_val)

    p_rgh = np.zeros(N_faces)  # Lagrange multiplier in Boussinesq

    out = {"U": U, "k": k, "epsilon": epsilon, "T": T_face, "p_rgh": p_rgh}
    if q_face is not None:
        out["q"] = np.maximum(q_face, 0.0)
    return out


# ---------------------------------------------------------------------------
# 3D AGL profile block (consumed by training export — Wind9kDataset schema)
# ---------------------------------------------------------------------------

def build_era5_3d_agl(
    era5_grid: dict,
    z_levels_agl: np.ndarray,
    site_ground_elev_m: float,
    u_star: float,
    z0: float,
    fd_x: float,
    fd_y: float,
    p_surf_Pa: float = 101325.0,
) -> dict[str, np.ndarray]:
    """Per-corner AGL profiles for the training grid.zarr `input/era5_3d/`.

    Produces 3×3 × len(z_levels_agl) arrays of u, v, T, q, k matching the
    schema expected by ``services/module2b-surrogate/src/dataset_wind9k.py``.
    Per corner, the surface layer is log-law with site-scalar ``u_star`` and
    ``z0`` (component-wise blend with ERA5 between Z_BLEND_LOW and
    Z_BLEND_HIGH), matching `interpolate_profiles_at_xyz`.

    The AGL reference is taken at the site (``site_ground_elev_m``) for all
    9 corners. This is an approximation — the true AGL of a corner depends on
    that corner's own terrain elevation, but ERA5 ground elevation varies by
    <200 m over 25 km which is small relative to the log-spaced 5 m–5 km grid.

    Returns
    -------
    dict with float32 arrays:
        lat, lon : (N,)
        u, v, T, q, k : (N, N, len(z_levels_agl))
    where N is len(era5_grid["lats"]) (typically 3).
    """
    z_levels_agl = np.asarray(z_levels_agl, dtype=np.float64)
    n_z = len(z_levels_agl)
    lats = np.asarray(era5_grid["lats"], dtype=np.float32)
    lons = np.asarray(era5_grid["lons"], dtype=np.float32)
    M = len(lats)
    has_q = "q" in era5_grid and era5_grid["q"] is not None

    # Build per-corner PCHIP with surface anchors (uses z_agl = z_geo - site_ground)
    interps_uv: dict[str, np.ndarray] = {"u": np.empty((M, M), dtype=object),
                                         "v": np.empty((M, M), dtype=object)}
    interps_Tq: dict[str, np.ndarray] = {"T": np.empty((M, M), dtype=object)}
    if has_q:
        interps_Tq["q"] = np.empty((M, M), dtype=object)

    from scipy.interpolate import PchipInterpolator

    z_geo = np.asarray(era5_grid["z_geo"], dtype=np.float64)  # (M, M, L)
    u_raw = np.asarray(era5_grid["u"], dtype=np.float64)
    v_raw = np.asarray(era5_grid["v"], dtype=np.float64)
    T_raw = np.asarray(era5_grid["T"], dtype=np.float64)
    q_raw = np.asarray(era5_grid["q"], dtype=np.float64) if has_q else None
    t2m = np.asarray(era5_grid.get("t2m", np.full((M, M), np.nan)), dtype=np.float64)
    d2m = np.asarray(era5_grid.get("d2m", np.full((M, M), np.nan)), dtype=np.float64)

    def _strict_mono(z: np.ndarray) -> np.ndarray:
        z = z.copy()
        for m in range(1, len(z)):
            if z[m] <= z[m - 1]:
                z[m] = z[m - 1] + 1e-3
        return z

    # Minimum AGL above which ERA5 pressure-level data is kept (below: drop).
    # ERA5 extrapolates below-ground levels to unphysical cold / zero-wind
    # values — if we let those through the PCHIP, they corrupt the anchored
    # surface value (t2m at z=2 would get "interpolated" between 297K and
    # the below-ground extrapolated ~270K level squeezed next to it by
    # strict-monotonic clamping).
    Z_ERA5_MIN_AGL = 3.0  # m, > anchor z=2m

    for i in range(M):
        for j in range(M):
            z_agl = z_geo[i, j, :] - site_ground_elev_m
            order = np.argsort(z_agl)
            z_sorted = z_agl[order]
            valid = z_sorted > Z_ERA5_MIN_AGL
            if valid.sum() < 2:
                # Degenerate column (ERA5 entirely below-ground under the site).
                # Clamp to 2 highest levels so PCHIP still builds.
                valid = np.zeros_like(valid)
                valid[-2:] = True
            z_s = z_sorted[valid]
            u_s = u_raw[i, j, order][valid]
            v_s = v_raw[i, j, order][valid]
            T_s_raw = T_raw[i, j, order][valid]

            # u, v : no surface anchor (log-law blend overrides below 250 m)
            interps_uv["u"][i, j] = PchipInterpolator(
                _strict_mono(z_s), u_s, extrapolate=True)
            interps_uv["v"][i, j] = PchipInterpolator(
                _strict_mono(z_s), v_s, extrapolate=True)

            # T with t2m anchor at z=2m
            if np.isfinite(t2m[i, j]):
                z_s_T = np.concatenate([[2.0], z_s])
                T_s = np.concatenate([[t2m[i, j]], T_s_raw])
            else:
                z_s_T, T_s = z_s, T_s_raw
            interps_Tq["T"][i, j] = PchipInterpolator(
                _strict_mono(z_s_T), T_s, extrapolate=True)

            if has_q:
                q_s_raw = q_raw[i, j, order][valid]
                if np.isfinite(d2m[i, j]):
                    q2m = _d2m_to_q(float(d2m[i, j]), p_surf_Pa)
                    z_s_q = np.concatenate([[2.0], z_s])
                    q_s = np.concatenate([[q2m], q_s_raw])
                else:
                    z_s_q, q_s = z_s, q_s_raw
                interps_Tq["q"][i, j] = PchipInterpolator(
                    _strict_mono(z_s_q), q_s, extrapolate=True)

    # Evaluate per-corner + apply log-law blend (same as interpolate_profiles_at_xyz)
    speed_log = np.maximum(
        (u_star / KAPPA) * np.log((z_levels_agl + z0) / z0), 0.0)
    t_blend = np.clip(
        (z_levels_agl - Z_BLEND_LOW) / (Z_BLEND_HIGH - Z_BLEND_LOW), 0.0, 1.0)
    alpha = t_blend * t_blend * t_blend * (t_blend * (t_blend * 6.0 - 15.0) + 10.0)

    out_u = np.zeros((M, M, n_z), dtype=np.float32)
    out_v = np.zeros((M, M, n_z), dtype=np.float32)
    out_T = np.zeros((M, M, n_z), dtype=np.float32)
    out_q = np.zeros((M, M, n_z), dtype=np.float32)

    for i in range(M):
        for j in range(M):
            ux_era5 = interps_uv["u"][i, j](z_levels_agl)
            uy_era5 = interps_uv["v"][i, j](z_levels_agl)
            out_u[i, j, :] = ((1.0 - alpha) * speed_log * fd_x + alpha * ux_era5).astype(np.float32)
            out_v[i, j, :] = ((1.0 - alpha) * speed_log * fd_y + alpha * uy_era5).astype(np.float32)
            out_T[i, j, :] = interps_Tq["T"][i, j](z_levels_agl).astype(np.float32)
            if has_q:
                out_q[i, j, :] = np.maximum(
                    interps_Tq["q"][i, j](z_levels_agl), 0.0).astype(np.float32)

    k_val = u_star ** 2 / CMU ** 0.5
    out_k = np.full((M, M, n_z), k_val, dtype=np.float32)

    return {
        "lat": lats, "lon": lons,
        "u": out_u, "v": out_v, "T": out_T, "q": out_q, "k": out_k,
    }


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
    speed_interp, T_interp, p_interp, fd_x, fd_y, u_star, z0, ux_interp, uy_interp, q_interp = _build_interpolators(inflow)

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
            T_ref = float(_trapz(T_era5[mask], z_era5[mask]) / (z_era5[mask][-1] - z_era5[mask][0]))
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
            T_ref = float(_trapz(T_era5[mask], z_era5[mask]) / (z_era5[mask][-1] - z_era5[mask][0]))
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
        ux_interp=ux_interp, uy_interp=uy_interp, q_interp=q_interp,
    )

    u_path = case_dir / "0" / "U"
    k_path = case_dir / "0" / "k"
    epsilon_path = case_dir / "0" / "epsilon"
    t_path = case_dir / "0" / "T"
    q_path = case_dir / "0" / "q"
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
    if q_path.exists() and "q" in cell_fields:
        _patch_internal_field_scalar(q_path, cell_fields["q"])
    if p_path.exists():
        _patch_internal_field_scalar(p_path, cell_fields["p_rgh"])

    # ---- Boundary data for MappedFile BCs ----
    logger.info("Reading boundary face centres from %s", case_dir)
    boundary_faces = read_boundary_face_centres(case_dir)

    # Auto-detect lateral patches (cylindrical → "lateral"; box → cardinal four)
    lateral_patches = detect_lateral_patches(boundary_faces)

    # Compute profiles at each patch's face centres and write boundaryData.
    # If the inflow JSON carries a 3×3 ERA5 block, use 3D BCs
    # (bilinear lat/lon + PCHIP z + log-law blend); else fall back to the
    # legacy 1D profile (vertical-only, uniform horizontal).
    era5_grid = inflow.get("era5_grid")
    site_lat = inflow.get("site_lat")
    site_lon = inflow.get("site_lon")
    use_3d_bc = era5_grid is not None and site_lat is not None and site_lon is not None
    if use_3d_bc:
        logger.info("Using 3D lateral BCs: bilinear(lat,lon) + PCHIP(z) + log-law blend "
                    "[grid %dx%d, site=(%.3f, %.3f)]",
                    len(era5_grid["lats"]), len(era5_grid["lons"]), site_lat, site_lon)
    else:
        logger.info("era5_grid absent from inflow JSON — using legacy 1D lateral BCs")

    # Surface pressure for d2m → q conversion (use bottom of p_profile if available)
    if "p_profile" in inflow and len(inflow["p_profile"]) > 0:
        p_surf_Pa = float(inflow["p_profile"][0])
    else:
        p_surf_Pa = 101325.0

    # Site ground elevation (MSL) for AGL conversion in the log-law blend.
    # The lateral cylinder drapes over terrain → min z across all lateral
    # face centres ≈ terrain minimum. Used globally so all patches share the
    # same AGL reference.
    lateral_min_z = None
    for pname, fc in boundary_faces.items():
        if pname in lateral_patches and len(fc) > 0:
            zmin = float(fc[:, 2].min())
            lateral_min_z = zmin if lateral_min_z is None else min(lateral_min_z, zmin)
    if lateral_min_z is not None:
        logger.info("Site ground elevation (lateral min z) = %.1f m MSL", lateral_min_z)

    patch_fields: dict[str, dict[str, np.ndarray]] = {}

    for patch_name, face_centres in boundary_faces.items():
        if patch_name not in lateral_patches:
            continue
        if len(face_centres) == 0:
            continue

        if use_3d_bc:
            pf = interpolate_profiles_at_xyz(
                face_centres, era5_grid,
                site_lat=float(site_lat), site_lon=float(site_lon),
                u_star=u_star, z0=z0,
                fd_x=fd_x, fd_y=fd_y,
                T_ref=T_ref, is_bbsf=is_bbsf,
                p_surf_Pa=p_surf_Pa,
                site_ground_elev_m=lateral_min_z,
            )
        else:
            pf = interpolate_profiles_at_z(
                face_centres[:, 2], speed_interp, T_interp, p_interp,
                fd_x, fd_y, u_star, z0, T_ref, is_bbsf=is_bbsf,
                ux_interp=ux_interp, uy_interp=uy_interp, q_interp=q_interp,
            )

        fields = {
            "U": pf["U"],
            "k": pf["k"],
            "epsilon": pf["epsilon"],
            "p_rgh": pf["p_rgh"],
        }
        if t_path.exists():
            fields["T"] = pf["T"]
        if q_path.exists() and "q" in pf:
            fields["q"] = pf["q"]

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
