"""
export_sf_dataset.py — Extract CFD fields → dual Zarr format for surrogate training.

For each converged SF case:
  1. Read all cell fields via fluidfoam (x, y, z, U, k, T, q, ...)
  2. Read inflow.json for ERA5 profiles
  3. Read z0 field from boundaryData (if available)
  4. Export unstructured format (GNN/MLP) — cells within R_fine
  5. Export regular grid (U-Net 3D) — 128×128×32 on 4×4 km with:
     - Input: terrain, z0, ERA5 profiles
     - Target: CFD fields (u, v, w, T, q)
     - Residual: CFD - ERA5

Usage
-----
    cd services/module2a-cfd
    python export_sf_dataset.py \
        --cases-dir  ../../data/cases/poc_tbm_25ts_wc/ \
        --output-dir ../../data/cfd-database/surrogate/ \
        --half-extent 2000 --grid-size 128
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
from scipy.interpolate import griddata

logger = logging.getLogger(__name__)

# 32 log-spaced AGL levels [m]: dense near ground, sparse at top
Z_LEVELS_AGL = np.geomspace(5, 5000, 32).astype(np.float32)

CMU = 0.09  # k-epsilon model constant


# ── Field loading ────────────────────────────────────────────────────


def load_fields(case_dir: Path) -> dict | None:
    """Load OpenFOAM fields from the latest time step."""
    from export_cfd import _load_openfoam_fields

    try:
        return _load_openfoam_fields(case_dir)
    except (FileNotFoundError, Exception) as e:
        logger.warning("Cannot load fields from %s: %s", case_dir, e)
        return None


def compute_terrain_elevation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Estimate terrain elevation at each (x, y) by finding the minimum z in a column.

    For each cell, find the lowest z among cells within 25m horizontal distance.
    This gives terrain surface height without needing the STL.
    """
    # Unique horizontal positions (binned to 10m)
    x_bin = np.round(x / 10.0) * 10.0
    y_bin = np.round(y / 10.0) * 10.0
    xy_bin = x_bin * 1e7 + y_bin
    _, inverse, counts = np.unique(xy_bin, return_inverse=True, return_counts=True)

    elev = np.full(len(x), np.nan)
    for col_idx in range(len(counts)):
        mask = inverse == col_idx
        elev[mask] = z[mask].min()

    return elev.astype(np.float32)


def filter_zone(fields: dict, r_max: float) -> dict:
    """Filter cells within horizontal distance r_max from origin."""
    x, y = fields["x"], fields["y"]
    r = np.sqrt(x**2 + y**2)
    mask = r < r_max

    n_total, n_kept = len(x), int(mask.sum())
    logger.info("Zone filter: %d / %d cells (R < %.0f m)", n_kept, n_total, r_max)

    filtered = {}
    for key in ["x", "y", "z"]:
        filtered[key] = fields[key][mask]
    filtered["U"] = fields["U"][mask]
    for key in ["k", "nut", "T", "q", "epsilon"]:
        if key in fields:
            filtered[key] = fields[key][mask]
    return filtered


# ── ERA5 inflow helpers ──────────────────────────────────────────────


def read_inflow(case_dir: Path) -> dict | None:
    """Read inflow.json from a case directory."""
    path = case_dir / "inflow.json"
    if not path.exists():
        logger.warning("No inflow.json in %s", case_dir)
        return None
    with open(path) as f:
        return json.load(f)


def interpolate_era5_profiles(inflow: dict, z_levels: np.ndarray) -> dict:
    """Interpolate ERA5 inflow profiles at target z-levels.

    Returns dict with keys: u, v, T, q, k — each shape (nz,) float32.
    """
    z_src = np.array(inflow["z_levels"], dtype=np.float64)

    # Wind components
    if "ux_profile" in inflow:
        ux_src = np.array(inflow["ux_profile"], dtype=np.float64)
        uy_src = np.array(inflow["uy_profile"], dtype=np.float64)
    else:
        u_src = np.array(inflow["u_profile"], dtype=np.float64)
        fx = float(inflow.get("flowDir_x", 1.0))
        fy = float(inflow.get("flowDir_y", 0.0))
        ux_src = u_src * fx
        uy_src = u_src * fy

    u_interp = np.interp(z_levels, z_src, ux_src).astype(np.float32)
    v_interp = np.interp(z_levels, z_src, uy_src).astype(np.float32)

    # Temperature
    T_interp = None
    if "T_profile" in inflow:
        T_src = np.array(inflow["T_profile"], dtype=np.float64)
        T_interp = np.interp(z_levels, z_src, T_src).astype(np.float32)

    # Humidity
    q_interp = None
    if "q_profile" in inflow:
        q_src = np.array(inflow["q_profile"], dtype=np.float64)
        q_interp = np.interp(z_levels, z_src, q_src).astype(np.float32)

    # TKE: uniform k = u_star² / sqrt(Cmu) (as OF initializes)
    u_star = float(inflow.get("u_star", 0.3))
    k_val = u_star**2 / np.sqrt(CMU)
    k_interp = np.full(len(z_levels), k_val, dtype=np.float32)

    return {"u": u_interp, "v": v_interp, "T": T_interp, "q": q_interp, "k": k_interp}


# ── z0 field helpers ─────────────────────────────────────────────────


def _parse_of_scalar_list(text: str) -> np.ndarray:
    """Parse OpenFOAM ASCII scalar list: N\\n(\\nval1\\nval2\\n...\\n)"""
    start = text.index("(") + 1
    end = text.rindex(")")
    return np.array([float(v) for v in text[start:end].split()], dtype=np.float32)


def _parse_of_vector_list(text: str) -> np.ndarray:
    """Parse OpenFOAM ASCII vector list: N\\n(\\n(x y z)\\n...\\n)"""
    inner_start = text.index("(") + 1
    inner = text[inner_start : text.rindex(")")]
    pattern = r"\(([0-9eE.+\- ]+)\)"
    matches = re.findall(pattern, inner)
    vecs = []
    for m in matches:
        parts = m.split()
        if len(parts) == 3:
            vecs.append([float(p) for p in parts])
    return np.array(vecs, dtype=np.float32)


def read_z0_field(case_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Read z0 field from boundaryData/terrain.

    Returns (xy_points (N, 2), z0_values (N,)) or None.
    """
    z0_path = case_dir / "constant" / "boundaryData" / "terrain" / "0" / "z0"
    pts_path = case_dir / "constant" / "boundaryData" / "terrain" / "points"

    if not z0_path.exists() or not pts_path.exists():
        return None

    try:
        z0_vals = _parse_of_scalar_list(z0_path.read_text())
        pts_3d = _parse_of_vector_list(pts_path.read_text())
        return pts_3d[:, :2], z0_vals
    except Exception as e:
        logger.warning("Failed to read z0 field: %s", e)
        return None


# ── Unstructured export (GNN/MLP) ────────────────────────────────────


def export_unstructured(
    fields: dict,
    elev: np.ndarray,
    z_agl: np.ndarray,
    output_path: Path,
) -> None:
    """Export unstructured format (for GNN + MLP)."""
    import zarr

    store = zarr.open_group(str(output_path), mode="w")
    n = len(fields["x"])

    store.create_array("x", data=fields["x"].astype(np.float32))
    store.create_array("y", data=fields["y"].astype(np.float32))
    store.create_array("z", data=fields["z"].astype(np.float32))
    store.create_array("z_agl", data=z_agl.astype(np.float32))
    store.create_array("elev", data=elev.astype(np.float32))
    store.create_array("U", data=fields["U"].astype(np.float32))
    store.create_array("k", data=fields.get("k", np.zeros(n)).astype(np.float32))
    store.create_array("nut", data=fields.get("nut", np.zeros(n)).astype(np.float32))
    if "T" in fields:
        store.create_array("T", data=fields["T"].astype(np.float32))
    if "q" in fields:
        store.create_array("q", data=fields["q"].astype(np.float32))
    if "epsilon" in fields:
        store.create_array("epsilon", data=fields["epsilon"].astype(np.float32))

    store.attrs["n_cells"] = n
    logger.info("Unstructured Zarr: %s (%d cells)", output_path, n)


# ── Regular grid export (U-Net 3D) ──────────────────────────────────


def _fill_nan_3d(
    grid: np.ndarray,
    points_3d: np.ndarray,
    values: np.ndarray,
    z_levels: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
) -> None:
    """Fill NaN values in a 3D grid with nearest-neighbour interpolation (in-place)."""
    ny, nx, nz = grid.shape[:3]
    is_vector = grid.ndim == 4
    n_comp = grid.shape[3] if is_vector else 1

    for iz in range(nz):
        z_target = np.full_like(xg, z_levels[iz])
        target_pts = np.column_stack([xg.ravel(), yg.ravel(), z_target.ravel()])

        for comp in range(n_comp):
            if is_vector:
                layer = grid[:, :, iz, comp]
                src_vals = values[:, comp]
            else:
                layer = grid[:, :, iz]
                src_vals = values

            if not np.isnan(layer).any():
                continue

            vals_nn = griddata(points_3d, src_vals, target_pts, method="nearest")
            nan_mask = np.isnan(layer)
            layer[nan_mask] = vals_nn.reshape(ny, nx)[nan_mask]


def export_regular_grid(
    fields: dict,
    elev: np.ndarray,
    z_agl: np.ndarray,
    output_path: Path,
    *,
    half_extent: float = 2000.0,
    grid_size: int = 128,
    z_levels: np.ndarray | None = None,
    era5_profiles: dict | None = None,
    z0_data: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    """Export regular grid for U-Net 3D surrogate.

    Grid covers [-half_extent, +half_extent]² horizontally, with z_levels AGL.

    Zarr structure:
        input/terrain       (ny, nx)        — elevation [m]
        input/z0            (ny, nx)        — roughness [m]
        input/era5/{u,v,T,q,k}  (nz,)      — ERA5 profiles
        target/U            (ny, nx, nz, 3) — CFD velocity
        target/T            (ny, nx, nz)    — CFD temperature
        target/q            (ny, nx, nz)    — CFD humidity
        target/k            (ny, nx, nz)    — CFD TKE
        residual/U          (ny, nx, nz, 3) — velocity residual
        residual/T          (ny, nx, nz)    — T residual
        residual/q          (ny, nx, nz)    — q residual
        coords/{x_1d, y_1d, z_levels_agl}
    """
    import zarr

    if z_levels is None:
        z_levels = Z_LEVELS_AGL

    nx = ny = grid_size
    nz = len(z_levels)
    grid_res = 2 * half_extent / grid_size

    x_1d = np.linspace(-half_extent + grid_res / 2, half_extent - grid_res / 2, nx)
    y_1d = np.linspace(-half_extent + grid_res / 2, half_extent - grid_res / 2, ny)
    xg, yg = np.meshgrid(x_1d, y_1d)

    logger.info(
        "Grid: %d×%d×%d (%.1fm), extent=±%.0fm", nx, ny, nz, grid_res, half_extent
    )

    # ── Terrain (2D) ──
    points_2d = np.column_stack([fields["x"], fields["y"]])
    terrain_2d = griddata(points_2d, elev, (xg, yg), method="linear")
    if np.any(np.isnan(terrain_2d)):
        terrain_nn = griddata(points_2d, elev, (xg, yg), method="nearest")
        terrain_2d = np.where(np.isnan(terrain_2d), terrain_nn, terrain_2d)
    terrain_2d = terrain_2d.astype(np.float32)

    # ── z0 (2D) ──
    z0_2d = np.full((ny, nx), 0.05, dtype=np.float32)
    if z0_data is not None:
        z0_xy, z0_vals = z0_data
        z0_interp = griddata(z0_xy, z0_vals, (xg, yg), method="nearest")
        z0_2d = z0_interp.astype(np.float32)

    # ── 3D field interpolation ──
    points_3d = np.column_stack([fields["x"], fields["y"], z_agl])

    U_grid = np.full((ny, nx, nz, 3), np.nan, dtype=np.float32)
    k_grid = np.full((ny, nx, nz), np.nan, dtype=np.float32)
    T_grid = np.full((ny, nx, nz), np.nan, dtype=np.float32) if "T" in fields else None
    q_grid = np.full((ny, nx, nz), np.nan, dtype=np.float32) if "q" in fields else None

    for iz, z_level in enumerate(z_levels):
        z_target = np.full_like(xg, z_level)
        target_pts = np.column_stack([xg.ravel(), yg.ravel(), z_target.ravel()])

        for comp in range(3):
            vals = griddata(points_3d, fields["U"][:, comp], target_pts, method="linear")
            U_grid[:, :, iz, comp] = vals.reshape(ny, nx)

        if fields.get("k") is not None:
            vals = griddata(points_3d, fields["k"], target_pts, method="linear")
            k_grid[:, :, iz] = vals.reshape(ny, nx)

        if T_grid is not None:
            vals = griddata(points_3d, fields["T"], target_pts, method="linear")
            T_grid[:, :, iz] = vals.reshape(ny, nx)

        if q_grid is not None:
            vals = griddata(points_3d, fields["q"], target_pts, method="linear")
            q_grid[:, :, iz] = vals.reshape(ny, nx)

        if iz % 8 == 0:
            logger.info("  Interpolated level %d/%d (z_agl=%.0fm)", iz + 1, nz, z_level)

    # Fill NaN with nearest
    _fill_nan_3d(U_grid, points_3d, fields["U"], z_levels, xg, yg)
    if fields.get("k") is not None:
        _fill_nan_3d(k_grid, points_3d, fields["k"], z_levels, xg, yg)
    if T_grid is not None:
        _fill_nan_3d(T_grid, points_3d, fields["T"], z_levels, xg, yg)
    if q_grid is not None:
        _fill_nan_3d(q_grid, points_3d, fields["q"], z_levels, xg, yg)

    nan_pct = np.isnan(U_grid).sum() / U_grid.size * 100
    if nan_pct > 0:
        logger.warning("%.1f%% NaN remain in U_grid after filling", nan_pct)

    # ── Compute residuals ──
    U_residual = T_residual = q_residual = None
    if era5_profiles is not None:
        era5_u = era5_profiles["u"]  # (nz,)
        era5_v = era5_profiles["v"]  # (nz,)

        U_residual = U_grid.copy()
        for iz in range(nz):
            U_residual[:, :, iz, 0] -= era5_u[iz]
            U_residual[:, :, iz, 1] -= era5_v[iz]
            # w: ERA5 w ≈ 0, so residual ≈ w_cfd (unchanged)

        if T_grid is not None and era5_profiles["T"] is not None:
            T_residual = T_grid.copy()
            for iz in range(nz):
                T_residual[:, :, iz] -= era5_profiles["T"][iz]

        if q_grid is not None and era5_profiles["q"] is not None:
            q_residual = q_grid.copy()
            for iz in range(nz):
                q_residual[:, :, iz] -= era5_profiles["q"][iz]

    # ── Write Zarr ──
    store = zarr.open_group(str(output_path), mode="w")

    # Coordinates
    coords_grp = store.create_group("coords")
    coords_grp.create_array("x_1d", data=x_1d.astype(np.float32))
    coords_grp.create_array("y_1d", data=y_1d.astype(np.float32))
    coords_grp.create_array("z_levels_agl", data=z_levels)

    # Input channels (compact: 2D terrain + 1D profiles)
    inp = store.create_group("input")
    inp.create_array("terrain", data=terrain_2d)
    inp.create_array("z0", data=z0_2d)
    if era5_profiles is not None:
        era5_grp = inp.create_group("era5")
        for var in ["u", "v", "T", "q", "k"]:
            val = era5_profiles.get(var)
            if val is not None:
                era5_grp.create_array(var, data=val)

    # Target channels (full CFD solution)
    tgt = store.create_group("target")
    tgt.create_array("U", data=U_grid)
    tgt.create_array("k", data=k_grid)
    if T_grid is not None:
        tgt.create_array("T", data=T_grid)
    if q_grid is not None:
        tgt.create_array("q", data=q_grid)

    # Residual channels (CFD - ERA5)
    if U_residual is not None:
        res = store.create_group("residual")
        res.create_array("U", data=U_residual)
        if T_residual is not None:
            res.create_array("T", data=T_residual)
        if q_residual is not None:
            res.create_array("q", data=q_residual)

    # Metadata
    inner_pad = (grid_size - grid_size // 2) // 2  # pixels of context on each side
    store.attrs.update(
        {
            "grid_res_m": float(grid_res),
            "half_extent_m": float(half_extent),
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "inner_pad": inner_pad,
            "prediction_extent_m": float(half_extent),
        }
    )
    logger.info("Regular grid Zarr: %s (%d×%d×%d)", output_path, nx, ny, nz)


# ── Case export ──────────────────────────────────────────────────────


def export_case(
    case_dir: Path,
    case_id: str,
    output_dir: Path,
    *,
    r_fine: float = 1000.0,
    r_context: float = 3500.0,
    half_extent: float = 2000.0,
    grid_size: int = 128,
) -> bool:
    """Export a single case to both formats. Returns True on success."""
    output_case = output_dir / case_id

    # Skip if already exported
    grid_zarr = output_case / "grid.zarr"
    unstruct_zarr = output_case / "unstructured.zarr"
    if grid_zarr.exists() and unstruct_zarr.exists():
        logger.info("Already exported: %s", case_id)
        return True

    # Load all fields
    fields = load_fields(case_dir)
    if fields is None:
        return False

    # Read ERA5 inflow profiles
    inflow = read_inflow(case_dir)
    era5_profiles = None
    if inflow is not None:
        era5_profiles = interpolate_era5_profiles(inflow, Z_LEVELS_AGL)

    # Read z0 field
    z0_data = read_z0_field(case_dir)

    output_case.mkdir(parents=True, exist_ok=True)

    # ── Unstructured export (fine zone for GNN/MLP) ──
    fields_fine = filter_zone(fields, r_fine)
    elev_fine = compute_terrain_elevation(
        fields_fine["x"], fields_fine["y"], fields_fine["z"]
    )
    z_agl_fine = fields_fine["z"] - elev_fine
    export_unstructured(fields_fine, elev_fine, z_agl_fine, unstruct_zarr)

    # ── Regular grid export (context zone for U-Net) ──
    fields_ctx = filter_zone(fields, r_context)
    elev_ctx = compute_terrain_elevation(
        fields_ctx["x"], fields_ctx["y"], fields_ctx["z"]
    )
    z_agl_ctx = fields_ctx["z"] - elev_ctx
    export_regular_grid(
        fields_ctx,
        elev_ctx,
        z_agl_ctx,
        grid_zarr,
        half_extent=half_extent,
        grid_size=grid_size,
        era5_profiles=era5_profiles,
        z0_data=z0_data,
    )

    # Save inflow metadata
    inflow_json = case_dir / "inflow.json"
    if inflow_json.exists():
        import shutil

        shutil.copy2(inflow_json, output_case / "inflow.json")

    return True


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Export CFD fields to dual Zarr for surrogate training"
    )
    parser.add_argument("--cases-dir", required=True, help="Directory with case subdirs")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--half-extent",
        type=float,
        default=2000.0,
        help="Half-extent of regular grid [m] (default: 2000 = 4×4 km)",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=128,
        help="Grid points per axis (default: 128)",
    )
    parser.add_argument(
        "--r-fine",
        type=float,
        default=1000.0,
        help="Fine zone radius for unstructured export [m]",
    )
    parser.add_argument(
        "--r-context",
        type=float,
        default=3500.0,
        help="Context zone radius for regular grid export [m]",
    )
    args = parser.parse_args()

    cases_dir = Path(args.cases_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted(
        [d for d in cases_dir.iterdir() if d.is_dir() and d.name.startswith("ts_")]
    )
    if not case_dirs:
        # Also try case_* pattern
        case_dirs = sorted(
            [d for d in cases_dir.iterdir() if d.is_dir() and d.name.startswith("case_")]
        )

    logger.info("Found %d cases in %s", len(case_dirs), cases_dir)

    n_ok = 0
    for case_dir in case_dirs:
        case_id = case_dir.name
        if export_case(
            case_dir,
            case_id,
            output_dir,
            r_fine=args.r_fine,
            r_context=args.r_context,
            half_extent=args.half_extent,
            grid_size=args.grid_size,
        ):
            n_ok += 1

    logger.info("Exported %d / %d cases", n_ok, len(case_dirs))


if __name__ == "__main__":
    main()
