"""
export_sf_dataset.py — Extract CFD fields from fine zone → dual Zarr format.

For each converged SF case:
  1. Read fields via fluidfoam (x, y, z, U, k, nut)
  2. Filter cells in fine zone: sqrt(x² + y²) < R_fine
  3. Compute z_agl = z - terrain_elevation(x, y)
  4. Export unstructured format (GNN/MLP) + regular grid (U-Net 3D)

Usage
-----
    cd services/module2a-cfd
    python export_sf_dataset.py \
        --cases-dir  ../../data/cases/sf_poc/ \
        --output-dir ../../data/cfd-database/sf_poc/ \
        --r-fine     1200.0 \
        --grid-res   50.0
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy.interpolate import griddata

logger = logging.getLogger(__name__)

# AGL levels for regular grid [m]
Z_LEVELS_AGL = np.array([
    10, 20, 50, 100, 300, 500, 700, 900, 1100, 1300,
    1500, 1700, 1900, 2100, 2300, 2500, 2700, 2900,
], dtype=np.float32)


def load_fields(case_dir: Path) -> dict | None:
    """Load OpenFOAM fields from the latest time step."""
    from export_cfd import _load_openfoam_fields, _is_time_dir

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
    from scipy.spatial import cKDTree

    # Build 2D tree (horizontal)
    xy = np.column_stack([x, y])
    tree = cKDTree(xy)

    # For each cell, find neighbours within 25m horizontal radius
    # and take the minimum z as terrain elevation
    elev = np.full(len(x), np.nan)
    chunk_size = 5000

    # Unique horizontal positions (binned to 10m)
    x_bin = np.round(x / 10.0) * 10.0
    y_bin = np.round(y / 10.0) * 10.0
    xy_bin = x_bin * 1e7 + y_bin
    _, inverse, counts = np.unique(xy_bin, return_inverse=True, return_counts=True)

    # For each unique horizontal position, min z = terrain elevation
    for col_idx in range(len(counts)):
        mask = inverse == col_idx
        elev[mask] = z[mask].min()

    return elev.astype(np.float32)


def filter_fine_zone(fields: dict, r_fine: float) -> dict:
    """Filter cells within the fine zone (horizontal distance < r_fine from origin)."""
    x, y = fields["x"], fields["y"]
    r = np.sqrt(x**2 + y**2)
    mask = r < r_fine

    n_total = len(x)
    n_fine = mask.sum()
    logger.info("Fine zone filter: %d / %d cells (R < %.0f m)", n_fine, n_total, r_fine)

    filtered = {}
    for key in ["x", "y", "z"]:
        filtered[key] = fields[key][mask]
    filtered["U"] = fields["U"][mask]
    for key in ["k", "nut", "T", "q", "epsilon"]:
        if key in fields:
            filtered[key] = fields[key][mask]

    return filtered


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


def export_regular_grid(
    fields: dict,
    elev: np.ndarray,
    z_agl: np.ndarray,
    grid_res: float,
    r_fine: float,
    output_path: Path,
    r_context: float | None = None,
    terrain_stl_path: Path | None = None,
) -> None:
    """Export regular grid format (for U-Net 3D).

    Interpolates CFD fields onto a regular horizontal grid × terrain-following levels.
    """
    import zarr

    # Build regular horizontal grid inscribed in the fine zone circle
    half_extent = r_fine / np.sqrt(2)  # inscribed square
    nx = int(2 * half_extent / grid_res)
    ny = nx
    x_1d = np.linspace(-half_extent + grid_res / 2, half_extent - grid_res / 2, nx)
    y_1d = np.linspace(-half_extent + grid_res / 2, half_extent - grid_res / 2, ny)
    xg, yg = np.meshgrid(x_1d, y_1d)

    logger.info("Regular grid: %d×%d (%.0fm), extent=%.0fm", nx, ny, grid_res, 2 * half_extent)

    # Interpolate terrain elevation onto the regular grid
    points_2d = np.column_stack([fields["x"], fields["y"]])
    terrain_2d = griddata(points_2d, elev, (xg, yg), method="linear")

    # Fill NaN edges with nearest
    if np.any(np.isnan(terrain_2d)):
        terrain_nearest = griddata(points_2d, elev, (xg, yg), method="nearest")
        terrain_2d = np.where(np.isnan(terrain_2d), terrain_nearest, terrain_2d)

    # Interpolate fields at each (x, y, z_level_agl)
    nz = len(Z_LEVELS_AGL)
    U_grid = np.full((ny, nx, nz, 3), np.nan, dtype=np.float32)
    k_grid = np.full((ny, nx, nz), np.nan, dtype=np.float32)

    # Optional scalar grids
    scalar_grids = {}
    for sname in ["T", "q", "epsilon"]:
        if sname in fields:
            scalar_grids[sname] = np.full((ny, nx, nz), np.nan, dtype=np.float32)

    points_3d = np.column_stack([fields["x"], fields["y"], z_agl])

    for iz, z_level in enumerate(Z_LEVELS_AGL):
        # Target points at this AGL level
        z_target = np.full_like(xg, z_level)
        target_pts = np.column_stack([xg.ravel(), yg.ravel(), z_target.ravel()])

        for comp in range(3):
            vals = griddata(points_3d, fields["U"][:, comp], target_pts, method="linear")
            U_grid[:, :, iz, comp] = vals.reshape(ny, nx)

        k_vals = fields.get("k")
        if k_vals is not None:
            vals = griddata(points_3d, k_vals, target_pts, method="linear")
            k_grid[:, :, iz] = vals.reshape(ny, nx)

        for sname, sgrid in scalar_grids.items():
            vals = griddata(points_3d, fields[sname], target_pts, method="linear")
            sgrid[:, :, iz] = vals.reshape(ny, nx)

    # Fill remaining NaN with nearest interpolation
    nan_mask_u = np.isnan(U_grid)
    if nan_mask_u.any():
        pct = nan_mask_u.sum() / nan_mask_u.size * 100
        logger.info("Filling %.1f%% NaN in U_grid with nearest interpolation", pct)
        for iz in range(nz):
            z_level = Z_LEVELS_AGL[iz]
            z_target = np.full_like(xg, z_level)
            target_pts = np.column_stack([xg.ravel(), yg.ravel(), z_target.ravel()])
            for comp in range(3):
                if np.isnan(U_grid[:, :, iz, comp]).any():
                    vals = griddata(points_3d, fields["U"][:, comp],
                                    target_pts, method="nearest")
                    layer = U_grid[:, :, iz, comp]
                    nan_mask = np.isnan(layer)
                    layer[nan_mask] = vals.reshape(ny, nx)[nan_mask]

    # Write to Zarr
    store = zarr.open_group(str(output_path), mode="w")
    store.create_array("terrain", data=terrain_2d.astype(np.float32))
    store.create_array("U", data=U_grid)
    store.create_array("k", data=k_grid)
    for sname, sgrid in scalar_grids.items():
        store.create_array(sname, data=sgrid)
    store.create_array("z_levels_agl", data=Z_LEVELS_AGL)
    store.create_array("x_1d", data=x_1d.astype(np.float32))
    store.create_array("y_1d", data=y_1d.astype(np.float32))

    store.attrs.update({
        "grid_res_m": grid_res,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "r_fine_m": r_fine,
    })
    logger.info("Regular grid Zarr: %s (%d×%d×%d)", output_path, nx, ny, nz)


def export_case(
    case_dir: Path,
    case_id: str,
    output_dir: Path,
    r_fine: float,
    grid_res: float,
) -> bool:
    """Export a single case to both formats. Returns True on success."""
    output_case = output_dir / case_id

    # Skip if already exported
    if (output_case / "unstructured.zarr").exists() and (output_case / "grid.zarr").exists():
        logger.info("Already exported: %s", case_id)
        return True

    # Load fields
    fields = load_fields(case_dir)
    if fields is None:
        return False

    # Filter fine zone
    fields_fine = filter_fine_zone(fields, r_fine)

    # Compute terrain elevation and z_agl
    elev = compute_terrain_elevation(
        fields_fine["x"], fields_fine["y"], fields_fine["z"],
    )
    z_agl = fields_fine["z"] - elev

    output_case.mkdir(parents=True, exist_ok=True)

    # Export unstructured
    export_unstructured(fields_fine, elev, z_agl, output_case / "unstructured.zarr")

    # Export regular grid
    export_regular_grid(
        fields_fine, elev, z_agl,
        grid_res=grid_res,
        r_fine=r_fine,
        output_path=output_case / "grid.zarr",
    )

    # Save inflow metadata alongside
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

    parser = argparse.ArgumentParser(description="Export SF PoC CFD fields to dual Zarr")
    parser.add_argument("--cases-dir", required=True, help="Directory with case subdirs")
    parser.add_argument("--output-dir", required=True, help="Output Zarr directory")
    parser.add_argument("--r-fine", type=float, default=1200.0, help="Fine zone radius [m]")
    parser.add_argument("--grid-res", type=float, default=50.0, help="Regular grid resolution [m]")
    args = parser.parse_args()

    cases_dir = Path(args.cases_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find case directories (ts_YYYYMMDD_HHMM pattern)
    case_dirs = sorted([
        d for d in cases_dir.iterdir()
        if d.is_dir() and d.name.startswith("ts_")
    ])

    logger.info("Found %d cases in %s", len(case_dirs), cases_dir)

    n_ok = 0
    for case_dir in case_dirs:
        case_id = case_dir.name
        if export_case(case_dir, case_id, output_dir, args.r_fine, args.grid_res):
            n_ok += 1

    logger.info("Exported %d / %d cases", n_ok, len(case_dirs))


if __name__ == "__main__":
    main()
