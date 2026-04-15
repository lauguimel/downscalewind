"""
export_to_grid_zarr.py — Direct OF case → grid.zarr export (complex_terrain_v1).

Replaces the 2-step pipeline (export_campaign_zarr.py + convert_stacked_to_training.py)
with a single direct export. One grid.zarr per solved OF case, 128×128×32 grid on
log-spaced AGL levels, with extended schema:

Inputs (for ML surrogate):
    input/terrain              (128, 128)        elevation [m]
    input/z0                   (128, 128)        WorldCover roughness [m]
    input/era5/{u,v,T,q,k}     (32,)             1D ERA5 profile on AGL
    input/era5_3d/{u,v,T,q,k}  (3, 3, 32)        ERA5 3×3 grid (if provided)
    input/era5_surface/{t2m,d2m,u10,v10}  (3,3)  ERA5 surface vars

Targets (from CFD):
    target/U                   (128, 128, 32, 3) [m/s]
    target/T                   (128, 128, 32)    [K]
    target/q                   (128, 128, 32)    [kg/kg]
    target/k                   (128, 128, 32)    [m²/s²]    (optional)
    target/epsilon             (128, 128, 32)    [m²/s³]    (optional)
    target/nut                 (128, 128, 32)    [m²/s]     (optional)

Residuals (CFD - ERA5 lifted):
    residual/U, residual/T, residual/q, residual/k (optional)

Coordinates:
    coords/x_1d, coords/y_1d   (128,)            [m] centred on site
    coords/z_agl               (32,)             log-spaced 5 → 5000 m
    coords/elev                (128, 128)        terrain [m]

Attributes (zarr.attrs):
    site_id, group, timestamp_iso, era5_source_path, country, climate_zone,
    lat, lon, elevation_m, mean_slope_deg, std_elev_local_m,
    mesh.* (inner_size_m, resolution_m, cells_z, grading_z, AR_first_cell,
            first_cell_m, height_m),
    physics.* (coriolis_enabled, canopy_enabled, turbulence_model,
               T_passive, q_passive, wall_function),
    inflow.* (u_hub, u_star, z0_eff, T_ref, q_ref, Ri_b, wind_dir,
              t2m_K, d2m_K, u10_ms, v10_ms),
    solver.* (n_iter, wall_time_s, final_residual_U, final_residual_p,
              final_residual_T, converged).

Usage
-----
    python services/module2a-cfd/export_to_grid_zarr.py \\
        --case-dir /path/to/case_ts042 \\
        --time 2000 \\
        --site-manifest data/campaign/complex_terrain_v1/manifests/sites.yaml \\
        --campaign-manifest data/campaign/complex_terrain_v1/manifests/campaign.yaml \\
        --out /path/to/cases/site_id_case_ts042/grid.zarr \\
        --include-turb k epsilon nut          # optional extra targets
        --device cuda
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

# ── Reuse logic from existing scripts ─────────────────────────────────────
# These imports mirror the local helpers; if path changes, copy them inline.
try:
    from export_campaign_zarr import (
        load_mesh,
        compute_terrain_elevation,
        load_fields,
        load_inflow,
        load_stl,
    )
except ImportError:
    # When running from outside services/module2a-cfd, add path dynamically
    import sys
    _here = Path(__file__).resolve().parent
    sys.path.insert(0, str(_here))
    from export_campaign_zarr import (  # type: ignore  # noqa
        load_mesh,
        compute_terrain_elevation,
        load_fields,
        load_inflow,
        load_stl,
    )


logger = logging.getLogger(__name__)


# 32 log-spaced AGL levels [m]: dense near ground (5 m), sparse at top (5 km)
Z_LEVELS_AGL = np.geomspace(5.0, 5000.0, 32).astype(np.float32)

# Grid geometry (ML surrogate target)
DEFAULT_GRID_SIZE = 128
DEFAULT_HALF_EXTENT_M = 3000.0     # 6 km × 6 km centred on site (matches 2 km inner + margin)
DEFAULT_R_CONTEXT = 4000.0         # radius of OF cells to read for IDW source
DEFAULT_R_FINE = 3000.0            # radius flagged as "fine" for debugging

# k-epsilon constants (ERA5 turb profile derivation)
CMU = 0.09


# ─────────────────────────────────────────────────────────────────────────────
# GPU IDW (copy from convert_stacked_to_training.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_idw_weights(
    src_pts: np.ndarray,
    tgt_pts: np.ndarray,
    k: int = 8,
    power: float = 2.0,
    device: str = "cuda",
    chunk_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    """Build k-NN IDW weights. GPU if torch available + CUDA, else CPU."""
    try:
        import torch
        use_torch = True
    except ImportError:
        use_torch = False

    if use_torch and device != "cpu":
        try:
            src_t = torch.from_numpy(src_pts.astype(np.float32)).to(device)
            n_tgt = len(tgt_pts)
            all_indices = np.empty((n_tgt, k), dtype=np.int64)
            all_weights = np.empty((n_tgt, k), dtype=np.float32)
            for i0 in range(0, n_tgt, chunk_size):
                i1 = min(i0 + chunk_size, n_tgt)
                tgt_chunk = torch.from_numpy(tgt_pts[i0:i1].astype(np.float32)).to(device)
                dists = torch.cdist(tgt_chunk, src_t)
                topk_dists, topk_idx = torch.topk(dists, k, dim=1, largest=False)
                topk_dists = topk_dists.clamp(min=1e-6)
                w = 1.0 / topk_dists.pow(power)
                w = w / w.sum(dim=1, keepdim=True)
                all_indices[i0:i1] = topk_idx.cpu().numpy()
                all_weights[i0:i1] = w.cpu().numpy()
            return all_indices, all_weights
        except RuntimeError as exc:
            logger.warning("GPU IDW failed (%s), falling back to CPU scipy", exc)

    # CPU fallback via scipy.cKDTree
    from scipy.spatial import cKDTree  # noqa
    tree = cKDTree(src_pts)
    dists, idx = tree.query(tgt_pts, k=k)
    dists = np.clip(dists, 1e-6, None)
    w = 1.0 / dists**power
    w = w / w.sum(axis=1, keepdims=True)
    return idx.astype(np.int64), w.astype(np.float32)


def apply_idw(values: np.ndarray, indices: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if values.ndim == 2:
        return (weights[:, :, None] * values[indices]).sum(axis=1).astype(np.float32)
    return (weights * values[indices]).sum(axis=1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# ERA5 profile utilities
# ─────────────────────────────────────────────────────────────────────────────

def interpolate_inflow_profiles(
    inflow: dict,
    z_levels: np.ndarray,
) -> dict:
    """Interpolate inflow.json profiles (ux, uy, T, q) at target z AGL levels.

    Also derives a k profile from u_star (k = u*² / sqrt(Cmu)).
    """
    z_src = np.asarray(inflow["z_levels"], dtype=np.float64)
    ux_src = np.asarray(inflow["ux_profile"], dtype=np.float64)
    uy_src = np.asarray(inflow["uy_profile"], dtype=np.float64)

    out: dict[str, Optional[np.ndarray]] = {
        "u": np.interp(z_levels, z_src, ux_src).astype(np.float32),
        "v": np.interp(z_levels, z_src, uy_src).astype(np.float32),
        "T": None, "q": None, "k": None,
    }
    if "T_profile" in inflow and inflow["T_profile"] is not None:
        T_src = np.asarray(inflow["T_profile"], dtype=np.float64)
        out["T"] = np.interp(z_levels, z_src, T_src).astype(np.float32)
    if "q_profile" in inflow and inflow["q_profile"] is not None:
        q_src = np.asarray(inflow["q_profile"], dtype=np.float64)
        out["q"] = np.interp(z_levels, z_src, q_src).astype(np.float32)

    u_star = float(inflow.get("u_star", 0.3))
    k_val = u_star**2 / np.sqrt(CMU)
    out["k"] = np.full(len(z_levels), k_val, dtype=np.float32)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Grid construction
# ─────────────────────────────────────────────────────────────────────────────

def build_target_grid(
    half_extent_m: float,
    grid_size: int,
    z_levels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (target_pts_N3, x_1d, y_1d)."""
    grid_res = 2 * half_extent_m / grid_size
    x_1d = np.linspace(
        -half_extent_m + grid_res / 2, half_extent_m - grid_res / 2, grid_size,
        dtype=np.float32)
    y_1d = np.linspace(
        -half_extent_m + grid_res / 2, half_extent_m - grid_res / 2, grid_size,
        dtype=np.float32)
    xg, yg, zg = np.meshgrid(x_1d, y_1d, z_levels, indexing="ij")
    target_pts = np.column_stack([
        xg.ravel(), yg.ravel(), zg.ravel()
    ]).astype(np.float32)
    return target_pts, x_1d, y_1d


# ─────────────────────────────────────────────────────────────────────────────
# Solver diagnostics
# ─────────────────────────────────────────────────────────────────────────────

_RES_RE = re.compile(
    r"Solving for (\w+).*?Initial residual = ([0-9.eE+-]+).*?Final residual = ([0-9.eE+-]+)")


def parse_solver_log(log_path: Path) -> dict:
    """Scan log.simpleFoam for last residuals and wall time if present."""
    result = {
        "final_residual_U": np.nan,
        "final_residual_p": np.nan,
        "final_residual_T": np.nan,
        "n_iter": -1,
        "wall_time_s": np.nan,
        "converged": False,
    }
    if not log_path.exists():
        return result
    text = log_path.read_text(errors="ignore")
    # Collect final residuals (take last occurrence per field)
    for m in _RES_RE.finditer(text):
        field, _, final = m.group(1), m.group(2), m.group(3)
        try:
            fr = float(final)
        except ValueError:
            continue
        if field == "Ux" or field == "U":
            result["final_residual_U"] = fr
        elif field == "p":
            result["final_residual_p"] = fr
        elif field == "T":
            result["final_residual_T"] = fr
    # Time steps performed
    time_matches = re.findall(r"^Time = (\d+)\s*$", text, re.MULTILINE)
    if time_matches:
        try:
            result["n_iter"] = int(time_matches[-1])
        except ValueError:
            pass
    # Wall time
    exec_time = re.findall(r"ExecutionTime = ([0-9.eE+-]+) s", text)
    if exec_time:
        try:
            result["wall_time_s"] = float(exec_time[-1])
        except ValueError:
            pass
    # Rough converged check: residuals < 1e-3 for U, p
    result["converged"] = bool(
        np.isfinite(result["final_residual_U"]) and result["final_residual_U"] < 5e-3
        and np.isfinite(result["final_residual_p"]) and result["final_residual_p"] < 5e-3
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Terrain → z0 from boundaryData
# ─────────────────────────────────────────────────────────────────────────────

def load_z0_at_xy(case_dir: Path, x_1d: np.ndarray, y_1d: np.ndarray,
                  device: str) -> np.ndarray:
    """Interpolate z0 from boundaryData/terrain onto the 2D grid."""
    z0_path = case_dir / "constant" / "boundaryData" / "terrain" / "0" / "z0"
    pts_path = case_dir / "constant" / "boundaryData" / "terrain" / "points"
    if not (z0_path.exists() and pts_path.exists()):
        logger.warning("z0 boundaryData missing for %s — filling with 0.1 m", case_dir.name)
        return np.full((len(y_1d), len(x_1d)), 0.1, dtype=np.float32)

    def _parse_list_scalars(text: str) -> np.ndarray:
        in_list = False
        vals = []
        for line in text.splitlines():
            line = line.strip()
            if line == "(":
                in_list = True
                continue
            if line == ")":
                break
            if in_list:
                try:
                    vals.append(float(line))
                except ValueError:
                    continue
        return np.asarray(vals, dtype=np.float32)

    def _parse_list_points(text: str) -> np.ndarray:
        in_list = False
        pts = []
        for line in text.splitlines():
            line = line.strip()
            if line == "(":
                in_list = True
                continue
            if line == ")":
                break
            if in_list and line.startswith("("):
                s = line.strip("()")
                parts = s.split()
                if len(parts) >= 3:
                    pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
        return np.asarray(pts, dtype=np.float32)

    z0_vals = _parse_list_scalars(z0_path.read_text())
    pts = _parse_list_points(pts_path.read_text())
    if len(z0_vals) == 0 or len(pts) == 0:
        return np.full((len(y_1d), len(x_1d)), 0.1, dtype=np.float32)

    src_xy = pts[:, :2].astype(np.float32)
    xg, yg = np.meshgrid(x_1d, y_1d, indexing="xy")
    tgt_xy = np.column_stack([xg.ravel(), yg.ravel()]).astype(np.float32)
    src3 = np.column_stack([src_xy, np.zeros(len(src_xy), dtype=np.float32)])
    tgt3 = np.column_stack([tgt_xy, np.zeros(len(tgt_xy), dtype=np.float32)])
    idx, w = build_idw_weights(src3, tgt3, k=4, power=2.0, device=device)
    z0_grid = apply_idw(z0_vals[:len(src_xy)], idx, w)
    return z0_grid.reshape(len(y_1d), len(x_1d))


# ─────────────────────────────────────────────────────────────────────────────
# Campaign + site metadata lookup
# ─────────────────────────────────────────────────────────────────────────────

def load_yaml(path: Optional[Path]) -> dict:
    if path is None or not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def site_from_manifest(manifest: dict, site_id: str) -> dict:
    for s in manifest.get("sites", []):
        if s.get("site_id") == site_id:
            return s
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Main export
# ─────────────────────────────────────────────────────────────────────────────

def export_case_to_grid(
    case_dir: Path,
    time_name: str,
    out_path: Path,
    *,
    site_manifest: Optional[Path] = None,
    campaign_manifest: Optional[Path] = None,
    grid_size: int = DEFAULT_GRID_SIZE,
    half_extent_m: float = DEFAULT_HALF_EXTENT_M,
    r_context: float = DEFAULT_R_CONTEXT,
    include_turb: tuple[str, ...] = (),
    device: str = "cuda",
    era5_3d: Optional[np.ndarray] = None,
    era5_surface_3x3: Optional[dict] = None,
) -> bool:
    """Export one OF case to a grid.zarr. Returns True on success."""
    import zarr

    t0 = time.time()
    case_name = case_dir.name

    # ── Load OF fields ──
    fields = load_fields(case_dir, time_name)
    if fields is None or "U" not in fields:
        logger.error("Cannot load fields from %s/%s", case_name, time_name)
        return False

    mesh = load_mesh(case_dir)
    x, y, z = mesh["x"], mesh["y"], mesh["z"]
    elev, z_agl = compute_terrain_elevation(x, y, z)

    # ── Context filter (reduce source cells for IDW) ──
    mask_ctx = (x**2 + y**2) < r_context**2
    x_ctx, y_ctx, z_ctx = x[mask_ctx], y[mask_ctx], z[mask_ctx]
    elev_ctx, z_agl_ctx = elev[mask_ctx], z_agl[mask_ctx]
    logger.info("%s: %d cells (%d in context)", case_name, len(x), len(x_ctx))

    # ── Build target grid + IDW weights (3D) ──
    target_pts, x_1d, y_1d = build_target_grid(half_extent_m, grid_size, Z_LEVELS_AGL)
    src_pts = np.column_stack([x_ctx, y_ctx, z_agl_ctx])
    idx3d, w3d = build_idw_weights(src_pts, target_pts, k=8, power=2.0, device=device)

    ny = nx = grid_size
    nz = len(Z_LEVELS_AGL)

    # Interpolate U, T, q
    U_ctx = fields["U"][mask_ctx]                                   # (n_ctx, 3)
    U_grid_flat = apply_idw(U_ctx, idx3d, w3d)                      # (nx*ny*nz, 3)
    U_grid = U_grid_flat.reshape(nx, ny, nz, 3).transpose(1, 0, 2, 3)  # (ny, nx, nz, 3)

    T_grid = q_grid = None
    if "T" in fields:
        T_grid = apply_idw(fields["T"][mask_ctx], idx3d, w3d).reshape(nx, ny, nz).transpose(1, 0, 2)
    if "q" in fields:
        q_grid = apply_idw(fields["q"][mask_ctx], idx3d, w3d).reshape(nx, ny, nz).transpose(1, 0, 2)

    # Optional turbulence targets
    turb_grids: dict[str, np.ndarray] = {}
    for t in include_turb:
        if t in fields:
            g = apply_idw(fields[t][mask_ctx], idx3d, w3d).reshape(nx, ny, nz).transpose(1, 0, 2)
            turb_grids[t] = g

    # Terrain 2D
    src2d = np.column_stack([x_ctx, y_ctx, np.zeros(len(x_ctx), dtype=np.float32)])
    xg, yg = np.meshgrid(x_1d, y_1d, indexing="xy")
    tgt2d = np.column_stack([xg.ravel(), yg.ravel(),
                             np.zeros(grid_size * grid_size, dtype=np.float32)]).astype(np.float32)
    idx2d, w2d = build_idw_weights(src2d, tgt2d, k=4, power=2.0, device=device)
    terrain_2d = apply_idw(elev_ctx, idx2d, w2d).reshape(ny, nx)

    # z0 grid
    z0_2d = load_z0_at_xy(case_dir, x_1d, y_1d, device=device)

    # ── Inflow profiles → ERA5 1D on AGL levels ──
    inflow = load_inflow(case_dir)
    era5_profiles = None
    if inflow is not None:
        era5_profiles = interpolate_inflow_profiles(inflow, Z_LEVELS_AGL)

    # ── Residuals (CFD - ERA5 lifted) ──
    U_res = T_res = q_res = None
    turb_res: dict[str, np.ndarray] = {}
    if era5_profiles is not None:
        U_res = U_grid.copy()
        for iz in range(nz):
            U_res[:, :, iz, 0] -= era5_profiles["u"][iz]
            U_res[:, :, iz, 1] -= era5_profiles["v"][iz]
        if T_grid is not None and era5_profiles["T"] is not None:
            T_res = T_grid.copy()
            for iz in range(nz):
                T_res[:, :, iz] -= era5_profiles["T"][iz]
        if q_grid is not None and era5_profiles["q"] is not None:
            q_res = q_grid.copy()
            for iz in range(nz):
                q_res[:, :, iz] -= era5_profiles["q"][iz]
        # k residual = target_k - era5_k (scalar profile)
        if "k" in turb_grids:
            kr = turb_grids["k"].copy()
            for iz in range(nz):
                kr[:, :, iz] -= era5_profiles["k"][iz]
            turb_res["k"] = kr

    # ── Solver diagnostics ──
    solver_diag = parse_solver_log(case_dir / "log.simpleFoam")

    # ── Resolve metadata from manifests ──
    site_manifest_data = load_yaml(site_manifest)
    campaign_manifest_data = load_yaml(campaign_manifest)
    # site_id derived from case naming convention "{site_id}_case_tsNNN"
    site_id = case_name.rsplit("_case_", 1)[0] if "_case_" in case_name else "unknown"
    site_meta = site_from_manifest(site_manifest_data, site_id)

    # Mesh info (for AR tracking)
    mesh_info = campaign_manifest_data.get("mesh", {})
    # Physics flags
    physics_info = campaign_manifest_data.get("physics", {})
    solver_info = campaign_manifest_data.get("solver", {})

    timestamp_iso = inflow.get("timestamp") if inflow else None
    era5_source_path = inflow.get("era5_source") if inflow else None

    # ── Write grid.zarr ──
    out_path.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.open_group(str(out_path), mode="w")

    coords_grp = store.create_group("coords")
    coords_grp.create_array("x_1d", data=x_1d.astype(np.float32))
    coords_grp.create_array("y_1d", data=y_1d.astype(np.float32))
    coords_grp.create_array("z_agl", data=Z_LEVELS_AGL)
    coords_grp.create_array("elev", data=terrain_2d.astype(np.float32))

    inp = store.create_group("input")
    inp.create_array("terrain", data=terrain_2d.astype(np.float32))
    inp.create_array("z0", data=z0_2d.astype(np.float32))
    if era5_profiles is not None:
        era5_grp = inp.create_group("era5")
        for var, val in era5_profiles.items():
            if val is not None:
                era5_grp.create_array(var, data=val)
    # ERA5 3×3 grid (optional, provided externally)
    if era5_3d is not None:
        era5_3d_grp = inp.create_group("era5_3d")
        for var, val in era5_3d.items():  # type: ignore[attr-defined]
            if val is not None:
                era5_3d_grp.create_array(var, data=np.asarray(val, dtype=np.float32))
    # ERA5 surface 3×3 (t2m, d2m, u10, v10)
    if era5_surface_3x3 is not None:
        era5_surf_grp = inp.create_group("era5_surface")
        for var, val in era5_surface_3x3.items():
            era5_surf_grp.create_array(var, data=np.asarray(val, dtype=np.float32))

    tgt = store.create_group("target")
    tgt.create_array("U", data=U_grid.astype(np.float32))
    if T_grid is not None:
        tgt.create_array("T", data=T_grid.astype(np.float32))
    if q_grid is not None:
        tgt.create_array("q", data=q_grid.astype(np.float32))
    for name, g in turb_grids.items():
        tgt.create_array(name, data=g.astype(np.float32))

    if U_res is not None:
        res = store.create_group("residual")
        res.create_array("U", data=U_res.astype(np.float32))
        if T_res is not None:
            res.create_array("T", data=T_res.astype(np.float32))
        if q_res is not None:
            res.create_array("q", data=q_res.astype(np.float32))
        for name, g in turb_res.items():
            res.create_array(name, data=g.astype(np.float32))

    # ── Attributes ──
    grid_res_m = 2 * half_extent_m / grid_size
    AR_first = None
    if mesh_info.get("first_cell_m_approx"):
        AR_first = float(mesh_info.get("horizontal_resolution_m", grid_res_m) /
                         mesh_info["first_cell_m_approx"])

    attrs = {
        "site_id": site_id,
        "case_name": case_name,
        "timestamp_iso": timestamp_iso or "",
        "era5_source_path": era5_source_path or "",
        "group": site_meta.get("group", "unknown"),
        "country": site_meta.get("country", "XX"),
        "climate_zone": site_meta.get("climate_zone", ""),
        "lat": float(site_meta.get("lat", np.nan)),
        "lon": float(site_meta.get("lon", np.nan)),
        "elevation_m": float(site_meta.get("elevation_m", np.nan)),
        "mean_slope_deg": float(site_meta.get("mean_slope_deg", np.nan)),
        "std_elev_local_m": float(site_meta.get("std_elev_local_m", np.nan)),
        # mesh
        "mesh.grid_res_m": float(grid_res_m),
        "mesh.half_extent_m": float(half_extent_m),
        "mesh.grid_size": int(grid_size),
        "mesh.nz": int(nz),
        "mesh.source_cfd_resolution_m": float(mesh_info.get("horizontal_resolution_m", np.nan)),
        "mesh.source_cfd_first_cell_m": float(mesh_info.get("first_cell_m_approx", np.nan)),
        "mesh.source_cfd_AR_first": float(AR_first) if AR_first is not None else np.nan,
        "mesh.source_cfd_cells_z": int(mesh_info.get("cells_z", -1)),
        "mesh.source_cfd_inner_size_m": float(mesh_info.get("inner_size_m", np.nan)),
        "mesh.source_cfd_height_m": float(mesh_info.get("height_m", np.nan)),
        # physics
        "physics.coriolis_enabled": bool(physics_info.get("coriolis_enabled", True)),
        "physics.canopy_enabled": bool(physics_info.get("canopy_enabled", False)),
        "physics.wall_function": str(physics_info.get("wall_function", "")),
        "physics.turbulence_model": str(solver_info.get("turbulence_model", "")),
        "physics.T_passive": str(solver_info.get("transport_T", "passive")),
        "physics.q_passive": str(solver_info.get("transport_q", "passive")),
        # inflow (from inflow.json)
        "inflow.u_hub": float(inflow.get("u_hub", np.nan)) if inflow else np.nan,
        "inflow.u_star": float(inflow.get("u_star", np.nan)) if inflow else np.nan,
        "inflow.z0_eff": float(inflow.get("z0_eff", np.nan)) if inflow else np.nan,
        "inflow.T_ref": float(inflow.get("T_ref", np.nan)) if inflow else np.nan,
        "inflow.q_ref": float(inflow.get("q_ref", np.nan)) if inflow and inflow.get("q_ref") is not None else np.nan,
        "inflow.Ri_b": float(inflow.get("Ri_b", np.nan)) if inflow else np.nan,
        "inflow.wind_dir": float(inflow.get("wind_dir", np.nan)) if inflow else np.nan,
        "inflow.t2m_K": float(inflow.get("t2m_K", np.nan)) if inflow else np.nan,
        "inflow.d2m_K": float(inflow.get("d2m_K", np.nan)) if inflow else np.nan,
        "inflow.u10_ms": float(inflow.get("u10_ms", np.nan)) if inflow else np.nan,
        "inflow.v10_ms": float(inflow.get("v10_ms", np.nan)) if inflow else np.nan,
        # solver
        "solver.n_iter": int(solver_diag["n_iter"]),
        "solver.wall_time_s": float(solver_diag["wall_time_s"]),
        "solver.final_residual_U": float(solver_diag["final_residual_U"]),
        "solver.final_residual_p": float(solver_diag["final_residual_p"]),
        "solver.final_residual_T": float(solver_diag["final_residual_T"]),
        "solver.converged": bool(solver_diag["converged"]),
        # export info
        "export.schema_version": 1,
        "export.utc_datetime": dt.datetime.utcnow().isoformat() + "Z",
        "export.script": "export_to_grid_zarr.py",
        "export.wall_time_s": float(time.time() - t0),
    }
    store.attrs.update({k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                        for k, v in attrs.items()})

    logger.info("%s → %s (%.1fs, cells=%d, converged=%s)",
                case_name, out_path.name, time.time() - t0, len(x), solver_diag["converged"])
    return True


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--time", type=str, required=True, help="OF time directory name")
    parser.add_argument("--out", type=Path, required=True, help="grid.zarr output path")
    parser.add_argument("--site-manifest", type=Path, default=None)
    parser.add_argument("--campaign-manifest", type=Path, default=None)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--half-extent-m", type=float, default=DEFAULT_HALF_EXTENT_M)
    parser.add_argument("--r-context", type=float, default=DEFAULT_R_CONTEXT)
    parser.add_argument("--include-turb", nargs="*", default=[],
                        choices=["k", "epsilon", "nut"])
    parser.add_argument("--device", type=str, default="cuda",
                        help="torch device for IDW; 'cpu' forces scipy fallback")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s")

    ok = export_case_to_grid(
        case_dir=args.case_dir,
        time_name=args.time,
        out_path=args.out,
        site_manifest=args.site_manifest,
        campaign_manifest=args.campaign_manifest,
        grid_size=args.grid_size,
        half_extent_m=args.half_extent_m,
        r_context=args.r_context,
        include_turb=tuple(args.include_turb),
        device=args.device,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
