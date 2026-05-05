"""
convert_stacked_to_training.py — Convert stacked campaign Zarr to per-case training format.

Reads stacked Zarr stores (one per site, 15 timestamps × 165k cells) and produces
the per-case format expected by train.py:
  - {case_id}/unstructured.zarr/  (for GNN/MLP)
  - {case_id}/grid.zarr/          (for U-Net 3D, 128×128×32)
  - {case_id}/inflow.json

GPU-accelerated: uses torch.cdist for k-NN IDW interpolation on GPU.
Interpolation weights are computed once per site and reused for all timestamps.

Usage
-----
    cd services/module2b-surrogate
    python convert_stacked_to_training.py \
        --input  /path/to/campaign_1500/ \
        --output /path/to/cfd-database/training_1500/
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# 32 log-spaced AGL levels [m]: dense near ground, sparse at top
Z_LEVELS_AGL = np.geomspace(5, 5000, 32).astype(np.float32)

CMU = 0.09  # k-epsilon model constant


# ── GPU k-NN IDW interpolation ──────────────────────────────────────


def build_idw_weights_gpu(
    src_pts: np.ndarray,
    tgt_pts: np.ndarray,
    k: int = 8,
    power: float = 2.0,
    z_scale: float = 1.0,
    device: str = "cuda",
    chunk_size: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build k-NN IDW interpolation weights on GPU.

    Parameters
    ----------
    src_pts : (N_src, 3) source points (x, y, z_agl)
    tgt_pts : (N_tgt, 3) target points (x, y, z_agl)
    k : number of nearest neighbors
    power : IDW power (2.0 = inverse distance squared)
    z_scale : scaling factor for z-axis (to balance with x,y)
    device : torch device
    chunk_size : process target points in chunks to limit VRAM

    Returns
    -------
    indices : (N_tgt, k) int64 — indices into src_pts
    weights : (N_tgt, k) float32 — normalized IDW weights
    """
    src_t = torch.from_numpy(src_pts.astype(np.float32)).to(device)
    # Scale z for anisotropic distance
    src_t[:, 2] *= z_scale

    n_tgt = len(tgt_pts)
    all_indices = torch.empty(n_tgt, k, dtype=torch.long)
    all_weights = torch.empty(n_tgt, k, dtype=torch.float32)

    for i0 in range(0, n_tgt, chunk_size):
        i1 = min(i0 + chunk_size, n_tgt)
        tgt_chunk = torch.from_numpy(tgt_pts[i0:i1].astype(np.float32)).to(device)
        tgt_chunk[:, 2] *= z_scale

        # (chunk, N_src) pairwise distances
        dists = torch.cdist(tgt_chunk, src_t)  # (chunk, N_src)

        # k smallest distances
        topk_dists, topk_idx = torch.topk(dists, k, dim=1, largest=False)

        # IDW weights: w_i = 1/d_i^p, normalized
        # Clamp to avoid division by zero for coincident points
        topk_dists = topk_dists.clamp(min=1e-6)
        w = 1.0 / topk_dists.pow(power)
        w_sum = w.sum(dim=1, keepdim=True)
        w = w / w_sum

        all_indices[i0:i1] = topk_idx.cpu()
        all_weights[i0:i1] = w.cpu()

    return all_indices, all_weights


def apply_idw(
    values: np.ndarray,
    indices: torch.Tensor,
    weights: torch.Tensor,
) -> np.ndarray:
    """Apply precomputed IDW weights to interpolate field values.

    Parameters
    ----------
    values : (N_src,) or (N_src, C) field values
    indices : (N_tgt, k) neighbor indices
    weights : (N_tgt, k) normalized weights

    Returns
    -------
    interpolated : (N_tgt,) or (N_tgt, C) float32
    """
    is_vector = values.ndim == 2
    if is_vector:
        # (N_tgt, k, C)
        neighbor_vals = values[indices.numpy()]
        # (N_tgt, k, 1) * (N_tgt, k, C) → sum over k
        result = (weights.numpy()[:, :, None] * neighbor_vals).sum(axis=1)
    else:
        neighbor_vals = values[indices.numpy()]  # (N_tgt, k)
        result = (weights.numpy() * neighbor_vals).sum(axis=1)
    return result.astype(np.float32)


# ── Helpers ─────────────────────────────────────────────────────────


def filter_zone(x, y, r_max):
    """Return boolean mask for cells within horizontal radius."""
    return np.sqrt(x**2 + y**2) < r_max


def interpolate_era5_profiles(inflow: dict, z_levels: np.ndarray) -> dict:
    """Interpolate ERA5 inflow profiles at target z-levels."""
    z_src = np.array(inflow["z_levels"], dtype=np.float64)

    ux_src = np.array(inflow["ux_profile"], dtype=np.float64)
    uy_src = np.array(inflow["uy_profile"], dtype=np.float64)

    u_interp = np.interp(z_levels, z_src, ux_src).astype(np.float32)
    v_interp = np.interp(z_levels, z_src, uy_src).astype(np.float32)

    T_interp = None
    if "T_profile" in inflow:
        T_src = np.array(inflow["T_profile"], dtype=np.float64)
        T_interp = np.interp(z_levels, z_src, T_src).astype(np.float32)

    q_interp = None
    if "q_profile" in inflow:
        q_src = np.array(inflow["q_profile"], dtype=np.float64)
        q_interp = np.interp(z_levels, z_src, q_src).astype(np.float32)

    u_star = float(inflow.get("u_star", 0.3))
    k_val = u_star**2 / np.sqrt(CMU)
    k_interp = np.full(len(z_levels), k_val, dtype=np.float32)

    return {"u": u_interp, "v": v_interp, "T": T_interp, "q": q_interp, "k": k_interp}


def build_target_grid(half_extent: float, grid_size: int, z_levels: np.ndarray):
    """Build 3D target grid points for interpolation.

    Returns
    -------
    target_pts : (ny*nx*nz, 3) float32
    x_1d, y_1d : (grid_size,) coordinate arrays
    """
    grid_res = 2 * half_extent / grid_size
    x_1d = np.linspace(-half_extent + grid_res / 2, half_extent - grid_res / 2, grid_size)
    y_1d = np.linspace(-half_extent + grid_res / 2, half_extent - grid_res / 2, grid_size)

    # Build full 3D target grid
    xg, yg, zg = np.meshgrid(x_1d, y_1d, z_levels, indexing="ij")
    # Reshape: (nx*ny*nz, 3)
    target_pts = np.column_stack([xg.ravel(), yg.ravel(), zg.ravel()]).astype(np.float32)

    return target_pts, x_1d, y_1d


def write_grid_zarr(
    output_path: Path,
    *,
    U_grid: np.ndarray,
    T_grid: np.ndarray | None,
    q_grid: np.ndarray | None,
    terrain_2d: np.ndarray,
    z0_2d: np.ndarray,
    x_1d: np.ndarray,
    y_1d: np.ndarray,
    z_levels: np.ndarray,
    era5_profiles: dict | None,
    half_extent: float,
    grid_size: int,
):
    """Write grid.zarr with target, residual, input, and coords."""
    import zarr

    ny = nx = grid_size
    nz = len(z_levels)
    grid_res = 2 * half_extent / grid_size

    # Compute residuals
    U_residual = T_residual = q_residual = None
    if era5_profiles is not None:
        U_residual = U_grid.copy()
        for iz in range(nz):
            U_residual[:, :, iz, 0] -= era5_profiles["u"][iz]
            U_residual[:, :, iz, 1] -= era5_profiles["v"][iz]

        if T_grid is not None and era5_profiles["T"] is not None:
            T_residual = T_grid.copy()
            for iz in range(nz):
                T_residual[:, :, iz] -= era5_profiles["T"][iz]

        if q_grid is not None and era5_profiles["q"] is not None:
            q_residual = q_grid.copy()
            for iz in range(nz):
                q_residual[:, :, iz] -= era5_profiles["q"][iz]

    store = zarr.open_group(str(output_path), mode="w")

    coords_grp = store.create_group("coords")
    coords_grp.create_array("x_1d", data=x_1d.astype(np.float32))
    coords_grp.create_array("y_1d", data=y_1d.astype(np.float32))
    coords_grp.create_array("z_levels_agl", data=z_levels)

    inp = store.create_group("input")
    inp.create_array("terrain", data=terrain_2d)
    inp.create_array("z0", data=z0_2d)
    if era5_profiles is not None:
        era5_grp = inp.create_group("era5")
        for var in ["u", "v", "T", "q", "k"]:
            val = era5_profiles.get(var)
            if val is not None:
                era5_grp.create_array(var, data=val)

    tgt = store.create_group("target")
    tgt.create_array("U", data=U_grid)
    if T_grid is not None:
        tgt.create_array("T", data=T_grid)
    if q_grid is not None:
        tgt.create_array("q", data=q_grid)

    if U_residual is not None:
        res = store.create_group("residual")
        res.create_array("U", data=U_residual)
        if T_residual is not None:
            res.create_array("T", data=T_residual)
        if q_residual is not None:
            res.create_array("q", data=q_residual)

    store.attrs.update({
        "grid_res_m": float(grid_res),
        "half_extent_m": float(half_extent),
        "nx": nx, "ny": ny, "nz": nz,
        "inner_pad": (grid_size - grid_size // 2) // 2,
        "prediction_extent_m": float(half_extent),
    })


# ── Site processing ──────────────────────────────────────────────────


def process_site(
    site_zarr: Path,
    output_dir: Path,
    half_extent: float,
    grid_size: int,
    r_context: float,
    r_fine: float,
    device: str = "cuda",
    skip_existing: bool = True,
) -> list[str]:
    """Process one site: build GPU IDW weights once, apply to all timestamps.

    Returns list of exported case_ids.
    """
    import zarr

    site_id = site_zarr.parent.name
    store = zarr.open_group(str(site_zarr), mode="r")
    t_site = time.time()

    # Read shared coordinates
    x = np.array(store["coords/x"][:], dtype=np.float32)
    y = np.array(store["coords/y"][:], dtype=np.float32)
    z = np.array(store["coords/z"][:], dtype=np.float32)
    z_agl = np.array(store["coords/z_agl"][:], dtype=np.float32)
    elev = np.array(store["coords/elev"][:], dtype=np.float32)

    n_ts = store["U"].shape[0]

    # Read metadata
    meta_case_ids = np.array(store["meta/case_id"][:])
    meta_u_hub = np.array(store["meta/u_hub"][:], dtype=np.float32)
    meta_wind_dir = np.array(store["meta/wind_dir"][:], dtype=np.float32)
    meta_T_ref = np.array(store["meta/T_ref"][:], dtype=np.float32)
    meta_q_ref = np.array(store["meta/q_ref"][:], dtype=np.float32) if "meta/q_ref" in store else None
    meta_Ri_b = np.array(store["meta/Ri_b"][:], dtype=np.float32) if "meta/Ri_b" in store else None
    meta_u_star = np.array(store["meta/u_star"][:], dtype=np.float32) if "meta/u_star" in store else None
    meta_z0_eff = np.array(store["meta/z0_eff"][:], dtype=np.float32) if "meta/z0_eff" in store else None

    # Inflow profiles
    z_levels_inflow = np.array(store["inflow/z_levels"][:], dtype=np.float32)
    ux_profiles = np.array(store["inflow/ux_profile"][:], dtype=np.float32)
    uy_profiles = np.array(store["inflow/uy_profile"][:], dtype=np.float32)
    T_profiles = np.array(store["inflow/T_profile"][:], dtype=np.float32) if "inflow/T_profile" in store else None
    q_profiles = np.array(store["inflow/q_profile"][:], dtype=np.float32) if "inflow/q_profile" in store else None

    has_T = "T" in store
    has_q = "q" in store

    # Zone masks
    mask_fine = filter_zone(x, y, r_fine)
    mask_ctx = filter_zone(x, y, r_context)

    x_ctx = x[mask_ctx]
    y_ctx = y[mask_ctx]
    z_agl_ctx = z_agl[mask_ctx]
    elev_ctx = elev[mask_ctx]

    n_ctx = int(mask_ctx.sum())
    n_fine = int(mask_fine.sum())
    logger.info("%s: %d cells, ctx=%d, fine=%d, ts=%d", site_id, len(x), n_ctx, n_fine, n_ts)

    # ── Build GPU IDW weights (ONCE per site) ──
    src_pts = np.column_stack([x_ctx, y_ctx, z_agl_ctx])
    target_pts, x_1d, y_1d = build_target_grid(half_extent, grid_size, Z_LEVELS_AGL)

    t0 = time.time()
    grid_idx, grid_w = build_idw_weights_gpu(
        src_pts, target_pts, k=8, power=2.0, z_scale=1.0, device=device,
    )
    logger.info("%s: GPU IDW weights built in %.1fs", site_id, time.time() - t0)

    # Terrain 2D: interpolate from 2D points (use 2D IDW, k=4)
    pts_2d_src = np.column_stack([x_ctx, y_ctx, np.zeros(n_ctx, dtype=np.float32)])
    xg, yg = np.meshgrid(x_1d, y_1d)
    pts_2d_tgt = np.column_stack([
        xg.ravel(), yg.ravel(), np.zeros(grid_size * grid_size, dtype=np.float32)
    ]).astype(np.float32)
    terr_idx, terr_w = build_idw_weights_gpu(
        pts_2d_src, pts_2d_tgt, k=4, power=2.0, device=device,
    )
    terrain_2d = apply_idw(elev_ctx, terr_idx, terr_w).reshape(grid_size, grid_size)

    # Shape for reshaping grid results
    # target_pts was built with meshgrid(x, y, z, indexing='ij') → (nx, ny, nz)
    # We need (ny, nx, nz) for the dataset → transpose
    nx = ny = grid_size
    nz = len(Z_LEVELS_AGL)

    exported = []

    for ts_idx in range(n_ts):
        case_id_raw = meta_case_ids[ts_idx]
        if isinstance(case_id_raw, bytes):
            case_id_raw = case_id_raw.decode()
        case_id = f"{site_id}_{case_id_raw}"
        case_dir = output_dir / case_id

        if skip_existing and (case_dir / "grid.zarr").exists() and (case_dir / "unstructured.zarr").exists():
            exported.append(case_id)
            continue

        case_dir.mkdir(parents=True, exist_ok=True)

        # Read fields
        U_all = np.array(store["U"][ts_idx], dtype=np.float32)
        T_all = np.array(store["T"][ts_idx], dtype=np.float32) if has_T else None
        q_all = np.array(store["q"][ts_idx], dtype=np.float32) if has_q else None
        k_all = np.array(store["k"][ts_idx], dtype=np.float32) if "k" in store else None
        nut_all = np.array(store["nut"][ts_idx], dtype=np.float32) if "nut" in store else None

        # Build inflow dict
        inflow = {
            "z_levels": z_levels_inflow[ts_idx].tolist(),
            "ux_profile": ux_profiles[ts_idx].tolist(),
            "uy_profile": uy_profiles[ts_idx].tolist(),
            "u_hub": float(meta_u_hub[ts_idx]),
            "wind_dir": float(meta_wind_dir[ts_idx]),
            "T_ref": float(meta_T_ref[ts_idx]),
            "u_star": float(meta_u_star[ts_idx]) if meta_u_star is not None else 0.3,
        }
        if T_profiles is not None:
            inflow["T_profile"] = T_profiles[ts_idx].tolist()
        if q_profiles is not None:
            inflow["q_profile"] = q_profiles[ts_idx].tolist()
        if meta_q_ref is not None:
            inflow["q_ref"] = float(meta_q_ref[ts_idx])
        if meta_Ri_b is not None:
            inflow["Ri_b"] = float(meta_Ri_b[ts_idx])

        with open(case_dir / "inflow.json", "w") as f:
            json.dump(inflow, f, indent=2)

        # ── Unstructured export (fine zone) ──
        unstruct_store = zarr.open_group(str(case_dir / "unstructured.zarr"), mode="w")
        unstruct_store.create_array("x", data=x[mask_fine])
        unstruct_store.create_array("y", data=y[mask_fine])
        unstruct_store.create_array("z", data=z[mask_fine])
        unstruct_store.create_array("z_agl", data=z_agl[mask_fine])
        unstruct_store.create_array("elev", data=elev[mask_fine])
        unstruct_store.create_array("U", data=U_all[mask_fine])
        if k_all is not None:
            unstruct_store.create_array("k", data=k_all[mask_fine])
        if nut_all is not None:
            unstruct_store.create_array("nut", data=nut_all[mask_fine])
        if T_all is not None:
            unstruct_store.create_array("T", data=T_all[mask_fine])
        if q_all is not None:
            unstruct_store.create_array("q", data=q_all[mask_fine])
        unstruct_store.attrs["n_cells"] = n_fine

        # ── Grid export (GPU-accelerated IDW) ──
        U_ctx = U_all[mask_ctx]
        U_grid_flat = apply_idw(U_ctx, grid_idx, grid_w)  # (N_tgt, 3)
        # Reshape: meshgrid was (nx, ny, nz) with indexing='ij' → transpose to (ny, nx, nz)
        U_grid = U_grid_flat.reshape(nx, ny, nz, 3).transpose(1, 0, 2, 3).copy()

        T_grid = None
        if T_all is not None:
            T_ctx = T_all[mask_ctx]
            T_grid = apply_idw(T_ctx, grid_idx, grid_w).reshape(nx, ny, nz).transpose(1, 0, 2).copy()

        q_grid = None
        if q_all is not None:
            q_ctx = q_all[mask_ctx]
            q_grid = apply_idw(q_ctx, grid_idx, grid_w).reshape(nx, ny, nz).transpose(1, 0, 2).copy()

        z0_val = float(meta_z0_eff[ts_idx]) if meta_z0_eff is not None else 0.05
        z0_2d = np.full((ny, nx), z0_val, dtype=np.float32)

        era5_profiles = interpolate_era5_profiles(inflow, Z_LEVELS_AGL)

        write_grid_zarr(
            case_dir / "grid.zarr",
            U_grid=U_grid, T_grid=T_grid, q_grid=q_grid,
            terrain_2d=terrain_2d, z0_2d=z0_2d,
            x_1d=x_1d, y_1d=y_1d, z_levels=Z_LEVELS_AGL,
            era5_profiles=era5_profiles,
            half_extent=half_extent, grid_size=grid_size,
        )

        exported.append(case_id)

    elapsed = time.time() - t_site
    logger.info("%s: %d cases exported in %.1fs (%.2fs/case)",
                site_id, len(exported), elapsed, elapsed / max(len(exported), 1))
    return exported


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Convert stacked campaign Zarr to per-case training format (GPU-accelerated)"
    )
    parser.add_argument("--input", required=True, help="Campaign directory with site_*/site_*.zarr")
    parser.add_argument("--output", required=True, help="Output directory for training data")
    parser.add_argument("--half-extent", type=float, default=2000.0)
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--r-fine", type=float, default=1000.0)
    parser.add_argument("--r-context", type=float, default=3500.0)
    parser.add_argument("--device", default="cuda", help="torch device for IDW (cuda or cpu)")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    site_zarrs = sorted(input_dir.glob("site_*/site_*.zarr"))
    if not site_zarrs:
        logger.error("No site Zarr stores found in %s", input_dir)
        return

    logger.info("Found %d site Zarr stores, device=%s", len(site_zarrs), args.device)

    total_exported = 0
    t_total = time.time()

    # Process sites sequentially (GPU already parallelizes the heavy work)
    for sz in site_zarrs:
        try:
            exported = process_site(
                sz, output_dir,
                args.half_extent, args.grid_size,
                args.r_context, args.r_fine,
                device=args.device,
                skip_existing=args.skip_existing,
            )
            total_exported += len(exported)
        except Exception as e:
            logger.error("Failed %s: %s", sz.parent.name, e, exc_info=True)

    elapsed = time.time() - t_total
    logger.info("Total: %d cases exported in %.0fs (%.1f cases/s)",
                total_exported, elapsed, total_exported / max(elapsed, 1))


if __name__ == "__main__":
    main()
