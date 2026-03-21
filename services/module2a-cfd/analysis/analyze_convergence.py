"""
analyze_convergence.py — Post-processing for the CFD convergence study

Loads completed simulation results + Perdigão observations and computes:
  - Phase 1: RMSE vs resolution (convergence curve), GCI index
  - Phase 2: RMSE vs domain size, boundary effect check
  - Phase 3: physics config comparison (barplot, marginal error)
  - Phase 4: precursor vs idealized profiles

Figures are in the style of Venkatraman et al. (WES 2023):
  vertical profiles of U, TKE, wind direction at validation towers.

Usage
-----
    python analyze_convergence.py --phase mesh_convergence
    python analyze_convergence.py --phase all --output figures/convergence/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Observation loading
# ---------------------------------------------------------------------------

def load_perdigao_obs(
    obs_zarr: Path,
    timestamp: str,
    tower_ids: list[str],
) -> dict:
    """Load Perdigão tower observations at a given timestamp.

    Returns dict of {tower_id: {"heights": [...], "speed": [...], "direction": [...]}}.
    """
    import zarr
    obs = {}

    try:
        store = zarr.open_group(str(obs_zarr), mode="r")
    except Exception as e:
        logger.warning("Cannot open obs zarr %s: %s", obs_zarr, e)
        return obs

    times = np.array(store["coords/time"][:])
    site_ids_raw = store["coords/site_id"][:]
    site_ids = [s.decode("ascii") if isinstance(s, bytes) else s for s in site_ids_raw]
    heights = np.array(store["coords/height_m"][:])

    # Convert timestamp to int64
    ts_int = np.datetime64(timestamp).astype("datetime64[ns]").astype(np.int64)
    t_idx = np.argmin(np.abs(times - ts_int))

    u = np.array(store["sites/u"][t_idx])  # (site, height)
    v = np.array(store["sites/v"][t_idx])

    for tid in tower_ids:
        if tid not in site_ids:
            logger.warning("Tower %s not in obs", tid)
            continue
        s_idx = site_ids.index(tid)
        u_t = u[s_idx, :]
        v_t = v[s_idx, :]
        speed = np.sqrt(u_t**2 + v_t**2)
        direction = (270.0 - np.degrees(np.arctan2(v_t, u_t))) % 360.0

        obs[tid] = {
            "heights": heights.tolist(),
            "speed": speed.tolist(),
            "direction": direction.tolist(),
        }

    return obs


# ---------------------------------------------------------------------------
# CFD result loading
# ---------------------------------------------------------------------------

def load_cfd_profiles(
    case_dir: Path,
    tower_coords: dict,
    site_lat: float,
    site_lon: float,
) -> dict:
    """Extract vertical profiles at tower locations from a CFD case.

    Reads the last time step, converts cell centres to lat/lon, and
    bins cells near each tower into vertical profiles.

    Returns dict of {tower_id: {"heights": [...], "speed": [...], "direction": [...]}}.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from init_from_era5 import read_cell_centres

    centres = read_cell_centres(case_dir)

    # Read last timestep U field
    time_dirs = sorted(
        [d for d in case_dir.iterdir()
         if d.is_dir() and d.name.replace('.', '').isdigit() and float(d.name) > 0],
        key=lambda d: float(d.name),
    )
    if not time_dirs:
        logger.warning("No time directories in %s", case_dir)
        return {}

    latest = time_dirs[-1]
    from init_from_era5 import _parse_of_vector_field
    U = _parse_of_vector_field(latest / "U")

    DEG_PER_M_LAT = 1.0 / 111_000.0
    DEG_PER_M_LON = 1.0 / (111_000.0 * np.cos(np.radians(site_lat)))

    profiles = {}
    for tid, tinfo in tower_coords.items():
        t_lat = tinfo["lat"]
        t_lon = tinfo["lon"]

        # Convert tower lat/lon to local x, y
        t_x = (t_lon - site_lon) / DEG_PER_M_LON
        t_y = (t_lat - site_lat) / DEG_PER_M_LAT

        # Find cells within a radius (adaptive to resolution)
        dx = centres[:, 0] - t_x
        dy = centres[:, 1] - t_y
        dist_xy = np.sqrt(dx**2 + dy**2)

        # Use 200m radius or adaptive
        radius = 200.0
        mask = dist_xy < radius

        if np.sum(mask) < 5:
            # Try larger radius
            radius = 500.0
            mask = dist_xy < radius

        if np.sum(mask) < 2:
            logger.warning("Tower %s: too few cells within %gm", tid, radius)
            continue

        z_sel = centres[mask, 2]
        u_sel = U[mask, 0]
        v_sel = U[mask, 1]

        # Bin by height
        obs_heights = tinfo.get("heights_m", [10, 20, 40, 60, 80, 100])
        z_edges = np.concatenate([[0], np.array(obs_heights) + 5, [3000]])
        z_mids = np.array(obs_heights, dtype=float)

        speed_prof = np.full(len(z_mids), np.nan)
        dir_prof = np.full(len(z_mids), np.nan)

        for i in range(len(z_mids)):
            bin_mask = (z_sel >= z_edges[i]) & (z_sel < z_edges[i + 1])
            if np.any(bin_mask):
                u_mean = u_sel[bin_mask].mean()
                v_mean = v_sel[bin_mask].mean()
                speed_prof[i] = np.sqrt(u_mean**2 + v_mean**2)
                dir_prof[i] = (270.0 - np.degrees(np.arctan2(v_mean, u_mean))) % 360.0

        profiles[tid] = {
            "heights": z_mids.tolist(),
            "speed": speed_prof.tolist(),
            "direction": dir_prof.tolist(),
        }

    return profiles


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(obs: dict, cfd: dict) -> dict:
    """Compute RMSE, bias, and hit rate between obs and CFD profiles."""
    metrics = {}
    for tid in obs:
        if tid not in cfd:
            continue
        obs_speed = np.array(obs[tid]["speed"])
        cfd_speed = np.array(cfd[tid]["speed"])

        valid = np.isfinite(obs_speed) & np.isfinite(cfd_speed)
        if np.sum(valid) == 0:
            continue

        diff = cfd_speed[valid] - obs_speed[valid]
        rmse = np.sqrt(np.mean(diff**2))
        bias = np.mean(diff)

        # Hit rate: fraction of points within 2 m/s or 30%
        threshold = np.maximum(2.0, 0.3 * obs_speed[valid])
        hit_rate = np.mean(np.abs(diff) < threshold)

        metrics[tid] = {
            "rmse": float(rmse),
            "bias": float(bias),
            "hit_rate": float(hit_rate),
            "n_points": int(np.sum(valid)),
        }

    # Average across towers
    if metrics:
        avg_rmse = np.mean([m["rmse"] for m in metrics.values()])
        avg_bias = np.mean([m["bias"] for m in metrics.values()])
        avg_hr = np.mean([m["hit_rate"] for m in metrics.values()])
        metrics["_average"] = {
            "rmse": float(avg_rmse),
            "bias": float(avg_bias),
            "hit_rate": float(avg_hr),
        }

    return metrics


def grid_convergence_index(
    rmse_coarse: float,
    rmse_fine: float,
    ratio: float = 2.0,
    p: float = 2.0,
    Fs: float = 1.25,
) -> float:
    """Compute Grid Convergence Index (GCI) between two mesh levels.

    GCI_fine = Fs * |eps| / (r^p - 1)
    where eps = (rmse_fine - rmse_coarse) / rmse_coarse, r = ratio.
    """
    if rmse_coarse == 0:
        return 0.0
    eps = abs(rmse_fine - rmse_coarse) / rmse_coarse
    gci = Fs * eps / (ratio**p - 1)
    return float(gci)


# ---------------------------------------------------------------------------
# Phase analysis
# ---------------------------------------------------------------------------

def analyze_mesh_convergence(manifest: dict, cases_dir: Path, obs: dict,
                             tower_coords: dict, site_lat: float, site_lon: float) -> dict:
    """Analyze Phase 1: RMSE vs resolution."""
    results = {}
    phase_cases = {k: v for k, v in manifest.items() if v["phase"] == "mesh_convergence"}

    for cid, info in sorted(phase_cases.items()):
        case_dir = cases_dir / cid
        if not case_dir.exists():
            logger.warning("Case dir missing: %s", cid)
            continue

        cfd = load_cfd_profiles(case_dir, tower_coords, site_lat, site_lon)
        metrics = compute_metrics(obs, cfd)

        results[cid] = {
            "resolution_m": info["resolution_m"],
            "direction_deg": info["direction_deg"],
            "metrics": metrics,
        }
        avg = metrics.get("_average", {})
        logger.info(
            "%s: %gm, %g° → RMSE=%.2f, bias=%.2f, HR=%.1f%%",
            cid, info["resolution_m"], info["direction_deg"],
            avg.get("rmse", -1), avg.get("bias", -1), avg.get("hit_rate", -1) * 100,
        )

    # Compute GCI between consecutive resolutions
    for direction in [231, 40]:
        dir_cases = sorted(
            [(k, v) for k, v in results.items() if v["direction_deg"] == direction],
            key=lambda x: -x[1]["resolution_m"],  # coarse to fine
        )
        for i in range(len(dir_cases) - 1):
            coarse_rmse = dir_cases[i][1]["metrics"].get("_average", {}).get("rmse", 0)
            fine_rmse = dir_cases[i + 1][1]["metrics"].get("_average", {}).get("rmse", 0)
            ratio = dir_cases[i][1]["resolution_m"] / dir_cases[i + 1][1]["resolution_m"]
            gci = grid_convergence_index(coarse_rmse, fine_rmse, ratio)
            results[dir_cases[i + 1][0]]["gci"] = gci
            logger.info("GCI %s→%s: %.1f%%", dir_cases[i][0], dir_cases[i + 1][0], gci * 100)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Analyze convergence study results")
    parser.add_argument("--phase", default="all",
                        help="Phase to analyze (mesh_convergence, domain_sensitivity, etc. or 'all')")
    parser.add_argument("--manifest", default=None,
                        help="Manifest JSON (default: data/convergence/cases/convergence_study_manifest.json)")
    parser.add_argument("--obs-zarr", default=None,
                        help="Perdigão obs zarr (default: data/raw/perdigao_obs.zarr)")
    parser.add_argument("--output", default="figures/convergence",
                        help="Output directory for figures and tables")
    args = parser.parse_args()

    # Load manifest
    manifest_path = Path(args.manifest) if args.manifest else (
        PROJECT_ROOT / "data" / "convergence" / "cases" / "convergence_study_manifest.json"
    )
    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        return
    with open(manifest_path) as f:
        manifest = json.load(f)

    logger.info("Loaded manifest: %d cases", len(manifest))

    # Load observations
    obs_zarr = Path(args.obs_zarr) if args.obs_zarr else (
        PROJECT_ROOT / "data" / "raw" / "perdigao_obs.zarr"
    )

    # Load study config for tower info
    study_cfg_path = PROJECT_ROOT / "configs" / "convergence_study.yaml"
    import yaml
    with open(study_cfg_path) as f:
        study_cfg = yaml.safe_load(f)

    # Load tower coordinates
    towers_path = PROJECT_ROOT / "configs" / "sites" / "perdigao_towers.yaml"
    if towers_path.exists():
        with open(towers_path) as f:
            towers_cfg = yaml.safe_load(f)
        tower_coords = towers_cfg.get("towers", {})
    else:
        tower_coords = {}
        logger.warning("Tower config not found: %s", towers_path)

    # Validation towers
    val_towers = study_cfg["study"]["validation_towers"]
    tower_subset = {k: v for k, v in tower_coords.items() if k in val_towers}

    # Load obs
    timestamp = study_cfg["study"]["timestamp"]
    obs = {}
    if obs_zarr.exists():
        obs = load_perdigao_obs(obs_zarr, timestamp, val_towers)
        logger.info("Loaded obs for %d towers at %s", len(obs), timestamp)
    else:
        logger.warning("Obs zarr not found: %s — metrics will be empty", obs_zarr)

    # Site coordinates
    site_cfg_path = PROJECT_ROOT / "configs" / "sites" / "perdigao.yaml"
    with open(site_cfg_path) as f:
        site_cfg = yaml.safe_load(f)
    site = site_cfg["site"]
    site_lat = site["coordinates"]["latitude"]
    site_lon = site["coordinates"]["longitude"]

    cases_dir = manifest_path.parent

    # Output
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run analyses
    phases = [args.phase] if args.phase != "all" else [
        "mesh_convergence", "domain_sensitivity", "physics_comparison",
    ]

    all_results = {}
    for phase in phases:
        if phase == "mesh_convergence":
            results = analyze_mesh_convergence(
                manifest, cases_dir, obs, tower_subset, site_lat, site_lon,
            )
            all_results[phase] = results

    # Save results
    results_path = output_dir / "convergence_analysis.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Analysis results → %s", results_path)


if __name__ == "__main__":
    main()
