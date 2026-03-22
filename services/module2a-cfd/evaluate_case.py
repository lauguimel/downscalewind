"""
evaluate_case.py — Evaluate converged OpenFOAM cases: convergence + profiles + ERA5 comparison.

Produces standardized diagnostic reports for cross-case comparison:
  1. Convergence plots (residuals + field averages/min/max vs iteration)
  2. Vertical profiles at 5 probes (centre, ERA5, ridges, valley) vs ERA5
  3. Boundary-layer zoom (0–500m AGL)
  4. Summary metrics JSON (RMSE, bias, speedup, timing)

Output convention:
  data/validation/{study_name}/{case_id}/
    convergence.png
    profiles.png
    boundary_layer.png
    metrics.json

Usage
-----
    # Single case (auto-detect output from case path)
    python evaluate_case.py --case data/cases/poc_mesh_convergence/case_res100

    # Explicit output + label
    python evaluate_case.py --case data/cases/study/case_A --label "Case A" \
        --output data/validation/study/case_A

    # Batch: all cases in a study folder
    python evaluate_case.py --batch data/cases/poc_mesh_convergence

    # Compare: overlay metrics from multiple evaluations
    python evaluate_case.py --compare \
        data/validation/study/case_A/metrics.json \
        data/validation/study/case_B/metrics.json

    # Programmatic (from run_sf_poc.py / run_local_study.py)
    from evaluate_case import evaluate_single
    metrics = evaluate_single(case_dir, output_dir, label)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Probe locations (Perdigão local coordinates, relative to site centre)
# ---------------------------------------------------------------------------

DEFAULT_PROBES = {
    "centre":    (0.0, 0.0),
    "era5":      (-850.0, 3774.0),
    "sw_ridge":  (-500.0, 900.0),   # TSE04
    "valley":    (-200.0, 200.0),
    "ne_ridge":  (-800.0, -400.0),  # TSE13
}

PROFILE_HEIGHTS_AGL = np.array([
    5, 10, 20, 40, 60, 80, 100, 150, 200, 300, 500,
    700, 1000, 1500, 2000, 3000, 4000,
])


def make_grid_probes(
    extent_m: float = 2400.0,
    n: int = 10,
) -> dict[str, tuple[float, float]]:
    """Generate an n×n grid of probes in the fine zone for statistical comparison.

    Returns dict of probe_name → (x, y) coordinates.
    Default: 10×10 grid over 2.4×2.4 km = probes every 240m.
    """
    half = extent_m / 2
    spacing = extent_m / n
    probes = {}
    for i in range(n):
        for j in range(n):
            x = -half + spacing / 2 + i * spacing
            y = -half + spacing / 2 + j * spacing
            probes[f"g{i:02d}_{j:02d}"] = (x, y)
    return probes


# ---------------------------------------------------------------------------
# Output path convention
# ---------------------------------------------------------------------------

def default_output_dir(case_dir: Path) -> Path:
    """Derive output dir from case path: data/cases/X/case_Y → data/validation/X/Y."""
    case_dir = case_dir.resolve()
    case_id = case_dir.name.removeprefix("case_")
    study_name = case_dir.parent.name
    return ROOT / "data" / "validation" / study_name / case_id


def default_label(case_dir: Path) -> str:
    """Derive label from case directory name."""
    return case_dir.name.removeprefix("case_").replace("_", " ")


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def parse_residuals(log_path: Path) -> dict[str, np.ndarray]:
    """Parse initial residuals from solver log."""
    text = log_path.read_text()
    pattern = r"Solving for (\w+), Initial residual = ([\d.e+-]+)"
    data = defaultdict(list)
    for var, val in re.findall(pattern, text):
        data[var].append(float(val))
    return {k: np.array(v) for k, v in data.items()}


def parse_field_stats(log_path: Path) -> dict:
    """Parse volAverage and fieldMinMax from solver log."""
    text = log_path.read_text()
    stats = {}

    # volAverage U → speed
    pat = r"volAverage\(region0\) of U = \(([-\d.e+]+) ([-\d.e+]+) ([-\d.e+]+)\)"
    u_avg = np.array([(float(a), float(b), float(c)) for a, b, c in re.findall(pat, text)])
    if len(u_avg):
        stats["U_avg"] = u_avg
        stats["speed_avg"] = np.sqrt(u_avg[:, 0]**2 + u_avg[:, 1]**2 + u_avg[:, 2]**2)

    for var in ["k", "epsilon", "T"]:
        pat = rf"volAverage\(region0\) of {var} = ([-\d.e+]+)"
        vals = [float(x) for x in re.findall(pat, text)]
        if vals:
            stats[f"{var}_avg"] = np.array(vals)

    for key, pat in [
        ("speed_max", r"max\(mag\(U\)\) = ([\d.e+-]+)"),
        ("speed_min", r"min\(mag\(U\)\) = ([\d.e+-]+)"),
        ("k_max", r"max\(k\) = ([\d.e+-]+)"),
        ("k_min", r"min\(k\) = ([\d.e+-]+)"),
    ]:
        vals = [float(x) for x in re.findall(pat, text)]
        if vals:
            stats[key] = np.array(vals)

    return stats


# ---------------------------------------------------------------------------
# Profile extraction
# ---------------------------------------------------------------------------

def extract_profiles(
    case_dir: Path,
    probes: dict[str, tuple[float, float]],
    search_radius: float = 300.0,
) -> dict:
    """Extract vertical profiles at probe locations from OF fields."""
    import fluidfoam

    case_str = str(case_dir)
    time_dirs = sorted(
        [d for d in case_dir.iterdir()
         if d.is_dir() and d.name.replace(".", "").isdigit() and float(d.name) > 0],
        key=lambda d: float(d.name),
    )
    if not time_dirs:
        raise FileNotFoundError(f"No time directories in {case_dir}")
    latest = time_dirs[-1].name
    logger.info("Reading fields at time=%s", latest)

    x, y, z = fluidfoam.readmesh(case_str)
    n_cells = len(x)

    def _read_vector_field(field_name):
        """Read OF vector field, fallback to regex parsing if fluidfoam fails."""
        try:
            return np.array(fluidfoam.readvector(case_str, latest, field_name)).T
        except Exception:
            logger.info("fluidfoam failed for %s, using regex parser", field_name)
            text = (case_dir / latest / field_name).read_text()
            pattern = r'\(([-\d.e+]+)\s+([-\d.e+]+)\s+([-\d.e+]+)\)'
            matches = re.findall(pattern, text)
            arr = np.array([(float(a), float(b), float(c)) for a, b, c in matches])
            # Skip dimensions line if present, align to mesh size
            if len(arr) > n_cells:
                arr = arr[-n_cells:]
            elif len(arr) < n_cells:
                arr = np.pad(arr, ((0, n_cells - len(arr)), (0, 0)))
            return arr

    def _read_scalar_field(field_name):
        """Read OF scalar field, fallback to regex parsing if fluidfoam fails."""
        try:
            return np.array(fluidfoam.readscalar(case_str, latest, field_name))
        except Exception:
            fpath = case_dir / latest / field_name
            if not fpath.exists():
                return None
            logger.info("fluidfoam failed for %s, using regex parser", field_name)
            text = fpath.read_text()
            # Find the number before the data block
            m = re.search(r'(\d+)\s*\(', text)
            if not m:
                return None
            start = text.index('(', m.start()) + 1
            end = text.index(')', start)
            vals = np.array([float(v) for v in text[start:end].split()])
            if len(vals) < n_cells:
                vals = np.pad(vals, (0, n_cells - len(vals)))
            return vals[:n_cells]

    U = _read_vector_field("U")
    k = _read_scalar_field("k")
    if k is None:
        k = np.zeros(n_cells)

    T = _read_scalar_field("T")
    has_T = T is not None

    epsilon = _read_scalar_field("epsilon")
    if epsilon is None:
        epsilon = np.zeros(n_cells)

    profiles = {}
    for name, (px, py) in probes.items():
        dist_h = np.sqrt((x - px)**2 + (y - py)**2)
        col_mask = dist_h < search_radius
        if col_mask.sum() < 5:
            logger.warning("Probe %s: only %d cells within %.0fm — skipped", name, col_mask.sum(), search_radius)
            continue

        z_col = z[col_mask]
        z_terrain = z_col.min()
        z_agl = z_col - z_terrain
        idx = np.argsort(z_agl)

        U_col = U[col_mask][idx]
        prof = {
            "z_agl": z_agl[idx],
            "z_terrain": float(z_terrain),
            "speed": np.sqrt(U_col[:, 0]**2 + U_col[:, 1]**2),
            "ux": U_col[:, 0],
            "uy": U_col[:, 1],
            "w": U_col[:, 2],
            "k": k[col_mask][idx],
            "epsilon": epsilon[col_mask][idx],
            "n_cells": int(col_mask.sum()),
        }
        if has_T:
            prof["T"] = T[col_mask][idx]
        profiles[name] = prof

    return profiles


def load_era5_profile(inflow_json: Path) -> dict:
    """Load ERA5 inflow profile from JSON."""
    with open(inflow_json) as f:
        inflow = json.load(f)
    z = np.array(inflow["z_levels"])
    u = np.array(inflow["u_profile"])
    fx, fy = inflow["flowDir_x"], inflow["flowDir_y"]
    result = {
        "z_agl": z, "speed": u, "ux": u * fx, "uy": u * fy,
        "u_hub": inflow["u_hub"], "wind_dir": inflow["wind_dir"],
        "T_ref": inflow.get("T_ref", 288.0),
    }
    if "T_profile" in inflow:
        result["T"] = np.array(inflow["T_profile"])
    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(profiles: dict, era5: dict, label: str, case_dir: Path) -> dict:
    """Compute summary metrics for cross-case comparison."""
    metrics = {"label": label, "case_dir": str(case_dir)}

    # ERA5 comparison
    if "era5" in profiles:
        p = profiles["era5"]
        speed_era5_interp = np.interp(p["z_agl"], era5["z_agl"], era5["speed"])
        for zmin, zmax, tag in [(0, 200, "bl"), (200, 1000, "mid"), (1000, 6000, "upper")]:
            mask = (p["z_agl"] >= zmin) & (p["z_agl"] < zmax)
            if mask.sum() < 3:
                continue
            diff = p["speed"][mask] - speed_era5_interp[mask]
            metrics[f"bias_{tag}"] = float(np.mean(diff))
            metrics[f"rmse_{tag}"] = float(np.sqrt(np.mean(diff**2)))
            std_true = np.std(speed_era5_interp[mask])
            metrics[f"corr_{tag}"] = float(np.corrcoef(p["speed"][mask], speed_era5_interp[mask])[0, 1]) if std_true > 1e-8 else 0.0

    # Speedup at centre
    if "centre" in profiles:
        p = profiles["centre"]
        mask_100m = (p["z_agl"] > 80) & (p["z_agl"] < 120)
        if mask_100m.any():
            speed_100m = float(np.mean(p["speed"][mask_100m]))
            metrics["speed_centre_100m"] = speed_100m
            metrics["speedup_100m"] = speed_100m / max(era5["u_hub"], 0.1)

    # Probe terrain elevations
    for name, prof in profiles.items():
        metrics[f"z_terrain_{name}"] = prof["z_terrain"]

    # Solver log
    log_sf = case_dir / "log.simpleFoam"
    if log_sf.exists():
        residuals = parse_residuals(log_sf)
        if "Ux" in residuals:
            metrics["final_residual_Ux"] = float(residuals["Ux"][-1])
            metrics["n_iterations"] = len(residuals["Ux"])
        stats = parse_field_stats(log_sf)
        if "speed_avg" in stats:
            metrics["final_speed_avg"] = float(stats["speed_avg"][-1])
        if "speed_max" in stats:
            metrics["final_speed_max"] = float(stats["speed_max"][-1])

        text = log_sf.read_text()
        # Take the LAST occurrence (final total, not per-iteration)
        clock_times = re.findall(r"ClockTime = (\d+)", text)
        if clock_times:
            metrics["wall_time_s"] = int(clock_times[-1])
        exec_times = re.findall(r"ExecutionTime = ([\d.]+)", text)
        if exec_times:
            metrics["cpu_time_s"] = float(exec_times[-1])
        # Also capture potentialFoam time if present
        log_pf = case_dir / "log.potentialFoam"
        if log_pf.exists():
            pf_text = log_pf.read_text()
            pf_clock = re.findall(r"ClockTime = (\d+)", pf_text)
            if pf_clock:
                metrics["potentialfoam_time_s"] = int(pf_clock[-1])
                metrics["wall_time_s"] = metrics.get("wall_time_s", 0) + int(pf_clock[-1])

    return metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_diagnostics(residuals, field_stats, profiles, era5, metrics, output_dir, label):
    """Generate all diagnostic plots."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Convergence (4 panels) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    for var in ["Ux", "Uy", "Uz", "k", "epsilon", "p"]:
        if var in residuals:
            ax.semilogy(residuals[var], label=var, linewidth=1)
    ax.set_ylabel("Initial residual")
    ax.set_title("Residuals")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if "speed_avg" in field_stats:
        ax.plot(field_stats["speed_avg"], "b-", label="|U| avg")
    if "speed_max" in field_stats:
        ax.plot(field_stats["speed_max"], "r-", alpha=0.7, label="|U| max")
    ax.set_ylabel("Wind speed [m/s]")
    ax.set_title("Speed convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for key, lbl, c in [("k_avg", "k avg", "g"), ("k_max", "k max", "r"), ("k_min", "k min", "b")]:
        if key in field_stats:
            ax.semilogy(field_stats[key], c=c, label=lbl, linewidth=1)
    ax.set_ylabel("TKE [m²/s²]")
    ax.set_xlabel("Iteration")
    ax.set_title("TKE convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if "T_avg" in field_stats:
        ax.plot(field_stats["T_avg"], "m-")
        ax.set_ylabel("Temperature [K]")
        ax.set_title("Volume-average T")
    elif "epsilon_avg" in field_stats:
        ax.semilogy(field_stats["epsilon_avg"], "m-")
        ax.set_ylabel("Dissipation [m²/s³]")
        ax.set_title("Volume-average ε")
    ax.set_xlabel("Iteration")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Convergence — {label}", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "convergence.png", dpi=150)
    plt.close(fig)

    # --- 2. Profiles vs ERA5 ---
    if not profiles:
        return

    n = len(profiles)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n), squeeze=False)
    for i, (name, prof) in enumerate(profiles.items()):
        z = prof["z_agl"]

        ax = axes[i, 0]
        ax.plot(prof["speed"], z, "b.-", markersize=2, label=f"CFD")
        ax.plot(era5["speed"], era5["z_agl"], "r--", linewidth=2, label="ERA5")
        ax.set_xlabel("Speed [m/s]"); ax.set_ylabel("z AGL [m]")
        ax.set_title(f"{name} (z_terrain={prof['z_terrain']:.0f}m) — Speed")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(0, 3000)

        ax = axes[i, 1]
        ax.semilogx(np.maximum(prof["k"], 1e-6), z, "g.-", markersize=2)
        ax.set_xlabel("TKE [m²/s²]"); ax.set_title(f"{name} — TKE")
        ax.grid(True, alpha=0.3); ax.set_ylim(0, 1000)

        ax = axes[i, 2]
        if "T" in prof:
            ax.plot(prof["T"], z, "m.-", markersize=2, label="CFD T")
            if "T" in era5:
                ax.plot(era5["T"], era5["z_agl"], "r--", linewidth=2, label="ERA5 T")
            ax.set_xlabel("T [K]"); ax.set_title(f"{name} — Temperature"); ax.legend(fontsize=8)
        else:
            ax.plot(prof["w"], z, "c.-", markersize=2)
            ax.set_xlabel("w [m/s]"); ax.set_title(f"{name} — Vertical velocity")
            ax.axvline(0, color="k", linewidth=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3); ax.set_ylim(0, 1000)

    fig.suptitle(f"Vertical profiles — {label}", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "profiles.png", dpi=150)
    plt.close(fig)

    # --- 3. Boundary layer zoom ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for name, prof in profiles.items():
        mask = prof["z_agl"] < 500
        axes[0].plot(prof["speed"][mask], prof["z_agl"][mask], ".-", markersize=2, label=name)
        axes[1].semilogx(np.maximum(prof["k"][mask], 1e-6), prof["z_agl"][mask], ".-", markersize=2, label=name)
        if "T" in prof:
            axes[2].plot(prof["T"][mask], prof["z_agl"][mask], ".-", markersize=2, label=name)

    axes[0].plot(era5["speed"], era5["z_agl"], "k--", linewidth=2, label="ERA5", alpha=0.5)
    axes[0].set_xlabel("Speed [m/s]"); axes[0].set_ylabel("z AGL [m]")
    axes[0].set_title("BL — Speed"); axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3); axes[0].set_ylim(0, 500)
    axes[1].set_xlabel("TKE [m²/s²]"); axes[1].set_title("BL — TKE")
    axes[1].grid(True, alpha=0.3); axes[1].set_ylim(0, 500)
    if "T" in era5:
        axes[2].plot(era5["T"], era5["z_agl"], "k--", linewidth=2, label="ERA5", alpha=0.5)
    axes[2].set_xlabel("T [K]"); axes[2].set_title("BL — Temperature")
    axes[2].grid(True, alpha=0.3); axes[2].set_ylim(0, 500)

    fig.suptitle(f"Boundary layer (0–500m) — {label}", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "boundary_layer.png", dpi=150)
    plt.close(fig)
    logger.info("Plots saved to %s", output_dir)


def plot_comparison(metrics_files: list[Path], output_path: Path):
    """Compare multiple cases from their metrics.json files."""
    import matplotlib.pyplot as plt

    all_m = [json.loads(f.read_text()) for f in metrics_files]
    labels = [m["label"] for m in all_m]
    n = len(labels)
    x = np.arange(n)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # RMSE
    ax = axes[0, 0]
    for tag, color, off in [("bl", "tab:blue", -0.25), ("mid", "tab:orange", 0), ("upper", "tab:green", 0.25)]:
        vals = [m.get(f"rmse_{tag}", np.nan) for m in all_m]
        ax.bar(x + off, vals, 0.25, label=f"RMSE {tag}", color=color)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("RMSE vs ERA5 [m/s]"); ax.set_title("RMSE by height"); ax.legend(fontsize=8)

    # Bias
    ax = axes[0, 1]
    for tag, color, off in [("bl", "tab:blue", -0.25), ("mid", "tab:orange", 0), ("upper", "tab:green", 0.25)]:
        vals = [m.get(f"bias_{tag}", np.nan) for m in all_m]
        ax.bar(x + off, vals, 0.25, label=f"Bias {tag}", color=color)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Bias [m/s]"); ax.set_title("Bias by height")
    ax.axhline(0, color="k", linewidth=0.5); ax.legend(fontsize=8)

    # Speedup
    ax = axes[0, 2]
    ax.bar(x, [m.get("speedup_100m", np.nan) for m in all_m], color="tab:purple")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Speedup"); ax.set_title("Speedup centre 100m AGL")
    ax.axhline(1, color="k", linewidth=0.5, ls="--")

    # Wall time
    ax = axes[1, 0]
    ax.bar(x, [m.get("wall_time_s", 0) for m in all_m], color="tab:gray")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Wall time [s]"); ax.set_title("Computation time")

    # Residual
    ax = axes[1, 1]
    ax.bar(x, [m.get("final_residual_Ux", np.nan) for m in all_m], color="tab:red")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Ux residual"); ax.set_title("Final residual"); ax.set_yscale("log")

    # Speed avg
    ax = axes[1, 2]
    ax.bar(x, [m.get("final_speed_avg", np.nan) for m in all_m], color="tab:cyan")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("|U| avg [m/s]"); ax.set_title("Final vol-avg speed")

    fig.suptitle("Cross-case comparison", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Comparison: %s", output_path)


# ---------------------------------------------------------------------------
# Public API (for run_sf_poc.py / run_local_study.py)
# ---------------------------------------------------------------------------

def _read_case_params(case_dir: Path) -> dict:
    """Extract simulation parameters from case files for DB registration."""
    params = {}
    # controlDict
    cd = case_dir / "system" / "controlDict"
    if cd.exists():
        text = cd.read_text()
        m = re.search(r"^application\s+(\w+)", text, re.MULTILINE)
        if m:
            params["solver"] = m.group(1)
        m = re.search(r"^endTime\s+(\d+)", text, re.MULTILINE)
        if m:
            params["n_iter"] = int(m.group(1))
        params["transport_T"] = "scalarTransportT" in text

    # fvSolution
    fvs = case_dir / "system" / "fvSolution"
    if fvs.exists():
        text = fvs.read_text()
        m = re.search(r"nNonOrthogonalCorrectors\s+(\d+)", text)
        if m:
            params["n_non_ortho_correctors"] = int(m.group(1))

    # U top BC
    u_file = case_dir / "0" / "U"
    if u_file.exists():
        text = u_file.read_text()
        if "inletOutlet" in text and "top" in text:
            params["top_bc_U"] = "inletOutlet"
        elif "pressureInletOutletVelocity" in text:
            params["top_bc_U"] = "pressureInletOutletVelocity"

    # fvOptions — coriolis, canopy
    fvo = case_dir / "constant" / "fvOptions"
    if fvo.exists():
        text = fvo.read_text()
        params["coriolis"] = "atmCoriolisUSource" in text
        params["canopy"] = "atmPlantCanopyUSource" in text

    # meshDict — cell sizes
    md = case_dir / "system" / "meshDict"
    if md.exists():
        text = md.read_text()
        m = re.search(r"maxCellSize\s+([\d.]+)", text)
        if m:
            params["maxCellSize"] = float(m.group(1))
        # Find finest cellSize in objectRefinements
        cell_sizes = [float(x) for x in re.findall(r"cellSize\s+([\d.]+)", text)]
        if cell_sizes:
            params["fine_cell_size"] = min(cell_sizes)

    # Inflow
    inflow_json = case_dir / "inflow.json"
    if inflow_json.exists():
        inflow = json.loads(inflow_json.read_text())
        params["u_hub"] = inflow.get("u_hub")
        params["wind_dir"] = inflow.get("wind_dir")
        params["T_ref"] = inflow.get("T_ref")
        params["timestamp"] = inflow.get("timestamp", "")

    return params


def _read_mesh_stats(case_dir: Path) -> dict:
    """Extract mesh stats from checkMesh or cartesianMesh log."""
    stats = {}
    for log_name in ["log.cartesianMesh", "log.checkMesh"]:
        log = case_dir / log_name
        if not log.exists():
            continue
        text = log.read_text()
        m = re.search(r"cells:\s+(\d+)", text)
        if m:
            stats["n_cells"] = int(m.group(1))
        m = re.search(r"Max:\s+([\d.]+)\s+average", text)
        if m:
            stats["max_non_ortho"] = float(m.group(1))
        m = re.search(r"Max skewness = ([\d.]+)", text)
        if m:
            stats["max_skewness"] = float(m.group(1))
        m = re.search(r"ClockTime = (\d+)", text)
        if m:
            stats["mesh_time_s"] = int(m.group(1))
    return stats


def evaluate_single(
    case_dir: Path,
    output_dir: Path | None = None,
    label: str | None = None,
    probes: dict | None = None,
    grid_probes: bool = True,
    register_db: bool = True,
) -> dict:
    """Evaluate a single case. Returns metrics dict.

    Extracts:
      - 5 named probes (centre, era5, ridges, valley)
      - 10×10 grid probes in fine zone (for statistical convergence)
      - Registers everything in the simulation DB

    Can be called programmatically:
        from evaluate_case import evaluate_single
        metrics = evaluate_single(Path("data/cases/study/case_A"))
    """
    case_dir = Path(case_dir)
    if output_dir is None:
        output_dir = default_output_dir(case_dir)
    if label is None:
        label = default_label(case_dir)
    if probes is None:
        probes = DEFAULT_PROBES

    output_dir.mkdir(parents=True, exist_ok=True)

    log_sf = case_dir / "log.simpleFoam"
    if not log_sf.exists():
        logger.warning("No log.simpleFoam in %s — skipping", case_dir)
        return {}

    residuals = parse_residuals(log_sf)
    field_stats = parse_field_stats(log_sf)
    era5 = load_era5_profile(case_dir / "inflow.json")

    # Named probes (5 locations)
    profiles = extract_profiles(case_dir, probes)
    metrics = compute_metrics(profiles, era5, label, case_dir)
    metrics["output_dir"] = str(output_dir)

    # Grid probes (10×10 = 100 locations for statistical convergence)
    grid_profiles = {}
    if grid_probes:
        grid = make_grid_probes()
        grid_profiles = extract_profiles(case_dir, grid, search_radius=150.0)
        metrics["n_grid_probes"] = len(grid_profiles)
        # Grid-averaged speed at 100m AGL
        speeds_100m = []
        for prof in grid_profiles.values():
            mask = (prof["z_agl"] > 80) & (prof["z_agl"] < 120)
            if mask.any():
                speeds_100m.append(float(np.mean(prof["speed"][mask])))
        if speeds_100m:
            arr = np.array(speeds_100m)
            metrics["grid_speed_100m_mean"] = float(arr.mean())
            metrics["grid_speed_100m_std"] = float(arr.std())
            metrics["grid_speed_100m_min"] = float(arr.min())
            metrics["grid_speed_100m_max"] = float(arr.max())

    # Save metrics + plots
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_diagnostics(residuals, field_stats, profiles, era5, metrics, output_dir, label)

    # Save grid probe data as CSV
    if grid_profiles:
        import csv
        csv_path = output_dir / "grid_probes.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["probe", "z_agl", "speed", "ux", "uy", "w", "k", "epsilon", "T"])
            for name, prof in grid_profiles.items():
                for i in range(len(prof["z_agl"])):
                    writer.writerow([
                        name, f"{prof['z_agl'][i]:.1f}",
                        f"{prof['speed'][i]:.4f}", f"{prof['ux'][i]:.4f}",
                        f"{prof['uy'][i]:.4f}", f"{prof['w'][i]:.4f}",
                        f"{prof['k'][i]:.6f}", f"{prof['epsilon'][i]:.6e}",
                        f"{prof['T'][i]:.2f}" if "T" in prof else "",
                    ])
        logger.info("Grid probes CSV: %s (%d probes)", csv_path, len(grid_profiles))

    # Register in simulation database
    if register_db:
        try:
            from simulation_registry import SimulationRegistry
            db = SimulationRegistry()
            params = _read_case_params(case_dir)
            run_id = db.register_run(label, case_dir, params)

            mesh_stats = _read_mesh_stats(case_dir)
            if mesh_stats:
                db.update_mesh(run_id, **mesh_stats)

            db.update_metrics(run_id, metrics)

            # Store named probe data in DB
            all_probes = {**profiles, **grid_profiles}
            db.store_probe_data(run_id, all_probes)

            db.close()
            logger.info("Registered in DB as run #%d", run_id)
        except Exception as exc:
            logger.warning("DB registration failed: %s", exc)

    logger.info("[%s] RMSE_bl=%.2f, bias_bl=%.2f, speedup=%.2f, wall=%ds, grid_probes=%d",
                label,
                metrics.get("rmse_bl", -1), metrics.get("bias_bl", -1),
                metrics.get("speedup_100m", -1), metrics.get("wall_time_s", -1),
                len(grid_profiles))
    return metrics


def evaluate_batch(cases_dir: Path, output_root: Path | None = None) -> list[dict]:
    """Evaluate all cases in a study directory. Returns list of metrics."""
    case_dirs = sorted([
        d for d in cases_dir.iterdir()
        if d.is_dir() and d.name.startswith("case_")
        and (d / "log.simpleFoam").exists()
    ])
    logger.info("Batch eval: %d cases in %s", len(case_dirs), cases_dir)

    all_metrics = []
    for case_dir in case_dirs:
        out = (output_root / case_dir.name.removeprefix("case_")) if output_root else None
        m = evaluate_single(case_dir, output_dir=out)
        if m:
            all_metrics.append(m)

    # Auto-generate comparison
    if len(all_metrics) >= 2:
        study_name = cases_dir.name
        val_dir = ROOT / "data" / "validation" / study_name
        metrics_files = sorted(val_dir.glob("*/metrics.json"))
        if len(metrics_files) >= 2:
            plot_comparison(metrics_files, val_dir / "comparison.png")

    return all_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Evaluate OpenFOAM cases — single, batch, or compare",
    )
    parser.add_argument("--case", help="Single case directory")
    parser.add_argument("--batch", help="Study directory (evaluates all case_* subdirs)")
    parser.add_argument("--compare", nargs="+", help="metrics.json files to compare")
    parser.add_argument("--output", help="Output directory (auto if omitted)")
    parser.add_argument("--label", help="Case label (auto if omitted)")
    args = parser.parse_args()

    if args.case:
        out = Path(args.output) if args.output else None
        evaluate_single(Path(args.case), output_dir=out, label=args.label)
    elif args.batch:
        out = Path(args.output) if args.output else None
        evaluate_batch(Path(args.batch), output_root=out)
    elif args.compare:
        if not args.output:
            args.output = str(Path(args.compare[0]).parent.parent / "comparison.png")
        plot_comparison([Path(f) for f in args.compare], Path(args.output))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
