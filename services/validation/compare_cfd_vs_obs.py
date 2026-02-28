"""
compare_cfd_vs_obs.py — CFD vs observation comparison and convergence metrics

Computes RMSE, bias, R² of CFD wind profiles against Perdigao mast observations,
and generates the convergence figure (RMSE + CPU time vs resolution).

Usage
-----
    python compare_cfd_vs_obs.py \
        --at-masts-dir data/cfd-database/perdigao \
        --obs           data/raw/perdigao_obs.zarr \
        --case-id       2017-05-15T12:00 \
        --output        figures/
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from services.validation.plot_style import apply_style, save, COLORS

logger = logging.getLogger(__name__)

HEIGHT_OBS_M = [10, 20, 40, 60, 80, 100]


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

def load_at_masts(at_masts_csv: Path) -> dict:
    """Load at_masts.csv produced by export_cfd.py."""
    rows = []
    with open(at_masts_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) if k not in ("tower_id",) else v
                         for k, v in row.items()})
    return rows


def compute_metrics(cfd_rows: list[dict], obs_rows: list[dict]) -> dict:
    """Compute RMSE, bias, R² between CFD and obs at matching tower/height pairs.

    Parameters
    ----------
    cfd_rows, obs_rows:
        Lists of dicts with keys: tower_id, height_m, speed_ms, dir_deg, u_ms, v_ms.

    Returns
    -------
    dict: overall + per-tower metrics.
    """
    # Build lookup: (tower_id, height_m) → speed_ms
    def make_lookup(rows):
        return {(r["tower_id"], float(r["height_m"])): float(r["speed_ms"])
                for r in rows}

    cfd_lut = make_lookup(cfd_rows)
    obs_lut = make_lookup(obs_rows)

    common_keys = set(cfd_lut.keys()) & set(obs_lut.keys())
    if not common_keys:
        logger.warning("No common (tower, height) pairs between CFD and obs")
        return {}

    cfd_vals = np.array([cfd_lut[k] for k in common_keys])
    obs_vals = np.array([obs_lut[k] for k in common_keys])

    rmse  = float(np.sqrt(np.mean((cfd_vals - obs_vals) ** 2)))
    bias  = float(np.mean(cfd_vals - obs_vals))
    ss_res = np.sum((obs_vals - cfd_vals) ** 2)
    ss_tot = np.sum((obs_vals - obs_vals.mean()) ** 2)
    r2    = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    metrics = {
        "overall": {"rmse": rmse, "bias": bias, "r2": r2, "n": len(common_keys)},
        "per_tower": {},
    }

    towers = set(k[0] for k in common_keys)
    for tower in sorted(towers):
        t_keys = [k for k in common_keys if k[0] == tower]
        tc = np.array([cfd_lut[k] for k in t_keys])
        to = np.array([obs_lut[k] for k in t_keys])
        t_rmse = float(np.sqrt(np.mean((tc - to) ** 2)))
        t_bias = float(np.mean(tc - to))
        metrics["per_tower"][tower] = {"rmse": t_rmse, "bias": t_bias, "n": len(t_keys)}

    return metrics


# ---------------------------------------------------------------------------
# Convergence figure
# ---------------------------------------------------------------------------

def plot_convergence(
    resolution_results: list[dict],
    output_dir: Path,
) -> None:
    """Plot RMSE vs resolution and CPU time vs resolution.

    Parameters
    ----------
    resolution_results:
        List of dicts with keys: resolution_m, context_cells, rmse, cpu_time_s.
    output_dir:
        Where to save figures.
    """
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Group by context_cells
    contexts = sorted(set(r.get("context_cells", 3) for r in resolution_results))
    ctx_colors = {1: "#d62728", 3: "#1f77b4", 5: "#2ca02c"}
    ctx_labels = {1: "1×1 (pipeline test)", 3: "3×3 (75 km)", 5: "5×5 (125 km)"}

    for ctx in contexts:
        rows = sorted([r for r in resolution_results if r.get("context_cells") == ctx],
                      key=lambda r: r["resolution_m"])
        if not rows:
            continue
        res   = [r["resolution_m"] for r in rows]
        rmse  = [r.get("rmse", float("nan")) for r in rows]
        cpu   = [r.get("cpu_time_s", float("nan")) for r in rows]
        color = ctx_colors.get(ctx, "gray")
        label = ctx_labels.get(ctx, f"{ctx}×{ctx}")

        ax1.plot(res, rmse, "o-", color=color, label=label)
        ax2.plot(res, [c / 60 for c in cpu], "o-", color=color, label=label)

    ax1.set_xscale("log")
    ax1.invert_xaxis()
    ax1.set_xlabel("Horizontal resolution [m]")
    ax1.set_ylabel("RMSE |u| [m/s]")
    ax1.set_title("Wind speed RMSE vs mast observations")
    ax1.legend(fontsize=8)

    ax2.set_xscale("log")
    ax2.invert_xaxis()
    ax2.set_xlabel("Horizontal resolution [m]")
    ax2.set_ylabel("CPU time [min]")
    ax2.set_title("Computational cost")
    ax2.legend(fontsize=8)

    output_dir.mkdir(parents=True, exist_ok=True)
    save(fig, str(output_dir / "convergence_rmse_cpu.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Vertical profiles figure
# ---------------------------------------------------------------------------

def plot_vertical_profiles(
    cfd_rows: list[dict],
    obs_rows: list[dict],
    era5_rows: list[dict] | None,
    output_path: Path,
    case_id: str = "",
) -> None:
    """Plot vertical wind profiles at T20, T25, T13 + speed-up ratio.

    Parameters
    ----------
    cfd_rows, obs_rows, era5_rows:
        Lists of dicts with keys: tower_id, height_m, speed_ms.
    output_path:
        Output figure path.
    """
    apply_style()
    towers = ["T20", "T25", "T13"]
    fig, axes = plt.subplots(1, len(towers) + 1, figsize=(14, 5))

    def get_profile(rows, tower):
        sub = sorted([r for r in rows if r["tower_id"] == tower],
                     key=lambda r: r["height_m"])
        if not sub:
            return np.array([]), np.array([])
        return (np.array([r["height_m"] for r in sub]),
                np.array([r["speed_ms"] for r in sub]))

    for ax, tower in zip(axes[:len(towers)], towers):
        z_cfd, s_cfd = get_profile(cfd_rows, tower)
        z_obs, s_obs = get_profile(obs_rows, tower)

        if len(s_obs) > 0:
            ax.plot(s_obs, z_obs, "k-o", lw=2, ms=5, label="Observations", zorder=4)
        if len(s_cfd) > 0:
            ax.plot(s_cfd, z_cfd, "-s", color=COLORS["cfd_1km"], lw=1.5,
                    ms=4, label="CFD", zorder=3)
        if era5_rows is not None:
            z_era5, s_era5 = get_profile(era5_rows, tower)
            if len(s_era5) > 0:
                ax.plot(s_era5, z_era5, "--", color=COLORS["era5"], lw=1.5,
                        label="ERA5", zorder=2)

        ax.set_xlabel("|u| [m/s]")
        ax.set_ylabel("Height [m]")
        ax.set_ylim(0, 200)
        ax.set_title(f"Tower {tower}")
        ax.legend(fontsize=7)

    # Speed-up ratio panel
    ax_su = axes[-1]
    ref_tower = "T25"   # valley reference
    ridge_towers = ["T20"]
    heights_su = [40, 80]

    for h in heights_su:
        def get_speed(rows, tower, height):
            sub = [r for r in rows if r["tower_id"] == tower
                   and abs(r["height_m"] - height) < 1]
            return float(sub[0]["speed_ms"]) if sub else float("nan")

        for ridge in ridge_towers:
            su_obs = get_speed(obs_rows, ridge, h) / (get_speed(obs_rows, ref_tower, h) + 1e-6)
            su_cfd = get_speed(cfd_rows, ridge, h) / (get_speed(cfd_rows, ref_tower, h) + 1e-6)
            ax_su.bar([f"{ridge}/{ref_tower}\nz={h}m"], [su_obs],
                      color="black", alpha=0.6, width=0.3, label="Obs" if h == heights_su[0] else "")
            ax_su.bar([f"{ridge}/{ref_tower}\nz={h}m"], [su_cfd],
                      color=COLORS["cfd_1km"], alpha=0.8, width=0.2, label="CFD" if h == heights_su[0] else "")

    ax_su.set_ylabel("Speed-up ratio U(ridge)/U(valley)")
    ax_su.set_title("Speed-up ratio")
    ax_su.legend(fontsize=7)
    ax_su.axhline(1.0, color="gray", lw=0.8, ls="--")

    fig.suptitle(f"CFD vs observations — {case_id}", fontsize=11)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save(fig, str(output_path))
    plt.close(fig)
    logger.info("Saved: %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="CFD vs observations comparison")
    parser.add_argument("--at-masts-dir", required=True,
                        help="Root of cfd-database (contains case subdirs)")
    parser.add_argument("--obs",      default=None,
                        help="Perdigao obs zarr (optional — shows ERA5-vs-CFD only)")
    parser.add_argument("--case-id",  default="",
                        help="Case ID filter (partial match)")
    parser.add_argument("--output",   default="figures/")
    args = parser.parse_args()

    output_dir = Path(args.output)

    # Find all at_masts.csv files
    at_masts_root = Path(args.at_masts_dir)
    csv_files = list(at_masts_root.rglob("at_masts.csv"))
    if not csv_files:
        print(f"No at_masts.csv found under {at_masts_root}")
        raise SystemExit(1)

    logger.info("Found %d at_masts.csv files", len(csv_files))

    # For each case, compute and print metrics
    for csv_file in csv_files:
        case_name = csv_file.parent.name
        if args.case_id and args.case_id not in case_name:
            continue

        cfd_rows = load_at_masts(csv_file)
        # Obs loading omitted here (zarr parsing is dataset-specific)
        # In practice, load from perdigao_obs.zarr and align by timestamp
        logger.info("Case %s: %d interpolated points", case_name, len(cfd_rows))
