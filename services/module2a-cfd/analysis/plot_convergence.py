"""
plot_convergence.py — Publication-quality mesh convergence plot

Reads multiple at_masts.csv (one per resolution) and perdigao_obs.zarr,
then produces:
  1. RMSE vs resolution (log-log) with Richardson extrapolation fit
  2. Per-tower profile comparison at 3 resolutions
  3. Scatter plots (one panel per resolution)

Usage
-----
    python plot_convergence.py \
        --obs-zarr   data/raw/perdigao_obs.zarr \
        --timestamp  2017-05-11T12:00 \
        --csv500     data/cfd-database/perdigao/20170511T12_500m/at_masts.csv \
        --csv250     data/cfd-database/perdigao/20170511T12_250m/at_masts.csv \
        --csv100     data/cfd-database/perdigao/20170511T12_100m/at_masts.csv \
        --output     data/validation/convergence_20170511T12/
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

logger = logging.getLogger(__name__)

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.5,
})

COLORS = {500: "#1b9e77", 250: "#d95f02", 100: "#7570b3"}
KEY_TOWERS = ["tse04", "tse09", "tse13"]


def load_cfd(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "tower_id": row["tower_id"],
                "height_m": float(row["height_m"]),
                "u": float(row["u_ms"]),
                "v": float(row["v_ms"]),
                "speed": float(row["speed_ms"]),
            })
    return rows


def load_obs(zarr_path: Path, timestamp: str) -> dict:
    import zarr
    store = zarr.open(str(zarr_path), mode="r")
    times = np.array(store["coords/time"][:]).astype("datetime64[ns]")
    tidx = int(np.argmin(np.abs(times - np.datetime64(timestamp))))
    site_ids_raw = np.array(store["coords/site_id"][:])
    site_ids = [s.decode() if isinstance(s, bytes) else str(s) for s in site_ids_raw]
    heights = np.array(store["coords/height_m"][:])
    u_all = np.array(store["sites/u"][tidx, :, :])
    v_all = np.array(store["sites/v"][tidx, :, :])
    obs = {}
    for si, sid in enumerate(site_ids):
        u, v = u_all[si, :], v_all[si, :]
        valid = ~np.isnan(u) & ~np.isnan(v)
        if valid.any():
            obs[sid] = {
                "heights": heights[valid].tolist(),
                "u": u[valid].tolist(),
                "v": v[valid].tolist(),
                "speed": np.sqrt(u[valid]**2 + v[valid]**2).tolist(),
            }
    return obs


def match(cfd_rows: list[dict], obs: dict) -> dict:
    m = {"tower_id": [], "height": [], "cfd_speed": [], "obs_speed": [],
         "cfd_u": [], "obs_u": [], "cfd_v": [], "obs_v": []}
    for row in cfd_rows:
        tid, h_cfd = row["tower_id"], row["height_m"]
        if tid not in obs:
            continue
        for i, h_obs in enumerate(obs[tid]["heights"]):
            if abs(h_cfd - h_obs) <= 5.0:
                m["tower_id"].append(tid)
                m["height"].append(h_cfd)
                m["cfd_speed"].append(row["speed"])
                m["obs_speed"].append(obs[tid]["speed"][i])
                m["cfd_u"].append(row["u"])
                m["obs_u"].append(obs[tid]["u"][i])
                m["cfd_v"].append(row["v"])
                m["obs_v"].append(obs[tid]["v"][i])
                break
    for k in m:
        if k != "tower_id":
            m[k] = np.array(m[k])
    return m


def compute_metrics(matched: dict) -> dict:
    """Compute RMSE, bias, R for speed and components."""
    cfd_s, obs_s = matched["cfd_speed"], matched["obs_speed"]
    n = len(cfd_s)
    if n == 0:
        return {"n": 0, "rmse": np.nan, "bias": np.nan, "r": np.nan, "mae": np.nan}
    bias = float(np.mean(cfd_s - obs_s))
    rmse = float(np.sqrt(np.mean((cfd_s - obs_s)**2)))
    mae = float(np.mean(np.abs(cfd_s - obs_s)))
    r = float(np.corrcoef(cfd_s, obs_s)[0, 1]) if n > 1 else np.nan

    # Per-tower metrics for key towers
    tower_metrics = {}
    for tid in KEY_TOWERS:
        mask = np.array([t == tid for t in matched["tower_id"]])
        if mask.sum() > 0:
            tower_metrics[tid] = {
                "rmse": float(np.sqrt(np.mean((cfd_s[mask] - obs_s[mask])**2))),
                "bias": float(np.mean(cfd_s[mask] - obs_s[mask])),
                "n": int(mask.sum()),
            }
    return {"n": n, "rmse": rmse, "bias": bias, "r": r, "mae": mae,
            "tower_metrics": tower_metrics}


# ---------------------------------------------------------------------------
# Figure 1: RMSE vs resolution (convergence)
# ---------------------------------------------------------------------------

def plot_convergence(resolutions: list[int], metrics_list: list[dict],
                     output_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    res = np.array(resolutions, dtype=float)
    rmse = np.array([m["rmse"] for m in metrics_list])
    bias = np.array([m["bias"] for m in metrics_list])
    mae = np.array([m["mae"] for m in metrics_list])

    valid = ~np.isnan(rmse)
    if valid.sum() < 2:
        logger.warning("Not enough valid resolutions for convergence plot")
        return

    # Left panel: RMSE convergence
    ax1.plot(res[valid], rmse[valid], "ko-", ms=8, zorder=5, label="RMSE")
    ax1.plot(res[valid], mae[valid], "s--", color="gray", ms=6, zorder=4, label="MAE")

    # Richardson extrapolation fit (if 3 points)
    if valid.sum() >= 3:
        from scipy.optimize import curve_fit
        def power_law(x, a, p):
            return a * x**p
        try:
            popt, _ = curve_fit(power_law, res[valid], rmse[valid], p0=[0.01, 1.0])
            x_fit = np.linspace(50, 600, 100)
            ax1.plot(x_fit, power_law(x_fit, *popt), "k:", lw=1.0,
                     label=f"fit: {popt[0]:.3f}·Δx^{popt[1]:.2f}")
        except Exception:
            pass

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax1.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax1.set_xticks(resolutions)
    ax1.set_xlabel("Horizontal resolution Δx [m]")
    ax1.set_ylabel("Error [m/s]")
    ax1.set_title("(a) Mesh convergence — wind speed")
    ax1.legend(loc="upper left")
    ax1.set_xlim(80, 600)

    # Right panel: per-tower RMSE
    tower_ids = KEY_TOWERS
    x_pos = np.arange(len(tower_ids))
    width = 0.25
    for i, (r, m) in enumerate(zip(resolutions, metrics_list)):
        if "tower_metrics" not in m:
            continue
        rmses = [m["tower_metrics"].get(t, {}).get("rmse", np.nan) for t in tower_ids]
        ax2.bar(x_pos + i * width, rmses, width, color=COLORS.get(r, f"C{i}"),
                label=f"{r} m", edgecolor="k", linewidth=0.5)

    ax2.set_xticks(x_pos + width)
    ax2.set_xticklabels(tower_ids)
    ax2.set_ylabel("RMSE [m/s]")
    ax2.set_title("(b) Per-tower RMSE")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "convergence.png")
    fig.savefig(output_dir / "convergence.pdf")
    plt.close(fig)
    logger.info("Convergence plot saved: %s", output_dir / "convergence.pdf")


# ---------------------------------------------------------------------------
# Figure 2: Vertical profiles at key towers (multi-resolution)
# ---------------------------------------------------------------------------

def plot_profiles_multi(resolutions: list[int], cfd_data: list[list[dict]],
                        obs: dict, output_dir: Path) -> None:
    towers = [t for t in KEY_TOWERS if t in obs]
    if not towers:
        return

    fig, axes = plt.subplots(1, len(towers), figsize=(4.2 * len(towers), 5.5),
                             sharey=True)
    if len(towers) == 1:
        axes = [axes]

    for ti, tid in enumerate(towers):
        ax = axes[ti]
        # Obs
        obs_h = np.array(obs[tid]["heights"])
        obs_spd = np.array(obs[tid]["speed"])
        ax.plot(obs_spd, obs_h, "ko-", ms=5, lw=2, label="Obs", zorder=5)

        # CFD at each resolution
        for ri, (res, rows) in enumerate(zip(resolutions, cfd_data)):
            h, spd = [], []
            for row in rows:
                if row["tower_id"] == tid:
                    h.append(row["height_m"])
                    spd.append(row["speed"])
            if h:
                order = np.argsort(h)
                ax.plot(np.array(spd)[order], np.array(h)[order],
                        "s--", ms=4, color=COLORS.get(res, f"C{ri}"),
                        label=f"{res} m", zorder=3)

        ax.set_xlabel("Wind speed [m/s]")
        if ti == 0:
            ax.set_ylabel("Height AGL [m]")
        ax.set_title(tid, fontweight="bold")
        ax.legend(fontsize=8)

    fig.suptitle("Vertical wind profiles — mesh convergence study\n"
                 "Perdigão, 2017-05-11 12:00 UTC, SSW 217°, $u_{hub}$ = 7.9 m/s",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_dir / "profiles_convergence.png")
    fig.savefig(output_dir / "profiles_convergence.pdf")
    plt.close(fig)
    logger.info("Profile convergence plot saved: %s", output_dir / "profiles_convergence.pdf")


# ---------------------------------------------------------------------------
# Figure 3: Scatter subplots
# ---------------------------------------------------------------------------

def plot_scatter_multi(resolutions: list[int], matched_list: list[dict],
                       output_dir: Path) -> None:
    n_res = len(resolutions)
    fig, axes = plt.subplots(1, n_res, figsize=(4.2 * n_res, 4.2), sharey=True, sharex=True)
    if n_res == 1:
        axes = [axes]

    vmax = 0
    for m in matched_list:
        if len(m["cfd_speed"]) > 0:
            vmax = max(vmax, m["cfd_speed"].max(), m["obs_speed"].max())
    vmax *= 1.15

    for i, (res, m) in enumerate(zip(resolutions, matched_list)):
        ax = axes[i]
        if len(m["cfd_speed"]) == 0:
            ax.set_title(f"Δx = {res} m\n(no data)")
            continue

        ax.scatter(m["obs_speed"], m["cfd_speed"], s=20, alpha=0.7,
                   color=COLORS.get(res, f"C{i}"), edgecolors="k", linewidths=0.3)
        ax.plot([0, vmax], [0, vmax], "k--", lw=0.8)
        ax.set_xlim(0, vmax)
        ax.set_ylim(0, vmax)
        ax.set_aspect("equal")

        bias = float(np.mean(m["cfd_speed"] - m["obs_speed"]))
        rmse = float(np.sqrt(np.mean((m["cfd_speed"] - m["obs_speed"])**2)))
        ax.text(0.05, 0.92,
                f"bias = {bias:+.2f} m/s\nRMSE = {rmse:.2f} m/s\nN = {len(m['cfd_speed'])}",
                transform=ax.transAxes, fontsize=8, verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))
        ax.set_title(f"Δx = {res} m", fontweight="bold")
        ax.set_xlabel("Observed [m/s]")
        if i == 0:
            ax.set_ylabel("CFD [m/s]")

    fig.suptitle("CFD vs Observed wind speed — all towers", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_dir / "scatter_convergence.png")
    fig.savefig(output_dir / "scatter_convergence.pdf")
    plt.close(fig)
    logger.info("Scatter convergence plot saved: %s", output_dir / "scatter_convergence.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Mesh convergence publication plots")
    parser.add_argument("--obs-zarr",  required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--csv500",    default=None, help="at_masts.csv for 500m")
    parser.add_argument("--csv250",    default=None, help="at_masts.csv for 250m")
    parser.add_argument("--csv100",    default=None, help="at_masts.csv for 100m")
    parser.add_argument("--output",    required=True, help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    obs = load_obs(Path(args.obs_zarr), args.timestamp)
    logger.info("Obs: %d towers with data", len(obs))

    resolutions = []
    cfd_data = []
    matched_list = []
    metrics_list = []

    for res, csv_path in [(500, args.csv500), (250, args.csv250), (100, args.csv100)]:
        if csv_path is None or not Path(csv_path).exists():
            continue
        rows = load_cfd(Path(csv_path))
        m = match(rows, obs)
        met = compute_metrics(m)
        resolutions.append(res)
        cfd_data.append(rows)
        matched_list.append(m)
        metrics_list.append(met)
        logger.info("  %dm: %d matched, RMSE=%.2f, bias=%+.2f, R=%.3f",
                     res, met["n"], met["rmse"], met["bias"], met["r"])

    # Summary table
    print(f"\n{'Δx [m]':>8s}  {'N':>4s}  {'RMSE':>7s}  {'MAE':>7s}  {'bias':>7s}  {'R':>6s}")
    print("-" * 48)
    for res, met in zip(resolutions, metrics_list):
        print(f"{res:>8d}  {met['n']:4d}  {met['rmse']:7.2f}  {met['mae']:7.2f}  "
              f"{met['bias']:+7.2f}  {met['r']:6.3f}")

    # Plots
    plot_convergence(resolutions, metrics_list, output_dir)
    plot_profiles_multi(resolutions, cfd_data, obs, output_dir)
    plot_scatter_multi(resolutions, matched_list, output_dir)

    print(f"\nPlots saved to: {output_dir}/")


if __name__ == "__main__":
    main()
