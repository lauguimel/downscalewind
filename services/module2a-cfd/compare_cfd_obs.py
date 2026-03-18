"""
compare_cfd_obs.py — Compare CFD results at mast positions vs Perdigão observations

Reads:
  1. at_masts.csv from export_cfd.py (CFD interpolated at tower locations)
  2. perdigao_obs.zarr (NCAR/EOL ISFS QC observations)

Produces:
  - Console table: per-tower RMSE, bias, correlation of wind speed
  - Scatter plot: CFD vs observed wind speed at all heights
  - Vertical profile comparison at selected towers

Usage
-----
    python compare_cfd_obs.py \
        --cfd-csv  data/cfd-database/perdigao/case_id/at_masts.csv \
        --obs-zarr data/raw/perdigao_obs.zarr \
        --timestamp 2017-05-11T12:00 \
        --output   data/validation/500m_20170511T12/
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def load_cfd_masts(csv_path: Path) -> list[dict]:
    """Load at_masts.csv into a list of dicts."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "tower_id": row["tower_id"],
                "height_m": float(row["height_m"]),
                "u":        float(row["u_ms"]),
                "v":        float(row["v_ms"]),
                "speed":    float(row["speed_ms"]),
                "dir":      float(row["dir_deg"]),
            })
    return rows


def load_obs_snapshot(zarr_path: Path, timestamp: str) -> dict:
    """Load observation data for a single timestamp.

    Returns
    -------
    dict: site_id → {heights: [...], u: [...], v: [...], speed: [...]}
    """
    import zarr

    store = zarr.open(str(zarr_path), mode="r")
    times = np.array(store["coords/time"][:]).astype("datetime64[ns]")
    target = np.datetime64(timestamp)
    tidx = int(np.argmin(np.abs(times - target)))
    actual_time = times[tidx]
    logger.info("Obs time: %s (requested: %s)", actual_time, timestamp)

    site_ids_raw = np.array(store["coords/site_id"][:])
    # Decode bytes to str if needed
    site_ids = [s.decode() if isinstance(s, bytes) else str(s) for s in site_ids_raw]
    heights = np.array(store["coords/height_m"][:])
    u_all = np.array(store["sites/u"][tidx, :, :])   # (n_sites, n_heights)
    v_all = np.array(store["sites/v"][tidx, :, :])

    obs = {}
    for si, sid in enumerate(site_ids):
        u = u_all[si, :]
        v = v_all[si, :]
        valid = ~np.isnan(u) & ~np.isnan(v)
        if valid.any():
            obs[sid] = {
                "heights": heights[valid].tolist(),
                "u":       u[valid].tolist(),
                "v":       v[valid].tolist(),
                "speed":   np.sqrt(u[valid]**2 + v[valid]**2).tolist(),
            }
    return obs


def compare(cfd_rows: list[dict], obs: dict) -> dict:
    """Match CFD to obs at common (tower, height) pairs.

    Returns dict with arrays: tower_id, height, cfd_speed, obs_speed, cfd_u, obs_u, cfd_v, obs_v
    """
    matched = {
        "tower_id": [], "height": [],
        "cfd_speed": [], "obs_speed": [],
        "cfd_u": [], "obs_u": [],
        "cfd_v": [], "obs_v": [],
    }

    for row in cfd_rows:
        tid = row["tower_id"]
        h_cfd = row["height_m"]
        if tid not in obs:
            continue
        obs_data = obs[tid]
        # Find closest obs height within 5m
        for i, h_obs in enumerate(obs_data["heights"]):
            if abs(h_cfd - h_obs) <= 5.0:
                matched["tower_id"].append(tid)
                matched["height"].append(h_cfd)
                matched["cfd_speed"].append(row["speed"])
                matched["obs_speed"].append(obs_data["speed"][i])
                matched["cfd_u"].append(row["u"])
                matched["obs_u"].append(obs_data["u"][i])
                matched["cfd_v"].append(row["v"])
                matched["obs_v"].append(obs_data["v"][i])
                break

    for k in matched:
        if k != "tower_id":
            matched[k] = np.array(matched[k])
    return matched


def print_summary(matched: dict) -> None:
    """Print per-tower and overall statistics."""
    towers = sorted(set(matched["tower_id"]))
    cfd_s = matched["cfd_speed"]
    obs_s = matched["obs_speed"]

    print(f"\n{'tower':>8s}  {'N':>3s}  {'bias':>7s}  {'RMSE':>7s}  {'R':>6s}")
    print("-" * 42)

    for tid in towers:
        mask = np.array([t == tid for t in matched["tower_id"]])
        n = mask.sum()
        if n == 0:
            continue
        bias = float(np.mean(cfd_s[mask] - obs_s[mask]))
        rmse = float(np.sqrt(np.mean((cfd_s[mask] - obs_s[mask])**2)))
        if n > 1 and np.std(obs_s[mask]) > 0:
            r = float(np.corrcoef(cfd_s[mask], obs_s[mask])[0, 1])
        else:
            r = np.nan
        print(f"{tid:>8s}  {n:3d}  {bias:+7.2f}  {rmse:7.2f}  {r:6.3f}")

    # Overall
    n = len(cfd_s)
    if n > 0:
        bias = float(np.mean(cfd_s - obs_s))
        rmse = float(np.sqrt(np.mean((cfd_s - obs_s)**2)))
        r = float(np.corrcoef(cfd_s, obs_s)[0, 1]) if n > 1 else np.nan
        print("-" * 42)
        print(f"{'TOTAL':>8s}  {n:3d}  {bias:+7.2f}  {rmse:7.2f}  {r:6.3f}")


def plot_scatter(matched: dict, output_dir: Path) -> None:
    """Scatter plot: CFD vs observed wind speed."""
    cfd_s = matched["cfd_speed"]
    obs_s = matched["obs_speed"]
    if len(cfd_s) == 0:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(obs_s, cfd_s, s=30, alpha=0.7, edgecolors="k", linewidths=0.5)

    vmax = max(cfd_s.max(), obs_s.max()) * 1.1
    ax.plot([0, vmax], [0, vmax], "k--", lw=0.8, label="1:1")
    ax.set_xlim(0, vmax)
    ax.set_ylim(0, vmax)
    ax.set_xlabel("Observed wind speed [m/s]")
    ax.set_ylabel("CFD wind speed [m/s]")
    ax.set_title("CFD vs Observed — all towers")
    ax.set_aspect("equal")

    bias = float(np.mean(cfd_s - obs_s))
    rmse = float(np.sqrt(np.mean((cfd_s - obs_s)**2)))
    ax.text(0.05, 0.92, f"bias={bias:+.2f} m/s\nRMSE={rmse:.2f} m/s\nN={len(cfd_s)}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.8))

    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "scatter_speed.png", dpi=150)
    plt.close(fig)
    logger.info("Scatter plot saved: %s", output_dir / "scatter_speed.png")


def plot_profiles(matched: dict, obs: dict, cfd_rows: list[dict],
                  output_dir: Path, towers: list[str] | None = None) -> None:
    """Vertical profiles at selected towers."""
    if towers is None:
        towers = ["tse04", "tse09", "tse13"]

    n_towers = len([t for t in towers if t in obs])
    if n_towers == 0:
        return

    fig, axes = plt.subplots(1, n_towers, figsize=(4 * n_towers, 6), sharey=True)
    if n_towers == 1:
        axes = [axes]

    idx = 0
    for tid in towers:
        if tid not in obs:
            continue
        ax = axes[idx]
        idx += 1

        # Obs profile
        obs_h = np.array(obs[tid]["heights"])
        obs_spd = np.array(obs[tid]["speed"])
        ax.plot(obs_spd, obs_h, "ko-", ms=5, label="Obs", zorder=3)

        # CFD profile
        cfd_h = []
        cfd_spd = []
        for row in cfd_rows:
            if row["tower_id"] == tid:
                cfd_h.append(row["height_m"])
                cfd_spd.append(row["speed"])
        if cfd_h:
            order = np.argsort(cfd_h)
            ax.plot(np.array(cfd_spd)[order], np.array(cfd_h)[order],
                    "rs--", ms=5, label="CFD", zorder=2)

        ax.set_xlabel("Wind speed [m/s]")
        if idx == 1:
            ax.set_ylabel("Height AGL [m]")
        ax.set_title(tid)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Vertical wind profiles — CFD vs Observed", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "profiles.png", dpi=150)
    plt.close(fig)
    logger.info("Profile plot saved: %s", output_dir / "profiles.png")


def load_venkatraman_profiles(
    csv_path: Path,
    variable: str = "wind_speed_ms",
    models: list[str] | None = None,
) -> dict[str, dict[str, dict]]:
    """Load Venkatraman et al. (2023) digitized profiles.

    Parameters
    ----------
    csv_path : Path to venkatraman2023_digitized.csv.
    variable : Variable to extract (default: wind_speed_ms).
    models : Model columns to extract (default: SF1, SF4, BBSF1).

    Returns
    -------
    {tower_id: {model_name: {"heights": [...], "speed": [...]}}}
    """
    if models is None:
        models = ["SF1", "SF4", "BBSF1"]

    data: dict[str, dict[str, dict]] = {}
    with open(csv_path) as f:
        # Filter comment lines before DictReader
        lines = [line for line in f if not line.startswith("#")]
    import io
    reader = csv.DictReader(io.StringIO("".join(lines)))
    for row in reader:
        if not row.get("variable"):
            continue
        if row["variable"] != variable:
            continue
        tid = row["tower_id"]
        z = float(row["z_m"])
        if tid not in data:
            data[tid] = {m: {"heights": [], "speed": []} for m in models}
            data[tid]["obs_ref"] = {"heights": [], "speed": [], "speed_std": []}

        # Obs reference from Venkatraman
        obs_val = row.get("obs", "")
        if obs_val:
            data[tid]["obs_ref"]["heights"].append(z)
            data[tid]["obs_ref"]["speed"].append(float(obs_val))
            std_val = row.get("obs_std", "")
            data[tid]["obs_ref"]["speed_std"].append(float(std_val) if std_val else 0)

        for m in models:
            val = row.get(m, "")
            if val:
                data[tid][m]["heights"].append(z)
                data[tid][m]["speed"].append(float(val))

    return data


def plot_multi_profiles(
    multi_cfd: dict[str, list[dict]],
    obs: dict,
    output_dir: Path,
    towers: list[str] | None = None,
    venkatraman_csv: Path | None = None,
    venkatraman_models: list[str] | None = None,
) -> None:
    """Overlay vertical profiles from multiple CFD cases on the same axes.

    Parameters
    ----------
    multi_cfd : {label: cfd_rows} — one entry per case to overlay.
    obs : Observation dict from load_obs_snapshot().
    output_dir : Directory for output figures.
    towers : Tower IDs to plot (default: tse04, tse09, tse13).
    venkatraman_csv : Path to venkatraman2023_digitized.csv (optional overlay).
    venkatraman_models : Which Venkatraman models to show (default: SF1, SF4, BBSF1).
    """
    if towers is None:
        towers = ["tse04", "tse09", "tse13"]

    active_towers = [t for t in towers if t in obs]
    n_towers = len(active_towers)
    if n_towers == 0:
        return

    # Load Venkatraman data if available
    vk_data = None
    if venkatraman_csv and venkatraman_csv.exists():
        vk_data = load_venkatraman_profiles(
            venkatraman_csv, models=venkatraman_models or ["SF1", "SF4", "BBSF1"],
        )
        logger.info("Loaded Venkatraman profiles for %d towers", len(vk_data))

    n_our_cases = len(multi_cfd)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_our_cases, 1)))
    vk_colors = {"SF1": "#aaaaaa", "SF4": "#888888", "BBSF1": "#555555"}
    vk_styles = {"SF1": ":", "SF4": "-.", "BBSF1": "-."}

    fig, axes = plt.subplots(1, n_towers, figsize=(4.5 * n_towers, 7), sharey=True)
    if n_towers == 1:
        axes = [axes]

    for idx, tid in enumerate(active_towers):
        ax = axes[idx]

        # Obs profile
        obs_h = np.array(obs[tid]["heights"])
        obs_spd = np.array(obs[tid]["speed"])
        ax.plot(obs_spd, obs_h, "ko-", ms=5, label="Obs", zorder=10)

        # Our CFD profiles (one line per case)
        for ci, (label, cfd_rows) in enumerate(multi_cfd.items()):
            cfd_h = []
            cfd_spd = []
            for row in cfd_rows:
                if row["tower_id"] == tid:
                    cfd_h.append(row["height_m"])
                    cfd_spd.append(row["speed"])
            if cfd_h:
                order = np.argsort(cfd_h)
                ax.plot(
                    np.array(cfd_spd)[order], np.array(cfd_h)[order],
                    marker="s", ms=4, linestyle="--", color=colors[ci],
                    label=label, zorder=5 - ci * 0.1,
                )

        # Venkatraman overlay
        if vk_data and tid in vk_data:
            vk_models = venkatraman_models or ["SF1", "SF4", "BBSF1"]
            for m in vk_models:
                if m in vk_data[tid] and vk_data[tid][m]["heights"]:
                    h = np.array(vk_data[tid][m]["heights"])
                    s = np.array(vk_data[tid][m]["speed"])
                    order = np.argsort(h)
                    ax.plot(
                        s[order], h[order],
                        marker="^", ms=3, linestyle=vk_styles.get(m, ":"),
                        color=vk_colors.get(m, "#999999"),
                        label=f"Vk {m}", zorder=3, alpha=0.7,
                    )

        ax.set_xlabel("Wind speed [m/s]")
        if idx == 0:
            ax.set_ylabel("Height AGL [m]")
        ax.set_title(tid)
        ax.grid(True, alpha=0.3)

    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               ncol=min(len(handles), 6), fontsize=7, framealpha=0.9)
    fig.suptitle("Vertical wind profiles — multi-case comparison", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(output_dir / "profiles_multi.png", dpi=200)
    plt.close(fig)
    logger.info("Multi-case profile plot saved: %s", output_dir / "profiles_multi.png")


def print_multi_summary(multi_matched: dict[str, dict]) -> None:
    """Print comparison table across multiple cases."""
    print(f"\n{'case':>20s}  {'N':>4s}  {'bias':>7s}  {'RMSE':>7s}  {'HR(%)':>6s}")
    print("-" * 52)
    for label, matched in sorted(multi_matched.items()):
        cfd_s = matched["cfd_speed"]
        obs_s = matched["obs_speed"]
        n = len(cfd_s)
        if n == 0:
            print(f"{label:>20s}  {0:4d}  {'—':>7s}  {'—':>7s}  {'—':>6s}")
            continue
        bias = float(np.mean(cfd_s - obs_s))
        rmse = float(np.sqrt(np.mean((cfd_s - obs_s)**2)))
        threshold = np.maximum(2.0, 0.3 * obs_s)
        hit_rate = float(np.mean(np.abs(cfd_s - obs_s) < threshold)) * 100
        print(f"{label:>20s}  {n:4d}  {bias:+7.2f}  {rmse:7.2f}  {hit_rate:6.1f}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Compare CFD vs Perdigao observations")
    parser.add_argument("--cfd-csv",   nargs="+", required=True,
                        help="at_masts.csv file(s). Multiple for multi-case comparison.")
    parser.add_argument("--labels",    nargs="*", default=None,
                        help="Labels for each case (same order as --cfd-csv)")
    parser.add_argument("--obs-zarr",  required=True, help="perdigao_obs.zarr")
    parser.add_argument("--timestamp", required=True, help="ISO-8601 timestamp")
    parser.add_argument("--output",    required=True, help="Output directory for plots")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load obs
    obs = load_obs_snapshot(Path(args.obs_zarr), args.timestamp)

    # Labels
    labels = args.labels if args.labels else [
        Path(p).parent.name for p in args.cfd_csv
    ]
    if len(labels) != len(args.cfd_csv):
        labels = [f"case_{i}" for i in range(len(args.cfd_csv))]

    # Single case: original behavior
    if len(args.cfd_csv) == 1:
        cfd_rows = load_cfd_masts(Path(args.cfd_csv[0]))
        logger.info("CFD: %d rows, Obs: %d towers", len(cfd_rows), len(obs))
        matched = compare(cfd_rows, obs)
        logger.info("Matched: %d pairs", len(matched["cfd_speed"]))
        if len(matched["cfd_speed"]) > 0:
            print_summary(matched)
            plot_scatter(matched, output_dir)
            plot_profiles(matched, obs, cfd_rows, output_dir)
    else:
        # Multi-case mode
        multi_cfd = {}
        multi_matched = {}
        for csv_path, label in zip(args.cfd_csv, labels):
            cfd_rows = load_cfd_masts(Path(csv_path))
            multi_cfd[label] = cfd_rows
            multi_matched[label] = compare(cfd_rows, obs)
            logger.info("Case '%s': %d rows, %d matched",
                        label, len(cfd_rows), len(multi_matched[label]["cfd_speed"]))

        print_multi_summary(multi_matched)
        plot_multi_profiles(multi_cfd, obs, output_dir)

    print(f"\nPlots saved to: {output_dir}/")


if __name__ == "__main__":
    main()
