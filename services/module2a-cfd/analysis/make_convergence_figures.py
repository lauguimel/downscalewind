"""
make_convergence_figures.py — Convergence study figures for traceability

Produces:
  1. convergence_at_masts.png   — RMSE vs resolution + per-tower bar chart
  2. profiles_convergence.png   — Vertical wind profiles at key towers
  3. scatter_convergence.png    — Scatter CFD vs obs (one panel per resolution)
  4. mesh_cross_section.png     — Cross-section through both ridges
  5. terrain_shaded.png         — Terrain mesh with hillshade

Usage:
    cd /Users/guillaume/Documents/Recherche/downscalewind
    conda run -n downscalewind python services/module2a-cfd/make_convergence_figures.py
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

logger = logging.getLogger(__name__)

# -- Paths --
ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"
OBS_ZARR = DATA / "raw" / "perdigao_obs.zarr"
CFD_DB = DATA / "cfd-database" / "perdigao"
CASES_DIR = DATA / "convergence" / "cases"
TIMESTAMP = "2017-05-11T12:00"
OUTPUT_DIR = DATA / "validation" / "figures" / "convergence"

# Available resolutions and their case dirs (for mesh viz)
RESOLUTIONS = [500, 250, 100]
CSV_PATHS = {
    res: CFD_DB / f"20170511T12_{res}m" / "at_masts.csv"
    for res in RESOLUTIONS
}
# For mesh visualization, use 231° cases
MESH_CASES = {
    500: CASES_DIR / "rheotool_001",
    250: CASES_DIR / "rheotool_003",
    100: CASES_DIR / "rheotool_005",
}

COLORS = {500: "#1b9e77", 250: "#d95f02", 100: "#7570b3", 50: "#e7298a"}
KEY_TOWERS = ["tse04", "tse09", "tse13"]

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


# ------------------------------------------------------------------ #
# Data loading
# ------------------------------------------------------------------ #

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
    m = {"tower_id": [], "height": [], "cfd_speed": [], "obs_speed": []}
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
                break
    for k in ("height", "cfd_speed", "obs_speed"):
        m[k] = np.array(m[k])
    return m


def compute_metrics(matched: dict) -> dict:
    cfd_s, obs_s = matched["cfd_speed"], matched["obs_speed"]
    n = len(cfd_s)
    if n == 0:
        return {"n": 0, "rmse": np.nan, "bias": np.nan, "r": np.nan, "mae": np.nan}
    bias = float(np.mean(cfd_s - obs_s))
    rmse = float(np.sqrt(np.mean((cfd_s - obs_s)**2)))
    mae = float(np.mean(np.abs(cfd_s - obs_s)))
    r = float(np.corrcoef(cfd_s, obs_s)[0, 1]) if n > 1 else np.nan
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


# ------------------------------------------------------------------ #
# Figure 1: Convergence — RMSE vs resolution + per-tower bars
# ------------------------------------------------------------------ #

def fig_convergence(resolutions, metrics_list, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    res = np.array(resolutions, dtype=float)
    rmse = np.array([m["rmse"] for m in metrics_list])
    mae = np.array([m["mae"] for m in metrics_list])

    ax1.plot(res, rmse, "ko-", ms=8, zorder=5, label="RMSE")
    ax1.plot(res, mae, "s--", color="gray", ms=6, zorder=4, label="MAE")

    if len(res) >= 3:
        from scipy.optimize import curve_fit
        def power_law(x, a, p):
            return a * x**p
        try:
            popt, _ = curve_fit(power_law, res, rmse, p0=[0.01, 1.0])
            x_fit = np.linspace(50, 600, 100)
            ax1.plot(x_fit, power_law(x_fit, *popt), "k:", lw=1.0,
                     label=f"fit: {popt[0]:.3f}$\\cdot\\Delta x^{{{popt[1]:.2f}}}$")
        except Exception:
            pass

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax1.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax1.set_xticks(resolutions)
    ax1.set_xlabel("Horizontal resolution $\\Delta x$ [m]")
    ax1.set_ylabel("Error [m/s]")
    ax1.set_title("(a) Mesh convergence — wind speed")
    ax1.legend(loc="upper left")
    ax1.set_xlim(80, 600)

    # Per-tower RMSE bars
    x_pos = np.arange(len(KEY_TOWERS))
    width = 0.25
    for i, (r, m) in enumerate(zip(resolutions, metrics_list)):
        rmses = [m.get("tower_metrics", {}).get(t, {}).get("rmse", np.nan) for t in KEY_TOWERS]
        ax2.bar(x_pos + i * width, rmses, width, color=COLORS.get(r, f"C{i}"),
                label=f"{r} m", edgecolor="k", linewidth=0.5)
    ax2.set_xticks(x_pos + width)
    ax2.set_xticklabels(KEY_TOWERS)
    ax2.set_ylabel("RMSE [m/s]")
    ax2.set_title("(b) Per-tower RMSE")
    ax2.legend()

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"convergence_at_masts.{ext}")
    plt.close(fig)
    logger.info("Saved convergence_at_masts")


# ------------------------------------------------------------------ #
# Figure 2: Vertical profiles at key towers
# ------------------------------------------------------------------ #

def fig_profiles(resolutions, cfd_data, obs, output_dir):
    towers = [t for t in KEY_TOWERS if t in obs]
    if not towers:
        logger.warning("No key towers in obs — skipping profiles")
        return

    fig, axes = plt.subplots(1, len(towers), figsize=(4.2 * len(towers), 5.5), sharey=True)
    if len(towers) == 1:
        axes = [axes]

    for ti, tid in enumerate(towers):
        ax = axes[ti]
        obs_h = np.array(obs[tid]["heights"])
        obs_spd = np.array(obs[tid]["speed"])
        ax.plot(obs_spd, obs_h, "ko-", ms=5, lw=2, label="Obs", zorder=5)

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
                 "Perdigao, 2017-05-11 12:00 UTC, SSW 217deg",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"profiles_convergence.{ext}")
    plt.close(fig)
    logger.info("Saved profiles_convergence")


# ------------------------------------------------------------------ #
# Figure 3: Scatter CFD vs obs
# ------------------------------------------------------------------ #

def fig_scatter(resolutions, matched_list, output_dir):
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
            ax.set_title(f"$\\Delta x$ = {res} m\n(no data)")
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
        ax.set_title(f"$\\Delta x$ = {res} m", fontweight="bold")
        ax.set_xlabel("Observed [m/s]")
        if i == 0:
            ax.set_ylabel("CFD [m/s]")

    fig.suptitle("CFD vs Observed wind speed — all towers", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"scatter_convergence.{ext}")
    plt.close(fig)
    logger.info("Saved scatter_convergence")


# ------------------------------------------------------------------ #
# Figure 4: Mesh cross-section through both ridges
# ------------------------------------------------------------------ #

def fig_mesh_cross_section(output_dir):
    """Cross-section perpendicular to ridges (NW-SE direction).

    The ridges are oriented NE-SW (40 deg azimuth), so we cut at ~130 deg.
    We slice along y=0 (which crosses both ridges in the 10km domain).
    """
    import pyvista as pv

    case_dir = MESH_CASES[100]
    stl_path = case_dir / "constant" / "triSurface" / "terrain.stl"
    if not stl_path.exists():
        logger.warning("STL not found: %s — skipping mesh cross-section", stl_path)
        return

    terrain = pv.read(str(stl_path))

    # Get terrain elevation range
    z_min_terrain = terrain.bounds[4]
    z_max_terrain = terrain.bounds[5]

    # Cross-section: slice terrain along Y=0 (perpendicular to ridges)
    # Get terrain profile along y=0
    x_range = np.linspace(-5000, 5000, 500)
    z_profile = np.zeros_like(x_range)

    # Sample terrain height by slicing at multiple y positions near y=0
    slice_terrain = terrain.slice(normal="y", origin=(0, 0, 0))
    if slice_terrain.n_points > 0:
        pts = slice_terrain.points
        sort_idx = np.argsort(pts[:, 0])
        x_pts = pts[sort_idx, 0]
        z_pts = pts[sort_idx, 2]
        z_profile = np.interp(x_range, x_pts, z_pts)

    z_lo = z_min_terrain - 100
    z_hi = z_max_terrain + 500

    fig, ax = plt.subplots(figsize=(12, 5))

    # Fill terrain
    ax.fill_between(x_range / 1000, z_lo * np.ones_like(x_range), z_profile,
                     color="#8B7355", alpha=0.6, label="Terrain")
    ax.plot(x_range / 1000, z_profile, "k-", lw=1.5)

    # Draw mesh cell edges for each resolution
    for res in [500, 250, 100]:
        case = MESH_CASES.get(res)
        if not case:
            continue
        stl_r = case / "constant" / "triSurface" / "terrain.stl"
        if not stl_r.exists():
            continue
        terr_r = pv.read(str(stl_r))
        sl = terr_r.slice(normal="y", origin=(0, 0, 0))
        if sl.n_points > 0:
            pts = sl.points
            sort_idx = np.argsort(pts[:, 0])
            x_p = pts[sort_idx, 0] / 1000
            z_p = pts[sort_idx, 2]
            # Draw vertical cell edge markers at resolution intervals
            for x_cell in np.arange(-5, 5, res / 1000):
                idx = np.argmin(np.abs(x_p - x_cell))
                if idx < len(z_p):
                    ax.plot([x_cell, x_cell], [z_p[idx], z_p[idx] + res],
                            color=COLORS[res], lw=0.6, alpha=0.5)

    # Show cell size indicators
    for res in RESOLUTIONS:
        y_pos = z_hi - 50 - RESOLUTIONS.index(res) * 80
        ax.annotate("", xy=(0, y_pos), xytext=(res / 1000, y_pos),
                     arrowprops=dict(arrowstyle="<->", color=COLORS[res], lw=2))
        ax.text(res / 2000, y_pos + 20, f"{res} m",
                ha="center", fontsize=8, color=COLORS[res], fontweight="bold")

    ax.set_xlabel("Distance [km] — perpendicular to ridges")
    ax.set_ylabel("Elevation [m ASL]")
    ax.set_title("Mesh cross-section through both ridges (y = 0)")
    ax.set_xlim(-5, 5)
    ax.set_ylim(z_lo, z_hi)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"mesh_cross_section.{ext}")
    plt.close(fig)
    logger.info("Saved mesh_cross_section")


# ------------------------------------------------------------------ #
# Figure 5: Terrain shaded relief
# ------------------------------------------------------------------ #

def _render_terrain_panel(ax, stl_path, res_label):
    """Render a single terrain hillshade panel onto ax."""
    import pyvista as pv
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import Normalize

    terrain = pv.read(str(stl_path))
    terrain = terrain.compute_normals(cell_normals=True, point_normals=False)
    normals = terrain.cell_data["Normals"]

    # Sun direction: azimuth 315 deg (NW), elevation 45 deg
    sun_az_rad = np.radians(315)
    sun_el_rad = np.radians(45)
    sun_dir = np.array([
        np.cos(sun_el_rad) * np.sin(sun_az_rad),
        np.cos(sun_el_rad) * np.cos(sun_az_rad),
        np.sin(sun_el_rad),
    ])

    shade = np.clip(np.dot(normals, sun_dir), 0.1, 1.0)
    cell_centers = terrain.cell_centers().points
    elev = cell_centers[:, 2]

    # Blend hillshade with elevation
    elev_norm = (elev - 100) / (600 - 100 + 1e-6)  # fixed range for consistent colormap
    combined = 0.6 * shade + 0.4 * np.clip(elev_norm, 0, 1)

    faces = terrain.faces.reshape(-1, 4)[:, 1:]
    pts = terrain.points

    verts_2d = [pts[face][:, :2] / 1000 for face in faces]

    cmap = plt.cm.terrain
    norm = Normalize(vmin=0.0, vmax=1.0)  # fixed range

    # Show triangle edges for coarse meshes to reveal mesh structure
    edge_lw = 0.15 if int(res_label.replace("m", "")) >= 250 else 0
    edge_color = "gray" if edge_lw > 0 else "none"

    poly = PolyCollection(verts_2d, array=combined, cmap=cmap, norm=norm,
                          edgecolors=edge_color, linewidths=edge_lw)
    ax.add_collection(poly)

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.set_title(f"$\\Delta x$ = {res_label}", fontweight="bold", fontsize=12)

    n_faces = terrain.n_cells
    ax.text(0.02, 0.02, f"{n_faces:,} faces",
            transform=ax.transAxes, fontsize=8,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))

    return elev


def fig_terrain_shaded(output_dir):
    """Terrain STL rendered with hillshade shading — one panel per resolution."""
    from matplotlib.colors import Normalize

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    elev_all = None
    for i, (res, case_dir) in enumerate(sorted(MESH_CASES.items(), reverse=True)):
        stl_path = case_dir / "constant" / "triSurface" / "terrain.stl"
        if not stl_path.exists():
            logger.warning("STL not found: %s", stl_path)
            continue
        elev = _render_terrain_panel(axes[i], stl_path, f"{res}m")
        if elev_all is None:
            elev_all = elev
        if i == 0:
            axes[i].set_ylabel("Northing [km]")
        axes[i].set_xlabel("Easting [km]")

    fig.suptitle("Perdigao terrain — mesh resolution comparison (hillshade)",
                 fontsize=13)

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.terrain,
                                norm=Normalize(vmin=100, vmax=600))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, shrink=0.7, label="Elevation [m ASL]",
                 orientation="vertical", pad=0.02)

    fig.tight_layout(rect=[0, 0, 0.92, 0.94])
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"terrain_shaded.{ext}")
    plt.close(fig)
    logger.info("Saved terrain_shaded (3 resolutions)")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load observations
    obs = load_obs(OBS_ZARR, TIMESTAMP)
    logger.info("Obs: %d towers with data at %s", len(obs), TIMESTAMP)

    # Load CFD and compute metrics
    resolutions = []
    cfd_data = []
    matched_list = []
    metrics_list = []

    for res in RESOLUTIONS:
        csv_path = CSV_PATHS[res]
        if not csv_path.exists():
            logger.warning("Missing %s — skipping %dm", csv_path, res)
            continue
        rows = load_cfd(csv_path)
        m = match(rows, obs)
        met = compute_metrics(m)
        resolutions.append(res)
        cfd_data.append(rows)
        matched_list.append(m)
        metrics_list.append(met)
        logger.info("  %dm: %d matched, RMSE=%.2f, bias=%+.2f, R=%.3f",
                     res, met["n"], met["rmse"], met["bias"], met["r"])

    # Summary
    print(f"\n{'dx [m]':>8s}  {'N':>4s}  {'RMSE':>7s}  {'MAE':>7s}  {'bias':>7s}  {'R':>6s}")
    print("-" * 48)
    for res, met in zip(resolutions, metrics_list):
        print(f"{res:>8d}  {met['n']:4d}  {met['rmse']:7.2f}  {met['mae']:7.2f}  "
              f"{met['bias']:+7.2f}  {met['r']:6.3f}")

    # Generate figures
    fig_convergence(resolutions, metrics_list, OUTPUT_DIR)
    fig_profiles(resolutions, cfd_data, obs, OUTPUT_DIR)
    fig_scatter(resolutions, matched_list, OUTPUT_DIR)
    fig_mesh_cross_section(OUTPUT_DIR)
    fig_terrain_shaded(OUTPUT_DIR)

    print(f"\nAll figures saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
