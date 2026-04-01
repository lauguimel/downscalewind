"""
fwi_poc.py — Proof-of-concept: FWI from CFD vs ERA5 baseline.

Computes FWI at ground level from CFD fields (T, q, U) and compares
with FWI from ERA5 inflow (uniform over domain). Shows that terrain-
induced heterogeneity creates spatially varying fire weather.

Produces for each site × timestamp:
  - 2D map of FWI at ~10m AGL (CFD)
  - 2D map of FWI from ERA5 (uniform)
  - Difference map (CFD - ERA5)
  - Histogram of FWI values
  - Summary statistics (mean, std, max, spatial coefficient of variation)

Usage (on UGA):
    python fwi_poc.py \
        --campaign-dir /home/guillaume/dsw/campaign_9k \
        --sites site_00025 site_00183 site_00093 \
        --timestamps 12 \
        --output /home/guillaume/dsw/fwi_poc_results
"""
from __future__ import annotations

import json
from pathlib import Path

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

# FWI module (copied to ~/dsw/)
import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path.home() / "dsw"))
from fwi import specific_humidity_to_rh, ffmc, isi, bui, dmc, dc, fwi, compute_fwi_field


def load_site(campaign_dir: Path, site_id: str):
    """Load stacked Zarr for a site."""
    import zarr
    zpath = campaign_dir / site_id / f"{site_id}.zarr"
    return zarr.open_group(str(zpath), mode="r")


def extract_ground_level(store, ts_idx: int, z_agl_max: float = 30.0):
    """Extract near-ground fields for a single timestamp.

    Returns dict with x, y, z_agl, elev, U_speed, T, q, rh, n_cells.
    For cells with z_agl < z_agl_max, picks the lowest cell per (x, y) column.
    """
    x = np.asarray(store["coords"]["x"])
    y = np.asarray(store["coords"]["y"])
    z_agl = np.asarray(store["coords"]["z_agl"])
    elev = np.asarray(store["coords"]["elev"])

    U = np.asarray(store["U"][ts_idx])  # (n_cells, 3)
    T = np.asarray(store["T"][ts_idx])  # (n_cells,)
    q = np.asarray(store["q"][ts_idx])  # (n_cells,)

    # Select near-ground cells
    mask = (z_agl > 2.0) & (z_agl < z_agl_max)

    # For proper 2D mapping: bin into horizontal grid and take lowest z_agl per bin
    # Use 100m horizontal bins
    bin_size = 100.0
    x_bins = ((x[mask] - x[mask].min()) / bin_size).astype(int)
    y_bins = ((y[mask] - y[mask].min()) / bin_size).astype(int)
    bin_ids = x_bins * 10000 + y_bins

    # For each bin, keep the cell with lowest z_agl
    unique_bins = np.unique(bin_ids)
    indices = np.where(mask)[0]
    selected = []
    for b in unique_bins:
        bin_mask = bin_ids == b
        bin_indices = indices[bin_mask]
        lowest = bin_indices[np.argmin(z_agl[bin_indices])]
        selected.append(lowest)
    selected = np.array(selected)

    speed = np.sqrt(np.sum(U[selected] ** 2, axis=-1))
    q_clipped = np.clip(q[selected], 0, None)

    return {
        "x": x[selected],
        "y": y[selected],
        "z_agl": z_agl[selected],
        "elev": elev[selected],
        "U_speed": speed,
        "T_K": T[selected],
        "q": q_clipped,
        "n_cells": len(selected),
    }


def compute_era5_fwi(store, ts_idx: int):
    """Compute FWI from ERA5 inflow (uniform over domain).

    Uses the lowest inflow level as representative of near-surface conditions.
    """
    T_ref = float(store["meta"]["T_ref"][ts_idx])
    q_ref = float(store["meta"]["q_ref"][ts_idx])
    u_hub = float(store["meta"]["u_hub"][ts_idx])

    # Also get lowest inflow level for more accurate surface estimate
    z_levels = np.asarray(store["inflow"]["z_levels"][ts_idx])
    T_profile = np.asarray(store["inflow"]["T_profile"][ts_idx])
    ux_profile = np.asarray(store["inflow"]["ux_profile"][ts_idx])
    uy_profile = np.asarray(store["inflow"]["uy_profile"][ts_idx])
    q_profile = np.asarray(store["inflow"]["q_profile"][ts_idx])

    # Use ~10m level
    valid = z_levels > 0
    z_v = z_levels[valid]
    if len(z_v) == 0:
        return {"fwi": 0.0, "isi": 0.0, "T_C": T_ref - 273.15, "rh": 50.0, "ws_kmh": u_hub * 3.6}

    i_10m = np.argmin(np.abs(z_v - 10.0))
    T_sfc = float(T_profile[valid][i_10m])
    q_sfc = max(float(q_profile[valid][i_10m]), 0.0)
    ux_sfc = float(ux_profile[valid][i_10m])
    uy_sfc = float(uy_profile[valid][i_10m])

    # Assume ~950 hPa near surface
    p_hpa = 950.0
    rh = float(specific_humidity_to_rh(q_sfc, T_sfc, p_hpa))
    ws_kmh = float(np.sqrt(ux_sfc**2 + uy_sfc**2) * 3.6)
    T_C = T_sfc - 273.15

    # Compute FWI (single step, default previous values)
    ffmc_val = float(ffmc(T_C, rh, ws_kmh, 0.0))
    isi_val = float(isi(ffmc_val, ws_kmh))
    dmc_val = float(dmc(T_C, rh, 0.0, 6.0, 7))
    dc_val = float(dc(T_C, 0.0, 15.0, 7))
    bui_val = float(bui(dmc_val, dc_val))
    fwi_val = float(fwi(isi_val, bui_val))

    return {"fwi": fwi_val, "isi": isi_val, "T_C": T_C, "rh": rh, "ws_kmh": ws_kmh,
            "ffmc": ffmc_val, "dmc": dmc_val, "dc": dc_val, "bui": bui_val}


def plot_fwi_comparison(ground, fwi_cfd, fwi_era5, era5_info, site_id, ts_idx, output_dir):
    """Generate comparison figure: CFD FWI map, ERA5 FWI, difference, histogram."""
    x = ground["x"]
    y = ground["y"]

    # Create triangulation for 2D plotting
    triang = tri.Triangulation(x, y)
    # Remove large triangles (gaps in mesh)
    xy = np.column_stack([x, y])
    centroids = np.mean(xy[triang.triangles], axis=1)
    edge_lens = []
    for i in range(3):
        j = (i + 1) % 3
        d = np.sqrt(np.sum((xy[triang.triangles[:, i]] - xy[triang.triangles[:, j]])**2, axis=1))
        edge_lens.append(d)
    max_edge = np.max(edge_lens, axis=0)
    mask = max_edge > 300  # remove triangles > 300m
    triang.set_mask(mask)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Row 1: FWI map (CFD), ERA5 uniform, Difference
    vmin, vmax = 0, max(np.percentile(fwi_cfd["fwi"], 99), era5_info["fwi"] * 1.5, 5)

    # CFD FWI
    ax = axes[0, 0]
    im = ax.tricontourf(triang, fwi_cfd["fwi"], levels=30, cmap="YlOrRd", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="FWI")
    ax.set_title(f"CFD FWI (ground level)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")

    # ERA5 uniform
    ax = axes[0, 1]
    era5_uniform = np.full_like(fwi_cfd["fwi"], era5_info["fwi"])
    im = ax.tricontourf(triang, era5_uniform, levels=30, cmap="YlOrRd", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="FWI")
    ax.set_title(f"ERA5 FWI (uniform = {era5_info['fwi']:.1f})")
    ax.set_xlabel("x [m]")
    ax.set_aspect("equal")

    # Difference
    ax = axes[0, 2]
    diff = fwi_cfd["fwi"] - era5_info["fwi"]
    dlim = max(abs(np.percentile(diff, 5)), abs(np.percentile(diff, 95)), 1)
    im = ax.tricontourf(triang, diff, levels=30, cmap="RdBu_r", vmin=-dlim, vmax=dlim)
    plt.colorbar(im, ax=ax, label="FWI (CFD - ERA5)")
    ax.set_title("FWI difference (CFD - ERA5)")
    ax.set_xlabel("x [m]")
    ax.set_aspect("equal")

    # Row 2: Wind speed, Temperature, ISI
    # Wind speed
    ax = axes[1, 0]
    im = ax.tricontourf(triang, ground["U_speed"], levels=30, cmap="viridis")
    plt.colorbar(im, ax=ax, label="Wind speed [m/s]")
    ax.set_title("Wind speed (ground)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")

    # Temperature
    ax = axes[1, 1]
    T_C = ground["T_K"] - 273.15
    im = ax.tricontourf(triang, T_C, levels=30, cmap="RdYlBu_r")
    plt.colorbar(im, ax=ax, label="T [°C]")
    ax.set_title("Temperature (ground)")
    ax.set_xlabel("x [m]")
    ax.set_aspect("equal")

    # ISI map (fire spread rate — most wind-sensitive component)
    ax = axes[1, 2]
    im = ax.tricontourf(triang, fwi_cfd["isi"], levels=30, cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="ISI")
    ax.set_title("ISI (Initial Spread Index)")
    ax.set_xlabel("x [m]")
    ax.set_aspect("equal")

    wind_dir = float(np.asarray(store["meta"]["wind_dir"][ts_idx]))
    fig.suptitle(
        f"{site_id} — ts{ts_idx:03d} | "
        f"ERA5: T={era5_info['T_C']:.1f}°C, RH={era5_info['rh']:.0f}%, "
        f"ws={era5_info['ws_kmh']:.0f} km/h, FWI={era5_info['fwi']:.1f} | "
        f"wind_dir={wind_dir:.0f}°",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()

    out_path = output_dir / f"{site_id}_ts{ts_idx:03d}_fwi.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_summary_stats(all_stats, output_dir):
    """Bar chart comparing spatial variability across sites."""
    if not all_stats:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    labels = [s["label"] for s in all_stats]
    x = np.arange(len(labels))

    # FWI: ERA5 vs CFD mean/max
    ax = axes[0]
    era5_fwi = [s["era5_fwi"] for s in all_stats]
    cfd_mean = [s["cfd_fwi_mean"] for s in all_stats]
    cfd_max = [s["cfd_fwi_max"] for s in all_stats]
    w = 0.25
    ax.bar(x - w, era5_fwi, w, label="ERA5 (uniform)", color="steelblue")
    ax.bar(x, cfd_mean, w, label="CFD mean", color="orange")
    ax.bar(x + w, cfd_max, w, label="CFD max", color="red")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("FWI")
    ax.set_title("FWI comparison")
    ax.legend()

    # Spatial CV of FWI
    ax = axes[1]
    cv = [s["cfd_fwi_cv"] for s in all_stats]
    ax.bar(x, cv, color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Coefficient of variation")
    ax.set_title("Spatial variability of FWI (CV)")

    # ISI: ERA5 vs CFD
    ax = axes[2]
    era5_isi = [s["era5_isi"] for s in all_stats]
    cfd_isi_mean = [s["cfd_isi_mean"] for s in all_stats]
    cfd_isi_max = [s["cfd_isi_max"] for s in all_stats]
    ax.bar(x - w, era5_isi, w, label="ERA5", color="steelblue")
    ax.bar(x, cfd_isi_mean, w, label="CFD mean", color="orange")
    ax.bar(x + w, cfd_isi_max, w, label="CFD max", color="red")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("ISI")
    ax.set_title("ISI comparison")
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "fwi_poc_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--campaign-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--sites", multiple=True, required=True, help="Site IDs (e.g. site_00025)")
@click.option("--timestamps", multiple=True, type=int, default=[12],
              help="Timestamp indices to analyze (default: 12 = hottest)")
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--z-agl-max", type=float, default=30.0, help="Max height AGL for ground extraction [m]")
def main(campaign_dir: Path, sites: tuple, timestamps: tuple, output: Path, z_agl_max: float):
    """FWI proof-of-concept: terrain-resolved fire weather from CFD."""
    output.mkdir(parents=True, exist_ok=True)

    all_stats = []

    for site_id in sites:
        print(f"\n{'='*60}")
        print(f"Processing {site_id}")
        print(f"{'='*60}")

        global store
        store = load_site(campaign_dir, site_id)

        for ts_idx in timestamps:
            print(f"\n  --- Timestamp {ts_idx} ---")

            # Extract ground-level fields
            ground = extract_ground_level(store, ts_idx, z_agl_max)
            print(f"  Ground cells: {ground['n_cells']}")
            print(f"  Wind speed: [{ground['U_speed'].min():.1f}, {ground['U_speed'].max():.1f}] m/s")
            print(f"  T: [{ground['T_K'].min()-273.15:.1f}, {ground['T_K'].max()-273.15:.1f}] °C")

            # Compute FWI from CFD fields
            p_hpa = 950.0  # approximate surface pressure
            fwi_cfd = compute_fwi_field(
                t_kelvin=ground["T_K"],
                q_kgkg=ground["q"],
                p_hpa=np.full_like(ground["T_K"], p_hpa),
                u_ms=ground["U_speed"],  # scalar speed (conservative: ISI uses |wind|)
                v_ms=np.zeros_like(ground["U_speed"]),
                rain_mm=np.zeros_like(ground["T_K"]),
                month=7,  # assume summer for PoC
            )

            # ERA5 baseline
            era5 = compute_era5_fwi(store, ts_idx)

            # Stats
            fwi_vals = fwi_cfd["fwi"]
            isi_vals = fwi_cfd["isi"]
            stats = {
                "site_id": site_id,
                "ts_idx": ts_idx,
                "label": f"{site_id}\nts{ts_idx:03d}",
                "era5_fwi": era5["fwi"],
                "era5_isi": era5["isi"],
                "cfd_fwi_mean": float(np.mean(fwi_vals)),
                "cfd_fwi_std": float(np.std(fwi_vals)),
                "cfd_fwi_max": float(np.max(fwi_vals)),
                "cfd_fwi_min": float(np.min(fwi_vals)),
                "cfd_fwi_cv": float(np.std(fwi_vals) / max(np.mean(fwi_vals), 0.01)),
                "cfd_isi_mean": float(np.mean(isi_vals)),
                "cfd_isi_max": float(np.max(isi_vals)),
                "cfd_rh_mean": float(np.mean(fwi_cfd["rh"])),
                "cfd_rh_std": float(np.std(fwi_cfd["rh"])),
                "cfd_ws_mean_kmh": float(np.mean(fwi_cfd["ws_kmh"])),
                "n_ground_cells": ground["n_cells"],
                "elev_range_m": float(ground["elev"].max() - ground["elev"].min()),
            }
            all_stats.append(stats)

            print(f"  ERA5 FWI: {era5['fwi']:.1f} (T={era5['T_C']:.1f}°C, RH={era5['rh']:.0f}%, ws={era5['ws_kmh']:.0f} km/h)")
            print(f"  CFD  FWI: mean={stats['cfd_fwi_mean']:.1f}, max={stats['cfd_fwi_max']:.1f}, CV={stats['cfd_fwi_cv']:.2f}")
            print(f"  CFD  ISI: mean={stats['cfd_isi_mean']:.1f}, max={stats['cfd_isi_max']:.1f}")
            print(f"  CFD  RH:  mean={stats['cfd_rh_mean']:.0f}%, std={stats['cfd_rh_std']:.1f}%")
            print(f"  Elev range: {stats['elev_range_m']:.0f} m")

            # Plot
            fig_path = plot_fwi_comparison(ground, fwi_cfd, era5, era5, site_id, ts_idx, output)
            print(f"  Figure: {fig_path}")

    # Summary figure
    plot_summary_stats(all_stats, output)

    # Save stats JSON
    with open(output / "fwi_poc_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nStats: {output / 'fwi_poc_stats.json'}")
    print(f"Summary figure: {output / 'fwi_poc_summary.png'}")


if __name__ == "__main__":
    main()
