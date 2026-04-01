"""
validate_fwi_at_towers.py — Compare FWI from observations vs ERA5 vs CFD at Perdigão towers.

Computes FWI at each tower location from:
  1. Observations (u, v, T, RH from perdigao_obs.zarr)
  2. ERA5 baseline (uniform T, RH, wind from era5_perdigao.zarr)
  3. CFD (interpolated at tower positions — requires campaign data)

Shows that terrain-resolved FWI differs from ERA5 uniform FWI,
and validates against observed FWI at 48 towers.

Usage:
    python validate_fwi_at_towers.py \
        --obs data/raw/perdigao_obs.zarr \
        --era5 data/raw/era5_perdigao.zarr \
        --output data/validation/figures/fwi_towers
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from shared.fwi import specific_humidity_to_rh, ffmc, isi, bui, dmc, dc, fwi


def load_obs(obs_path: Path, height_target: float = 10.0):
    """Load observations at a target height, compute wind speed.

    Returns dict with times, site_ids, and per-site arrays of T, RH, ws.
    """
    import zarr

    store = zarr.open_group(str(obs_path), mode="r")
    times_ns = np.asarray(store["coords"]["time"][:])
    times = times_ns.astype("datetime64[ns]")
    site_ids = [b.decode() if isinstance(b, bytes) else str(b)
                for b in store["coords"]["site_id"][:]]
    heights = np.asarray(store["coords"]["height_m"][:])
    lats = np.asarray(store["coords"]["lat"][:])
    lons = np.asarray(store["coords"]["lon"][:])

    # Find nearest height
    h_idx = int(np.argmin(np.abs(heights - height_target)))
    h_actual = float(heights[h_idx])

    u = np.asarray(store["sites"]["u"][:, :, h_idx])    # (n_times, n_sites)
    v = np.asarray(store["sites"]["v"][:, :, h_idx])
    T = np.asarray(store["sites"]["T"][:, :, h_idx])     # °C
    RH = np.asarray(store["sites"]["RH"][:, :, h_idx])   # %

    ws = np.sqrt(u**2 + v**2)  # m/s

    return {
        "times": times,
        "site_ids": site_ids,
        "lats": lats,
        "lons": lons,
        "height_m": h_actual,
        "T_C": T,
        "RH": RH,
        "ws_ms": ws,
        "u": u,
        "v": v,
    }


def load_era5_surface(era5_path: Path, lat: float, lon: float):
    """Load ERA5 surface fields (T2m, wind, humidity) at nearest grid point.

    Returns dict with times, T_C, RH (derived from q), ws_ms.
    """
    import zarr

    store = zarr.open_group(str(era5_path), mode="r")
    times_ns = np.asarray(store["coords"]["time"][:])
    times = times_ns.astype("datetime64[ns]")
    lats = np.asarray(store["coords"]["lat"][:])
    lons = np.asarray(store["coords"]["lon"][:])

    # Nearest grid point
    i_lat = int(np.argmin(np.abs(lats - lat)))
    i_lon = int(np.argmin(np.abs(lons - lon)))

    t2m = np.asarray(store["surface"]["t2m"][:, i_lat, i_lon])  # K
    u10 = np.asarray(store["surface"]["u10"][:, i_lat, i_lon])  # m/s
    v10 = np.asarray(store["surface"]["v10"][:, i_lat, i_lon])  # m/s

    # Get q from lowest pressure level for RH estimate
    levels = np.asarray(store["coords"]["level"][:])
    i_sfc = int(np.argmax(levels))  # highest pressure = lowest altitude
    t_pres = np.asarray(store["pressure"]["t"][:, i_sfc, i_lat, i_lon])  # K
    q_pres = np.asarray(store["pressure"]["q"][:, i_sfc, i_lat, i_lon])  # kg/kg
    p_sfc = float(levels[i_sfc])  # hPa

    rh = specific_humidity_to_rh(q_pres, t_pres, p_sfc)

    return {
        "times": times,
        "T_C": t2m - 273.15,
        "RH": rh,
        "ws_ms": np.sqrt(u10**2 + v10**2),
    }


def compute_daily_noon_fwi(times, T_C, RH, ws_ms, rain_mm=None):
    """Compute daily FWI at noon from time series.

    Aggregates to daily noon values (12:00 UTC ± 1h), then runs FWI system.
    Returns dict with dates and FWI components.
    """
    # Mask aberrant sensor values before aggregation
    T_C = np.where((T_C > -50) & (T_C < 60), T_C, np.nan)
    RH = np.where((RH >= 0) & (RH <= 105), RH, np.nan)
    ws_ms = np.where((ws_ms >= 0) & (ws_ms < 60), ws_ms, np.nan)

    # Find noon timesteps (11:00-13:00 UTC)
    hours = (times - times.astype("datetime64[D]")).astype("timedelta64[h]").astype(int)
    dates = np.unique(times.astype("datetime64[D]"))

    n_days = len(dates)
    T_noon = np.full(n_days, np.nan)
    RH_noon = np.full(n_days, np.nan)
    ws_noon = np.full(n_days, np.nan)

    for i, date in enumerate(dates):
        day_mask = times.astype("datetime64[D]") == date
        noon_mask = day_mask & (hours >= 11) & (hours <= 13)
        if not noon_mask.any():
            # Fallback: use afternoon (13-15 UTC) for Portuguese local noon
            noon_mask = day_mask & (hours >= 13) & (hours <= 15)
        if not noon_mask.any():
            continue

        with np.errstate(all="ignore"):
            T_noon[i] = np.nanmean(T_C[noon_mask])
            RH_noon[i] = np.nanmean(RH[noon_mask])
            ws_noon[i] = np.nanmean(ws_ms[noon_mask])

    # Valid days only
    valid = np.isfinite(T_noon) & np.isfinite(RH_noon) & np.isfinite(ws_noon)
    if valid.sum() < 3:
        return None

    dates_v = dates[valid]
    T_v = T_noon[valid]
    RH_v = np.clip(RH_noon[valid], 0, 100)
    ws_v = ws_noon[valid] * 3.6  # m/s → km/h
    rain_v = np.zeros_like(T_v) if rain_mm is None else rain_mm[valid]
    months_v = (dates_v.astype("datetime64[M]") - dates_v.astype("datetime64[Y]")).astype(int) + 1

    # Run FWI system
    n = len(T_v)
    out = {k: np.zeros(n) for k in ("ffmc", "dmc", "dc", "isi", "bui", "fwi")}
    ffmc_p, dmc_p, dc_p = 85.0, 6.0, 15.0

    for i in range(n):
        ffmc_p = float(ffmc(T_v[i], RH_v[i], ws_v[i], rain_v[i], ffmc_p))
        dmc_p = float(dmc(T_v[i], RH_v[i], rain_v[i], dmc_p, months_v[i]))
        dc_p = float(dc(T_v[i], rain_v[i], dc_p, months_v[i]))
        out["ffmc"][i] = ffmc_p
        out["dmc"][i] = dmc_p
        out["dc"][i] = dc_p
        out["isi"][i] = float(isi(ffmc_p, ws_v[i]))
        out["bui"][i] = float(bui(dmc_p, dc_p))
        out["fwi"][i] = float(fwi(out["isi"][i], out["bui"][i]))

    out["dates"] = dates_v
    out["T_C"] = T_v
    out["RH"] = RH_v
    out["ws_kmh"] = ws_v
    return out


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_fwi_timeseries(obs_fwi_by_site, era5_fwi, output_dir):
    """Time series of FWI: obs at key towers vs ERA5."""
    # Select towers with most valid data
    towers_by_coverage = sorted(
        obs_fwi_by_site.items(),
        key=lambda x: len(x[1]["dates"]) if x[1] is not None else 0,
        reverse=True,
    )
    top_towers = [(k, v) for k, v in towers_by_coverage if v is not None][:6]

    if not top_towers:
        print("  No valid tower FWI data for time series plot")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, (site_id, fwi_data) in enumerate(top_towers):
        ax = axes[idx]
        dates = fwi_data["dates"]
        ax.plot(dates, fwi_data["fwi"], "k-", lw=1.5, label="Obs FWI")
        ax.fill_between(dates, 0, fwi_data["fwi"], alpha=0.3, color="orange")

        # ERA5 FWI on same dates
        if era5_fwi is not None:
            era5_dates = era5_fwi["dates"]
            common = np.isin(dates, era5_dates)
            era5_common = np.isin(era5_dates, dates)
            if common.any():
                ax.plot(era5_dates[era5_common], era5_fwi["fwi"][era5_common],
                        "b--", lw=1.5, label="ERA5 FWI", alpha=0.7)

        ax.set_title(f"{site_id} (n={len(dates)}d)")
        ax.set_ylabel("FWI")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Daily FWI: Observed (towers) vs ERA5 (uniform)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "fwi_timeseries_obs_vs_era5.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fwi_scatter(obs_fwi_by_site, era5_fwi, output_dir):
    """Scatter plot: FWI obs vs ERA5, all towers pooled."""
    obs_vals = []
    era5_vals = []

    if era5_fwi is None:
        return

    for site_id, fwi_data in obs_fwi_by_site.items():
        if fwi_data is None:
            continue
        dates = fwi_data["dates"]
        era5_dates = era5_fwi["dates"]
        for i, d in enumerate(dates):
            j = np.where(era5_dates == d)[0]
            if len(j) > 0:
                obs_vals.append(fwi_data["fwi"][i])
                era5_vals.append(era5_fwi["fwi"][j[0]])

    if not obs_vals:
        return

    obs_vals = np.array(obs_vals)
    era5_vals = np.array(era5_vals)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # FWI scatter
    ax = axes[0]
    ax.scatter(era5_vals, obs_vals, s=8, alpha=0.3, c="darkorange")
    lims = [0, max(obs_vals.max(), era5_vals.max()) * 1.1]
    ax.plot(lims, lims, "k--", lw=0.8)
    ax.set_xlabel("ERA5 FWI (uniform)")
    ax.set_ylabel("Observed FWI (towers)")
    ax.set_title("FWI: Obs vs ERA5")
    rmse = np.sqrt(np.mean((obs_vals - era5_vals)**2))
    bias = np.mean(obs_vals - era5_vals)
    corr = np.corrcoef(obs_vals, era5_vals)[0, 1] if len(obs_vals) > 2 else 0
    ax.text(0.05, 0.95, f"RMSE={rmse:.1f}\nBias={bias:.1f}\nr={corr:.2f}\nn={len(obs_vals)}",
            transform=ax.transAxes, va="top", fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # Histogram of differences
    ax = axes[1]
    diff = obs_vals - era5_vals
    finite_diff = diff[np.isfinite(diff)]
    if len(finite_diff) > 0:
        ax.hist(finite_diff, bins=30, color="steelblue", alpha=0.7, edgecolor="white")
        diff_mean, diff_std = np.mean(finite_diff), np.std(finite_diff)
    else:
        diff_mean, diff_std = 0.0, 0.0
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.set_xlabel("FWI difference (Obs - ERA5)")
    ax.set_ylabel("Count")
    ax.set_title(f"FWI difference distribution\nmean={diff_mean:.1f}, std={diff_std:.1f}")

    # Per-tower spread: for each day, show range of obs FWI vs single ERA5
    ax = axes[2]
    # Group by date
    date_stats = {}
    for site_id, fwi_data in obs_fwi_by_site.items():
        if fwi_data is None:
            continue
        for i, d in enumerate(fwi_data["dates"]):
            d_str = str(d)
            if d_str not in date_stats:
                date_stats[d_str] = []
            date_stats[d_str].append(fwi_data["fwi"][i])

    if date_stats:
        dates_sorted = sorted(date_stats.keys())
        spreads = [np.std(date_stats[d]) for d in dates_sorted if len(date_stats[d]) > 3]
        means = [np.mean(date_stats[d]) for d in dates_sorted if len(date_stats[d]) > 3]
        ax.scatter(means, spreads, s=15, alpha=0.5, c="darkorange")
        ax.set_xlabel("Mean observed FWI across towers")
        ax.set_ylabel("Std FWI across towers (spatial spread)")
        ax.set_title("Spatial variability of observed FWI\n(ERA5 gives 0 spread)")

    plt.tight_layout()
    fig.savefig(output_dir / "fwi_scatter_obs_vs_era5.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_input_validation(obs_data, era5_data, output_dir):
    """Scatter: T, RH, wind observed vs ERA5 (input variables)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (var, label, unit) in enumerate([
        ("T_C", "Temperature", "°C"),
        ("RH", "Relative Humidity", "%"),
        ("ws_kmh", "Wind Speed", "km/h"),
    ]):
        ax = axes[idx]

        obs_all = []
        era5_all = []
        obs_times = obs_data["times"]
        era5_times = era5_data["times"]

        for t_idx in range(len(obs_times)):
            t = obs_times[t_idx]
            # Find nearest ERA5 time
            dt = np.abs((era5_times - t).astype("timedelta64[h]").astype(float))
            j = np.argmin(dt)
            if dt[j] > 3:  # skip if > 3h apart
                continue

            if var == "ws_kmh":
                obs_val = obs_data["ws_ms"][t_idx, :] * 3.6
                era5_val = era5_data["ws_ms"][j] * 3.6
            elif var == "T_C":
                obs_val = obs_data["T_C"][t_idx, :]
                era5_val = era5_data["T_C"][j]
            else:
                obs_val = obs_data["RH"][t_idx, :]
                era5_val = era5_data["RH"][j]

            # Physical range filter: reject sensor errors
            if var == "RH":
                valid = np.isfinite(obs_val) & (obs_val >= 0) & (obs_val <= 105)
            elif var == "T_C":
                valid = np.isfinite(obs_val) & (obs_val > -50) & (obs_val < 60)
            elif var == "ws_kmh":
                valid = np.isfinite(obs_val) & (obs_val >= 0) & (obs_val < 200)
            else:
                valid = np.isfinite(obs_val)
            for v in obs_val[valid]:
                obs_all.append(v)
                era5_all.append(float(era5_val))

        if not obs_all:
            continue

        obs_all = np.array(obs_all)
        era5_all = np.array(era5_all)

        # Subsample for plotting
        n_plot = min(5000, len(obs_all))
        idx_plot = np.random.choice(len(obs_all), n_plot, replace=False)

        ax.scatter(era5_all[idx_plot], obs_all[idx_plot], s=3, alpha=0.15, c="steelblue")
        lims = [min(obs_all.min(), era5_all.min()), max(obs_all.max(), era5_all.max())]
        ax.plot(lims, lims, "k--", lw=0.8)
        ax.set_xlabel(f"ERA5 {label} [{unit}]")
        ax.set_ylabel(f"Observed {label} [{unit}]")
        rmse = np.sqrt(np.mean((obs_all - era5_all)**2))
        bias = np.mean(obs_all - era5_all)
        ax.set_title(f"{label}\nRMSE={rmse:.1f}, Bias={bias:.1f}")

    plt.suptitle("Input variables: 48 towers vs ERA5", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "input_vars_obs_vs_era5.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--obs", type=click.Path(exists=True, path_type=Path), required=True,
              help="Perdigão observations Zarr (perdigao_obs.zarr)")
@click.option("--era5", type=click.Path(exists=True, path_type=Path), required=True,
              help="ERA5 Zarr (era5_perdigao.zarr)")
@click.option("--height", type=float, default=10.0, help="Tower height for comparison [m]")
@click.option("--output", type=click.Path(path_type=Path), required=True,
              help="Output directory for figures and stats")
def main(obs: Path, era5: Path, height: float, output: Path):
    """Validate FWI at Perdigão towers: obs vs ERA5."""
    output.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    print("Loading observations...")
    obs_data = load_obs(obs, height_target=height)
    n_sites = len(obs_data["site_ids"])
    print(f"  {n_sites} sites, height={obs_data['height_m']}m, "
          f"{len(obs_data['times'])} timesteps")

    # Count valid T+RH per site
    for i, sid in enumerate(obs_data["site_ids"]):
        n_t = np.sum(np.isfinite(obs_data["T_C"][:, i]))
        n_rh = np.sum(np.isfinite(obs_data["RH"][:, i]))
        if n_t > 100:
            print(f"  {sid}: T={n_t}, RH={n_rh} valid")

    print("\nLoading ERA5...")
    # Use site centroid for ERA5 grid point
    lat_c = float(np.nanmean(obs_data["lats"]))
    lon_c = float(np.nanmean(obs_data["lons"]))
    era5_data = load_era5_surface(era5, lat_c, lon_c)
    print(f"  {len(era5_data['times'])} timesteps, "
          f"T=[{era5_data['T_C'].min():.1f}, {era5_data['T_C'].max():.1f}] °C")

    # Compute daily noon FWI per tower
    print("\nComputing FWI per tower...")
    obs_fwi_by_site = {}
    for i, sid in enumerate(obs_data["site_ids"]):
        result = compute_daily_noon_fwi(
            obs_data["times"],
            obs_data["T_C"][:, i],
            obs_data["RH"][:, i],
            obs_data["ws_ms"][:, i],
        )
        obs_fwi_by_site[sid] = result
        if result is not None:
            print(f"  {sid}: {len(result['dates'])}d, "
                  f"FWI mean={np.mean(result['fwi']):.1f}, max={np.max(result['fwi']):.1f}")

    # ERA5 FWI (single time series, uniform)
    print("\nComputing ERA5 FWI...")
    era5_fwi = compute_daily_noon_fwi(
        era5_data["times"],
        era5_data["T_C"],
        era5_data["RH"],
        era5_data["ws_ms"],
    )
    if era5_fwi is not None:
        print(f"  ERA5: {len(era5_fwi['dates'])}d, "
              f"FWI mean={np.mean(era5_fwi['fwi']):.1f}, max={np.max(era5_fwi['fwi']):.1f}")

    # Plots
    print("\nGenerating figures...")
    plot_fwi_timeseries(obs_fwi_by_site, era5_fwi, output)
    plot_fwi_scatter(obs_fwi_by_site, era5_fwi, output)
    plot_input_validation(obs_data, era5_data, output)

    # Save summary stats
    stats = {
        "n_sites": n_sites,
        "height_m": obs_data["height_m"],
        "sites_with_fwi": sum(1 for v in obs_fwi_by_site.values() if v is not None),
        "era5_fwi_mean": float(np.mean(era5_fwi["fwi"])) if era5_fwi else None,
        "era5_fwi_max": float(np.max(era5_fwi["fwi"])) if era5_fwi else None,
    }

    # Per-site stats
    site_stats = []
    for sid, fwi_data in obs_fwi_by_site.items():
        if fwi_data is None:
            continue
        site_stats.append({
            "site_id": sid,
            "n_days": len(fwi_data["dates"]),
            "fwi_mean": float(np.mean(fwi_data["fwi"])),
            "fwi_max": float(np.max(fwi_data["fwi"])),
            "fwi_std": float(np.std(fwi_data["fwi"])),
            "T_mean": float(np.mean(fwi_data["T_C"])),
            "RH_mean": float(np.mean(fwi_data["RH"])),
            "ws_mean_kmh": float(np.mean(fwi_data["ws_kmh"])),
        })
    stats["sites"] = sorted(site_stats, key=lambda x: -x["fwi_max"])

    with open(output / "fwi_tower_stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)

    print(f"\nDone. Results in {output}/")
    print(f"  Sites with FWI: {stats['sites_with_fwi']}/{n_sites}")
    if site_stats:
        fwi_maxes = [s["fwi_max"] for s in site_stats]
        print(f"  FWI max across towers: {max(fwi_maxes):.1f}")
        print(f"  FWI max range: [{min(fwi_maxes):.1f}, {max(fwi_maxes):.1f}]")
        if era5_fwi:
            print(f"  ERA5 FWI max: {stats['era5_fwi_max']:.1f}")


if __name__ == "__main__":
    main()
