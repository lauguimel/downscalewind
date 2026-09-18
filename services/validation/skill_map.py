"""Skill map: where does the chain (raw surrogate / calibrated) beat ERA5?

Joins per-row evaluation predictions (eval_pairings.parquet from eval_devine_style.py) with the
pairings parquet (lat, lon, elev, pop, ERA5 10 m baseline) and per-station relief indices computed
from the 30 m DEM tiles, then reports MAE of ERA5 / raw / corrected and the skill
1 - MAE_model / MAE_ERA5 as a function of relief, observed wind class and height.

Relief indices (per station):
  relief_std_6km   std of elevation in the 6 km x 6 km CFD patch centred on the station
  relief_std_cell  std of elevation in a 0.25 deg box (one ERA5 cell)
  dz_cell          station DEM elevation minus mean elevation of that 0.25 deg box

The ERA5 comparison is restricted to ~10 m rows (ERA5 baseline is the 10 m wind) and to rows whose
ERA5 time offset is <= --max-era5-delta-min. Confidence intervals: bootstrap over STATIONS.
"""
from __future__ import annotations

import math
from pathlib import Path

import click
import numpy as np
import pandas as pd

WIND_BINS = [(0, 3), (3, 6), (6, 10), (10, 60)]


def _tile_name(lat_sw: int, lon_sw: int) -> str:
    ns = f"N{lat_sw:02d}" if lat_sw >= 0 else f"S{-lat_sw:02d}"
    ew = f"E{lon_sw:03d}" if lon_sw >= 0 else f"W{-lon_sw:03d}"
    return f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM.tif"


def _read_box(dem_dir: Path, lat: float, lon: float, half_lat: float, half_lon: float) -> np.ndarray | None:
    """Elevations inside the lat/lon box, mosaicking the 1 deg tiles it touches."""
    import rasterio
    from rasterio.merge import merge

    s, n, w, e = lat - half_lat, lat + half_lat, lon - half_lon, lon + half_lon
    paths = []
    for la in range(math.floor(s), math.floor(n) + 1):
        for lo in range(math.floor(w), math.floor(e) + 1):
            p = dem_dir / _tile_name(la, lo)
            if p.exists():
                paths.append(p)
    if not paths:
        return None
    srcs = [rasterio.open(p) for p in paths]
    try:
        arr, _ = merge(srcs, bounds=(w, s, e, n), nodata=0.0)  # missing tiles = sea
    finally:
        for src in srcs:
            src.close()
    return arr[0].astype(np.float32)


def relief_indices(stations: pd.DataFrame, dem_dir: Path) -> pd.DataFrame:
    rows = []
    for r in stations.itertuples():
        km_lat = 1.0 / 111.0
        km_lon = 1.0 / (111.0 * max(math.cos(math.radians(r.lat)), 0.1))
        patch = _read_box(dem_dir, r.lat, r.lon, 3.0 * km_lat, 3.0 * km_lon)
        cell = _read_box(dem_dir, r.lat, r.lon, 0.125, 0.125)
        if patch is None or cell is None or patch.size == 0:
            rows.append({"base_id": r.base_id})
            continue
        z_st = float(patch[patch.shape[0] // 2, patch.shape[1] // 2])
        rows.append({"base_id": r.base_id, "relief_std_6km": float(patch.std()),
                     "relief_std_cell": float(cell.std()), "dz_cell": z_st - float(cell.mean())})
    return pd.DataFrame(rows)


def _mae_table(d: pd.DataFrame, cols: dict[str, str]) -> dict:
    out = {"n_rows": int(len(d)), "n_stations": int(d.base_id.nunique()), "obs_mean": float(d.speed_obs.mean())}
    for name, c in cols.items():
        out[f"mae_{name}"] = float((d[c] - d.speed_obs).abs().mean())
        out[f"bias_{name}"] = float((d[c] - d.speed_obs).mean())
    return out


def _skill_ci(d: pd.DataFrame, model_col: str, ref_col: str, n_boot: int, rng) -> tuple[float, float, float]:
    """Skill 1 - MAE_model/MAE_ref with a station-level bootstrap 90 % interval."""
    g = d.assign(em=(d[model_col] - d.speed_obs).abs(), er=(d[ref_col] - d.speed_obs).abs()) \
         .groupby("base_id")[["em", "er"]].sum()
    em, er = g.em.to_numpy(), g.er.to_numpy()
    if len(g) < 3 or er.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    boot = 1.0 - em[idx].sum(1) / er[idx].sum(1)
    return float(1.0 - em.sum() / er.sum()), float(np.quantile(boot, 0.05)), float(np.quantile(boot, 0.95))


@click.command()
@click.option("--eval-pairings", type=Path, required=True)
@click.option("--pairings", type=Path, required=True, help="merged pairings parquet (lat, lon, pop, ERA5 baseline)")
@click.option("--dem-dir", type=Path, required=True)
@click.option("--out-dir", type=Path, required=True)
@click.option("--fig-dir", type=Path, default=None)
@click.option("--max-era5-delta-min", type=float, default=60.0)
@click.option("--n-relief-bins", type=int, default=5)
@click.option("--n-boot", type=int, default=2000)
@click.option("--label", default="M_I8")
def main(eval_pairings, pairings, dem_dir, out_dir, fig_dir, max_era5_delta_min, n_relief_bins, n_boot, label):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    ev = pd.read_parquet(eval_pairings)
    ev["timestamp"] = pd.to_datetime(ev["timestamp_iso"])
    pa = pd.read_parquet(pairings, columns=["station_id", "timestamp", "lat", "lon", "elev", "pop", "season",
                                            "speed_era5_baseline", "era5_time_delta_minutes"])
    pa = pa.drop_duplicates(["station_id", "timestamp"])
    df = ev.merge(pa, on=["station_id", "timestamp"], how="left", validate="m:1")
    # all heights of one physical tower share terrain and must stay together in the bootstrap
    df["base_id"] = df.station_id.str.replace(r"_h\d+$", "", regex=True)
    if "speed_era5_10m" not in df:
        df["speed_era5_10m"] = df["speed_era5_baseline"]
    click.echo(f"rows={len(df)} stations={df.base_id.nunique()} unmatched={int(df.lat.isna().sum())}")

    st = df.groupby("base_id").agg(lat=("lat", "first"), lon=("lon", "first"), elev=("elev", "first"),
                                   pop=("pop", "first")).reset_index()
    rel_path = out_dir / "station_relief.csv"
    if rel_path.exists():
        rel = pd.read_csv(rel_path)
    else:
        rel = relief_indices(st.dropna(subset=["lat", "lon"]), dem_dir)
        rel.to_csv(rel_path, index=False)
    df = df.merge(rel, on="base_id", how="left")

    cols = {"era5": "speed_era5_10m", "raw": "speed_pred_raw", "corr": "speed_pred_corr"}
    d10 = df[(df.height_obs <= 12.0) & (df.era5_time_delta_minutes.abs() <= max_era5_delta_min)
             & df.relief_std_6km.notna() & df.speed_era5_10m.notna()].copy()
    click.echo(f"10 m rows kept for the ERA5 comparison: {len(d10)} ({d10.base_id.nunique()} stations)")

    def block(d: pd.DataFrame, key: dict) -> dict:
        row = {**key, **_mae_table(d, cols)}
        for m in ("raw", "corr"):
            s, lo, hi = _skill_ci(d, cols[m], cols["era5"], n_boot, rng)
            row.update({f"skill_{m}_vs_era5": s, f"skill_{m}_lo": lo, f"skill_{m}_hi": hi})
        return row

    # 1. per station
    per_st = pd.DataFrame([{**block(g, {"base_id": k}), "pop": g["pop"].iloc[0],
                            **{c: g[c].iloc[0] for c in ("relief_std_6km", "relief_std_cell", "dz_cell", "elev")}}
                           for k, g in d10.groupby("base_id")])
    for m in ("raw", "corr"):  # a single station has no bootstrap: plain ratio
        per_st[f"skill_{m}_vs_era5"] = 1.0 - per_st[f"mae_{m}"] / per_st["mae_era5"]
    per_st.to_csv(out_dir / "skill_per_station.csv", index=False)

    # 2. by relief bin (station-level quantiles so each bin holds the same number of stations)
    tables = {}
    for idx_name in ("relief_std_6km", "relief_std_cell"):
        q = per_st[idx_name].quantile(np.linspace(0, 1, n_relief_bins + 1)).to_numpy().copy()
        q[0] -= 1e-6
        d10[f"bin_{idx_name}"] = pd.cut(d10[idx_name], np.unique(q))
        tables[idx_name] = pd.DataFrame([block(g, {"bin": str(k)})
                                         for k, g in d10.groupby(f"bin_{idx_name}", observed=True)])
    d10["dz_class"] = pd.cut(d10.dz_cell, [-1e4, -100, 100, 300, 1e4],
                             labels=["below cell (<-100 m)", "near cell mean", "above (+100..300 m)", "well above (>300 m)"])
    tables["dz_cell"] = pd.DataFrame([block(g, {"bin": str(k)}) for k, g in d10.groupby("dz_class", observed=True)])
    tables["pop"] = pd.DataFrame([block(g, {"bin": str(k)}) for k, g in d10.groupby("pop")])

    # 3. relief tercile x observed wind class
    t = per_st.relief_std_6km.quantile([1 / 3, 2 / 3]).to_numpy()
    d10["relief_tercile"] = pd.cut(d10.relief_std_6km, [-1, t[0], t[1], 1e5], labels=["low", "mid", "high"])
    rows = []
    for (rt, (lo, hi)) in [(a, b) for a in ["low", "mid", "high"] for b in WIND_BINS]:
        g = d10[(d10.relief_tercile == rt) & (d10.speed_obs >= lo) & (d10.speed_obs < hi)]
        if len(g) >= 100:
            rows.append(block(g, {"relief": rt, "obs_class": f"{lo}-{hi}"}))
    tables["relief_x_wind"] = pd.DataFrame(rows)

    # 4. all heights: corrected vs raw (ERA5 10 m is not a fair competitor aloft)
    dh = df[df.relief_std_6km.notna()].copy()
    dh["h_class"] = pd.cut(dh.height_obs, [0, 12, 35, 65, 105, 250], labels=["10", "20-30", "40-60", "80-100", ">100"])
    dh["relief_tercile"] = pd.cut(dh.relief_std_6km, [-1, t[0], t[1], 1e5], labels=["low", "mid", "high"])
    tables["height_x_relief"] = pd.DataFrame([
        {"h_class": str(h), "relief": str(rt), **_mae_table(g, {"raw": cols["raw"], "corr": cols["corr"]})}
        for (h, rt), g in dh.groupby(["h_class", "relief_tercile"], observed=True) if len(g) >= 100])

    md = [f"# Skill map — {label}\n",
          f"10 m rows, |ERA5 time offset| <= {max_era5_delta_min:.0f} min: {len(d10)} rows, "
          f"{d10.base_id.nunique()} val stations. Skill = 1 - MAE/MAE_ERA5 (>0 = better than ERA5), "
          "90 % interval by bootstrap over stations.\n"]
    for name, tb in tables.items():
        tb.to_csv(out_dir / f"skill_by_{name}.csv", index=False)
        md.append(f"\n## By {name}\n\n```\n{tb.round(3).to_string(index=False)}\n```\n")
    (out_dir / "REPORT_skill_map.md").write_text("".join(md))
    click.echo("".join(md))

    if fig_dir is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        for a, m in zip(ax, ("raw", "corr")):
            a.axhline(0, color="k", lw=0.8)
            a.scatter(per_st.relief_std_6km, per_st[f"skill_{m}_vs_era5"], s=np.sqrt(per_st.n_rows), alpha=0.5,
                      c=(per_st["pop"] == "steep").map({True: "C3", False: "C0"}))
            tb = tables["relief_std_6km"]
            mid = [d10.loc[d10["bin_relief_std_6km"].astype(str) == b, "relief_std_6km"].median() for b in tb.bin]
            a.errorbar(mid, tb[f"skill_{m}_vs_era5"], yerr=[tb[f"skill_{m}_vs_era5"] - tb[f"skill_{m}_lo"],
                       tb[f"skill_{m}_hi"] - tb[f"skill_{m}_vs_era5"]], fmt="o-", color="k", capsize=3)
            a.set_xscale("log"); a.set_ylim(-1.5, 0.8)
            a.set_xlabel("relief: std of elevation in the 6 km patch (m)")
            a.set_title(f"{label} {m} vs ERA5 10 m")
        ax[0].set_ylabel("skill = 1 - MAE / MAE_ERA5")
        fig.tight_layout(); fig.savefig(fig_dir / f"skill_vs_relief_{label}.png", dpi=160)


if __name__ == "__main__":
    main()
