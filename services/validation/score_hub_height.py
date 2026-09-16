"""Score hub-height wind products against turbine/mast observations at one site.

Products (each optional, scored only if its file exists):
  ours_raw / ours_corr   : predictions.parquet from run_hub_height_inference.py
  era5_10m_patch         : ERA5 10 m baseline carried in the same predictions file
  era5_100m              : ERA5 100 m wind (surface-only zarr, u100/v100), log-interp to hub
  newa_meso              : NEWA 3 km hourly series (parquet from data/raw/newa_ts), log-interp to hub
  gwa                    : Global Wind Atlas mean speed (climatology only → mean-bias line)

Metrics per product: n, bias, MAE, RMSE, corr, slope of pred~obs; by wind class (<4, 4-8, 8-12,
>12 m/s of the observation). For SCADA sites an empirical site power curve (binned
density-adjusted nacelle speed → power) converts every product's speed into power, giving a
monthly capacity-factor comparison that does not depend on the nacelle anemometer bias.
Writes <out_dir>/metrics_{products,windclass,monthly_cf}.csv + REPORT.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd

WIND_BINS = [(0, 4), (4, 8), (8, 12), (12, 60)]


def _metrics(obs: pd.Series, pred: pd.Series) -> dict:
    m = obs.notna() & pred.notna()
    o, p = obs[m].to_numpy(), pred[m].to_numpy()
    if len(o) < 10:
        return {"n": int(len(o))}
    e = p - o
    slope = np.polyfit(o, p, 1)[0]
    return {"n": int(len(o)), "obs_mean": float(o.mean()), "pred_mean": float(p.mean()),
            "bias": float(e.mean()), "mae": float(np.abs(e).mean()),
            "rmse": float(np.sqrt((e ** 2).mean())), "corr": float(np.corrcoef(o, p)[0, 1]),
            "slope": float(slope)}


def _log_interp(w_lo: pd.Series, w_hi: pd.Series, z_lo: float, z_hi: float, z: float) -> pd.Series:
    return w_lo + (w_hi - w_lo) * (np.log(z / z_lo) / np.log(z_hi / z_lo))


def _era5_100m_at(store: Path, lat: float, lon: float) -> pd.Series | None:
    if not store.exists():
        return None
    import zarr
    g = zarr.open_group(str(store), mode="r")
    lats, lons = g["coords/lat"][:], g["coords/lon"][:]
    i, j = int(np.argmin(np.abs(lats - lat))), int(np.argmin(np.abs(lons - lon)))
    t = pd.to_datetime(g["coords/time"][:].astype("int64"))
    u = g["surface/u100"][:, i, j]; v = g["surface/v100"][:, i, j]
    return pd.Series(np.sqrt(u ** 2 + v ** 2), index=t, name="era5_100m")


def _site_power_curve(sc: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Binned density-adjusted nacelle speed → mean power (kW), 0.5 m/s bins."""
    d = sc.dropna(subset=["ws_density_adj", "power_kw"])
    d = d[(d.power_kw >= 0)]
    bins = np.arange(0, 30.5, 0.5)
    cut = pd.cut(d.ws_density_adj, bins)
    curve = d.groupby(cut, observed=False).power_kw.mean()
    centers = np.array([b.mid for b in curve.index])
    return centers, curve.to_numpy()


@click.command()
@click.option("--pairings", type=Path, required=True, help="obs parquet (speed_obs, power_kw…)")
@click.option("--predictions", type=Path, default=None, help="predictions.parquet (optional)")
@click.option("--newa", type=Path, default=None, help="NEWA parquet (timestamp,height_m,ws_newa)")
@click.option("--era5-u100", type=Path, default=None, help="surface-only zarr with u100/v100")
@click.option("--gwa-json", type=Path, default=None)
@click.option("--gwa-site", default=None)
@click.option("--hub-height", type=float, required=True)
@click.option("--rated-kw", type=float, default=None, help="for capacity factor (SCADA sites)")
@click.option("--out-dir", type=Path, required=True)
def main(pairings, predictions, newa, era5_u100, gwa_json, gwa_site, hub_height, rated_kw, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    sc = pd.read_parquet(pairings)
    sc["timestamp"] = pd.to_datetime(sc["timestamp"])
    lat, lon = float(sc.lat.mean()), float(sc.lon.mean())
    # site-mean hourly observation (turbines/masts averaged → one series)
    agg = {"speed_obs": "mean"}
    if "power_kw" in sc:
        agg["power_kw"] = "mean"
    site = sc.groupby("timestamp").agg(agg)
    site["n_units"] = sc.groupby("timestamp").size()
    site = site[site.n_units >= max(1, int(0.7 * sc.station_id.nunique()))]

    prods: dict[str, pd.Series] = {}
    if predictions and predictions.exists():
        pr = pd.read_parquet(predictions)
        pr["timestamp"] = pd.to_datetime(pr["timestamp_iso"])
        g = pr.groupby("timestamp").agg(ours_raw=("speed_raw", "mean"),
                                        ours_corr=("speed_corr", "mean"),
                                        era5_10m_patch=("speed_era5_baseline_patch", "mean"))
        for c in g.columns:
            prods[c] = g[c]
    if newa and newa.exists():
        nw = pd.read_parquet(newa)
        w = nw.pivot(index="timestamp", columns="height_m", values="ws_newa")
        hs = sorted(w.columns)
        lo = max([h for h in hs if h <= hub_height], default=hs[0])
        hi = min([h for h in hs if h >= hub_height], default=hs[-1])
        prods["newa_meso"] = w[lo] if lo == hi else _log_interp(w[lo], w[hi], lo, hi, hub_height)
    if era5_u100:
        s = _era5_100m_at(era5_u100, lat, lon)
        if s is not None:
            # 100 m → hub via log law with z0=0.1 m (open hilly land) — documented approximation
            prods["era5_100m_loglaw"] = s * (np.log(hub_height / 0.1) / np.log(100.0 / 0.1))

    df = site.join(pd.DataFrame(prods), how="left")
    rows, rows_wc = [], []
    for name, ser in prods.items():
        rows.append({"product": name, **_metrics(df.speed_obs, df[name])})
        for lo_, hi_ in WIND_BINS:
            m = (df.speed_obs >= lo_) & (df.speed_obs < hi_)
            rows_wc.append({"product": name, "obs_class": f"{lo_}-{hi_}",
                            **_metrics(df.speed_obs[m], df[name][m])})
    if gwa_json and gwa_site:
        gwa = json.loads(Path(gwa_json).read_text())[gwa_site]
        hs = sorted((float(k.rstrip("m")), v) for k, v in gwa.items())
        lo = max([h for h in hs if h[0] <= hub_height], default=hs[0])
        hi = min([h for h in hs if h[0] >= hub_height], default=hs[-1])
        g_hub = lo[1] if lo == hi else float(_log_interp(pd.Series([lo[1]]), pd.Series([hi[1]]),
                                                          lo[0], hi[0], hub_height)[0])
        rows.append({"product": "gwa_climatology", "n": int(df.speed_obs.notna().sum()),
                     "obs_mean": float(df.speed_obs.mean()), "pred_mean": g_hub,
                     "bias": g_hub - float(df.speed_obs.mean())})
    met = pd.DataFrame(rows); met.to_csv(out_dir / "metrics_products.csv", index=False)
    pd.DataFrame(rows_wc).to_csv(out_dir / "metrics_windclass.csv", index=False)

    cf_txt = ""
    if "power_kw" in site and rated_kw:
        centers, curve = _site_power_curve(sc)
        ok = ~np.isnan(curve)
        def to_power(ws: pd.Series) -> pd.Series:
            return pd.Series(np.interp(ws.to_numpy(dtype=float), centers[ok], curve[ok],
                                       left=0.0, right=float(curve[ok][-1])), index=ws.index)
        cf = pd.DataFrame({"obs": df.power_kw / rated_kw})
        for name in prods:
            cf[name] = to_power(df[name]) / rated_kw
        monthly = cf.groupby(cf.index.to_period("M")).mean()
        monthly.to_csv(out_dir / "metrics_monthly_cf.csv")
        annual = cf.mean()
        cf_txt = "\n## Capacity factor (site power curve applied to each product's speed)\n\n" + \
                 "annual: " + ", ".join(f"{k}={v:.3f}" for k, v in annual.items()) + "\n\n" + \
                 monthly.round(3).to_string() + "\n"

    report = f"# Hub-height benchmark — {pairings.name}\n\nsite lat={lat:.4f} lon={lon:.4f} hub={hub_height} m, " \
             f"hours scored={len(df)}\n\n## Products\n\n{met.round(3).to_string(index=False)}\n\n" \
             f"## By observed wind class\n\n{pd.DataFrame(rows_wc).round(3).to_string(index=False)}\n{cf_txt}"
    (out_dir / "REPORT.md").write_text(report)
    click.echo(report)


if __name__ == "__main__":
    main()
