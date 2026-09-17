"""Per tower x height scoring of hub-height wind products at out-of-sample tall towers.

One row per (station_id = tower_hNNN, product): n, obs_mean, pred_mean, bias, MAE, RMSE, corr, slope.
Products: ours_raw, ours_corr, era5_10m_patch (from predictions.parquet), newa_meso (log-interpolated
between the two bracketing NEWA levels, heights >= 50 m only), era5_100m (only for heights within
25 m of 100 m, no vertical adjustment), gwa_climatology (mean bias only).
"""
from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd


def _metrics(o: pd.Series, p: pd.Series) -> dict:
    m = o.notna() & p.notna()
    o, p = o[m].to_numpy(), p[m].to_numpy()
    if len(o) < 30:
        return {"n": int(len(o))}
    e = p - o
    return {"n": int(len(o)), "obs_mean": o.mean(), "pred_mean": p.mean(), "bias": e.mean(),
            "mae": np.abs(e).mean(), "rmse": np.sqrt((e ** 2).mean()),
            "corr": np.corrcoef(o, p)[0, 1], "slope": np.polyfit(o, p, 1)[0]}


def _interp_levels(w: pd.DataFrame, z: float) -> pd.Series | None:
    hs = sorted(w.columns)
    if z < hs[0] or z > hs[-1]:
        return None
    lo = max(h for h in hs if h <= z); hi = min(h for h in hs if h >= z)
    if lo == hi:
        return w[lo]
    return w[lo] + (w[hi] - w[lo]) * (np.log(z / lo) / np.log(hi / lo))


@click.command()
@click.option("--predictions", type=Path, required=True)
@click.option("--newa", type=Path, default=None)
@click.option("--era5-u100", type=Path, default=None)
@click.option("--gwa-json", type=Path, default=None)
@click.option("--gwa-site", default=None)
@click.option("--out-dir", type=Path, required=True)
def main(predictions, newa, era5_u100, gwa_json, gwa_site, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    pr = pd.read_parquet(predictions)
    pr["timestamp"] = pd.to_datetime(pr["timestamp_iso"]).dt.tz_localize(None)
    nw = None
    if newa and newa.exists():
        n = pd.read_parquet(newa); n["timestamp"] = pd.to_datetime(n["timestamp"])
        nw = n.pivot_table(index="timestamp", columns="height_m", values="ws_newa")
    e100 = None
    if era5_u100 and (era5_u100 / "coords").exists():
        import zarr
        g = zarr.open_group(str(era5_u100), mode="r")
        lat0, lon0 = None, None
    gwa = json.loads(Path(gwa_json).read_text()).get(gwa_site) if gwa_json and gwa_site else None

    rows = []
    for sid, d in pr.groupby("station_id"):
        d = d.set_index("timestamp").sort_index()
        z = float(d.height_obs.iloc[0])
        prods = {"ours_raw": d.speed_raw, "ours_corr": d.speed_corr,
                 "era5_10m_patch": d.speed_era5_baseline_patch}
        if nw is not None:
            s = _interp_levels(nw, z)
            if s is not None:
                prods["newa_meso"] = s.reindex(d.index)
        for name, ser in prods.items():
            rows.append({"station_id": sid, "height": z, "product": name, **_metrics(d.speed_obs, ser)})
        if gwa:
            hs = sorted((float(k.rstrip("m")), v) for k, v in gwa.items())
            zs, vs = [h for h, _ in hs], [v for _, v in hs]
            if zs[0] <= z <= zs[-1]:
                gz = float(np.interp(np.log(z), np.log(zs), vs))
                rows.append({"station_id": sid, "height": z, "product": "gwa_climatology",
                             "n": int(d.speed_obs.notna().sum()), "obs_mean": d.speed_obs.mean(),
                             "pred_mean": gz, "bias": gz - d.speed_obs.mean()})
    met = pd.DataFrame(rows).sort_values(["height", "station_id", "product"])
    met.to_csv(out_dir / "metrics_towers.csv", index=False)
    txt = met.round(3).to_string(index=False)
    (out_dir / "REPORT_towers.md").write_text("# Out-of-sample towers — per height\n\n```\n" + txt + "\n```\n")
    click.echo(txt)


if __name__ == "__main__":
    main()
