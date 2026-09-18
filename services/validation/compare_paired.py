"""Paired comparison of two models evaluated on the SAME val pairings.

Joins two eval_pairings.parquet files (eval_devine_style.py) row by row on
(station_id, timestamp_iso, height_obs) and reports, per stratum, the difference of MAE
  delta = MAE_B - MAE_A   (< 0 : B better)
with a 90 % interval from a bootstrap over STATIONS of the per-station error sums, i.e.
the two models are compared on the same hours and the same stations.  Strata: relief
quintile (from skill_map's station_relief.csv), observed wind class, height class, pop.
"""
from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import pandas as pd

WIND_BINS = [(0, 3), (3, 6), (6, 10), (10, 60)]
KEY = ["station_id", "timestamp_iso", "height_obs"]


def _paired_delta(d: pd.DataFrame, ea: str, eb: str, n_boot: int, rng) -> dict:
    g = d.groupby("base_id")[[ea, eb]].sum()
    n = d.groupby("base_id").size().reindex(g.index).to_numpy()
    a, b = g[ea].to_numpy(), g[eb].to_numpy()
    out = {"n_rows": int(len(d)), "n_stations": int(len(g)),
           "mae_A": float(a.sum() / n.sum()), "mae_B": float(b.sum() / n.sum())}
    out["delta_B_minus_A"] = out["mae_B"] - out["mae_A"]
    out["rel_pct"] = 100.0 * out["delta_B_minus_A"] / out["mae_A"]
    if len(g) >= 3:
        idx = rng.integers(0, len(g), size=(n_boot, len(g)))
        boot = (b[idx].sum(1) - a[idx].sum(1)) / n[idx].sum(1)
        out["lo90"], out["hi90"] = float(np.quantile(boot, 0.05)), float(np.quantile(boot, 0.95))
    return out


@click.command()
@click.option("--eval-a", type=Path, required=True, help="reference eval_pairings.parquet (e.g. M_I8)")
@click.option("--eval-b", type=Path, required=True, help="challenger eval_pairings.parquet (e.g. ablation A)")
@click.option("--col", default="speed_pred_corr", show_default=True, help="prediction column compared in both")
@click.option("--station-relief", type=Path, default=None, help="station_relief.csv from skill_map.py")
@click.option("--pairings", type=Path, default=None, help="merged pairings parquet for pop / era5 offset")
@click.option("--label-a", default="A")
@click.option("--label-b", default="B")
@click.option("--n-boot", type=int, default=2000)
@click.option("--out-dir", type=Path, required=True)
def main(eval_a, eval_b, col, station_relief, pairings, label_a, label_b, n_boot, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    a = pd.read_parquet(eval_a)[KEY + ["speed_obs", col]].rename(columns={col: "pa"})
    b = pd.read_parquet(eval_b)[KEY + [col]].rename(columns={col: "pb"})
    # a few (station, hour) pairs occur twice at season boundaries: keep the first
    n_dup = int(a.duplicated(KEY).sum())
    a = a.drop_duplicates(KEY, keep="first")
    b = b.drop_duplicates(KEY, keep="first")
    d = a.merge(b, on=KEY, how="inner", validate="1:1")
    click.echo(f"paired rows: {len(d)}  (A={len(a)}, B={len(b)}, duplicate keys dropped in A: {n_dup})")
    d["base_id"] = d.station_id.str.replace(r"_h\d+$", "", regex=True)
    d["ea"] = (d.pa - d.speed_obs).abs()
    d["eb"] = (d.pb - d.speed_obs).abs()
    if station_relief is not None:
        d = d.merge(pd.read_csv(station_relief), on="base_id", how="left")
    if pairings is not None:
        pa = pd.read_parquet(pairings, columns=["station_id", "timestamp", "pop"]).drop_duplicates(["station_id", "timestamp"])
        pa["timestamp_iso"] = pa.timestamp.dt.strftime("%Y-%m-%dT%H:%M:%S")
        d = d.merge(pa[["station_id", "timestamp_iso", "pop"]], on=["station_id", "timestamp_iso"], how="left")

    blocks = {"all": [{"stratum": "all", **_paired_delta(d, "ea", "eb", n_boot, rng)}]}
    d["h_class"] = pd.cut(d.height_obs, [0, 12, 35, 65, 105, 250], labels=["10", "20-30", "40-60", "80-100", ">100"])
    blocks["height"] = [{"stratum": str(k), **_paired_delta(g, "ea", "eb", n_boot, rng)}
                        for k, g in d.groupby("h_class", observed=True)]
    d10 = d[d.height_obs <= 12.0]
    blocks["wind_10m"] = [{"stratum": f"{lo}-{hi}", **_paired_delta(d10[(d10.speed_obs >= lo) & (d10.speed_obs < hi)], "ea", "eb", n_boot, rng)}
                          for lo, hi in WIND_BINS]
    if "relief_std_6km" in d:
        st = d10.groupby("base_id").relief_std_6km.first()
        q = np.unique(st.quantile(np.linspace(0, 1, 6)).to_numpy().copy()); q[0] -= 1e-6
        d10 = d10.assign(rbin=pd.cut(d10.relief_std_6km, q))
        blocks["relief_10m"] = [{"stratum": str(k), **_paired_delta(g, "ea", "eb", n_boot, rng)}
                                for k, g in d10.groupby("rbin", observed=True)]
        t = st.quantile([1 / 3, 2 / 3]).to_numpy()
        d10 = d10.assign(rt=pd.cut(d10.relief_std_6km, [-1, t[0], t[1], 1e5], labels=["low", "mid", "high"]))
        blocks["relief_x_wind_10m"] = [
            {"stratum": f"{rt} {lo}-{hi}", **_paired_delta(g, "ea", "eb", n_boot, rng)}
            for rt in ["low", "mid", "high"] for lo, hi in WIND_BINS
            for g in [d10[(d10.rt == rt) & (d10.speed_obs >= lo) & (d10.speed_obs < hi)]] if len(g) >= 100]
        if "dz_cell" in d:
            blocks["dz_cell_10m"] = [{"stratum": str(k), **_paired_delta(g, "ea", "eb", n_boot, rng)}
                                     for k, g in d10.groupby(pd.cut(d10.dz_cell, [-1e4, -100, 100, 300, 1e4]), observed=True)]
    if "pop" in d:
        blocks["pop"] = [{"stratum": str(k), **_paired_delta(g, "ea", "eb", n_boot, rng)} for k, g in d.groupby("pop")]

    md = [f"# Paired comparison — B = {label_b} vs A = {label_a} (column {col})\n\n",
          "delta = MAE_B - MAE_A on identical rows; 90 % interval by bootstrap over stations. delta < 0: B better.\n"]
    for name, rows in blocks.items():
        tb = pd.DataFrame(rows)
        tb.to_csv(out_dir / f"paired_{name}.csv", index=False)
        md.append(f"\n## {name}\n\n```\n{tb.round(3).to_string(index=False)}\n```\n")
    (out_dir / "REPORT_paired.md").write_text("".join(md))
    click.echo("".join(md))


if __name__ == "__main__":
    main()
