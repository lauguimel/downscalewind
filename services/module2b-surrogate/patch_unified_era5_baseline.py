"""Fill missing ERA5 baseline speeds in a unified_scores.parquet from a Zarr store.

score_fuxi_vs_ours.py leaves speed_era5 = NaN for populations whose pairings do
not come from an inference parquet (Perdigao, ICOS EU towers): the baseline is
only copied from pre-computed speed_era5_baseline columns. This tool computes it
directly from an ERA5 store: bilinear interpolation of surface u10/v10 at the
station coordinates, nearest timestamp within --max-delta-h. It then rewrites
unified_scores.parquet in place and regenerates score_report.json via
score_fuxi_vs_ours.build_table.

Usage (Aqua, env fuxicfd):
  python patch_unified_era5_baseline.py \
    --out-dir ~/dsw/data/validation/fuxi_vs_ours \
    --pop perdigao --store ~/dsw/data/raw/era5_europe_spring2017_v2.zarr \
    --max-delta-h 3.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from score_fuxi_vs_ours import build_table  # noqa: E402


def _bilinear(field: np.ndarray, lats: np.ndarray, lons: np.ndarray,
              lat: float, lon: float) -> float:
    """Bilinear interp on a small regular grid; lats may be descending."""
    order = np.argsort(lats)
    la = lats[order]
    fi = field[..., order, :]
    i = int(np.clip(np.searchsorted(la, lat) - 1, 0, len(la) - 2))
    j = int(np.clip(np.searchsorted(lons, lon) - 1, 0, len(lons) - 2))
    ty = (lat - la[i]) / (la[i + 1] - la[i])
    tx = (lon - lons[j]) / (lons[j + 1] - lons[j])
    ty = float(np.clip(ty, 0.0, 1.0)); tx = float(np.clip(tx, 0.0, 1.0))
    f00, f01 = fi[i, j], fi[i, j + 1]
    f10, f11 = fi[i + 1, j], fi[i + 1, j + 1]
    return float((1 - ty) * ((1 - tx) * f00 + tx * f01)
                 + ty * ((1 - tx) * f10 + tx * f11))


def era5_speed10_at(store: Path, rows: pd.DataFrame, max_delta_h: float,
                    ) -> tuple[pd.Series, pd.Series]:
    g = zarr.open_group(str(store), mode="r")
    times = np.array(g["coords/time"]).astype("datetime64[ns]")
    lats = np.array(g["coords/lat"], dtype=np.float64)
    lons = np.array(g["coords/lon"], dtype=np.float64)
    u10 = np.array(g["surface/u10"])
    v10 = np.array(g["surface/v10"])
    speeds, deltas = [], []
    for _, r in rows.iterrows():
        ts = np.datetime64(pd.Timestamp(r["timestamp_iso"]).to_datetime64(), "ns")
        k = int(np.argmin(np.abs(times - ts)))
        dh = abs((times[k] - ts) / np.timedelta64(1, "h"))
        if dh > max_delta_h:
            speeds.append(np.nan); deltas.append(float(dh))
            continue
        u = _bilinear(u10[k], lats, lons, float(r["lat"]), float(r["lon"]))
        v = _bilinear(v10[k], lats, lons, float(r["lat"]), float(r["lon"]))
        speeds.append(float(np.hypot(u, v))); deltas.append(float(dh))
    return pd.Series(speeds, index=rows.index), pd.Series(deltas, index=rows.index)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Directory holding unified_scores.parquet + score_report.json")
    ap.add_argument("--pop", required=True, help="pop value to patch (e.g. perdigao)")
    ap.add_argument("--store", type=Path, required=True, help="ERA5 Zarr with surface u10/v10")
    ap.add_argument("--max-delta-h", type=float, default=3.0)
    args = ap.parse_args()

    uni_path = args.out_dir / "unified_scores.parquet"
    uni = pd.read_parquet(uni_path)
    mask = (uni["pop"].astype(str) == args.pop) & uni["speed_era5"].isna()
    print(f"pop={args.pop}: {int(mask.sum())} rows with NaN speed_era5 "
          f"(of {int((uni['pop'].astype(str) == args.pop).sum())})")
    if mask.sum():
        sp, dh = era5_speed10_at(args.store, uni[mask], args.max_delta_h)
        uni.loc[mask, "speed_era5"] = sp
        print(f"filled {int(sp.notna().sum())} / {int(mask.sum())} "
              f"(time delta median {np.nanmedian(dh):.2f} h, max {np.nanmax(dh):.2f} h)")
        uni.to_parquet(uni_path, index=False)

    report_path = args.out_dir / "score_report.json"
    report = json.loads(report_path.read_text())
    common = uni[uni["speed_fuxi"].notna() & uni["speed_ours"].notna()
                 & uni["speed_obs"].notna()].copy()
    report["table_common_subset"] = build_table(common)
    report["table_full_coverage"] = build_table(uni)
    report["coverage"]["n_era5"] = int(uni["speed_era5"].notna().sum())
    report.setdefault("patches", []).append(
        {"pop": args.pop, "store": str(args.store), "max_delta_h": args.max_delta_h})
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["table_common_subset"].get(args.pop, {}), indent=2))
    print(f"rewrote {uni_path} + {report_path}")


if __name__ == "__main__":
    main()
