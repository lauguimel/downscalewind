"""
build_combined_steep_plain_parquet.py — M_I3 combined dataset assembler.

Merges the 4 steep-season pairing parquets (Alps+Apennines steep stations) with
the 4-season plain/prod pairing parquet into ONE pairings parquet consumed by
`train_v2_devine_style.py` / `ObsCenteredDataset`.

Adds a `season` column (mam/jja/son/winter2223) derived from the source file so
the per-season cache materialisation step can route each pairing to the correct
ERA5 store. The training loader itself ignores `season` (it only needs
station_id, timestamp, lat, lon, elev, height_obs, speed_obs) — the column is a
convenience for the materialisation PBS.

Steep and plain station sets are disjoint (verified: steep∩prod = 0), so the
watertight station split in the loader stays leak-free across the union.

Usage (on Aqua, fuxicfd):
    python build_combined_steep_plain_parquet.py \
        --inference-dir ~/dsw/data/inference \
        --output ~/dsw/data/inference/combined_steep_plain_v2.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# (parquet basename, season tag)  →  season tag picks the ERA5 store downstream
STEEP_FILES = [
    ("steep_mam2023_v2.parquet", "mam2023"),
    ("steep_jja2023_v2.parquet", "jja2023"),
    ("steep_son2023_v2.parquet", "son2023"),
    ("steep_winter2223_v2.parquet", "winter2223"),
]
# Plain/prod is already all-seasons concatenated; tag rows by month.
PLAIN_FILE = ("noaa_seasons_all_v2.parquet", None)

REQUIRED = ["station_id", "timestamp", "source", "lat", "lon", "elev",
            "height_obs", "speed_obs"]


def _season_from_timestamp(ts: pd.Series) -> pd.Series:
    t = pd.to_datetime(ts)
    m = t.dt.month
    out = pd.Series("winter2223", index=t.index, dtype=object)
    out[m.isin([3, 4, 5])] = "mam2023"
    out[m.isin([6, 7, 8])] = "jja2023"
    out[m.isin([9, 10, 11])] = "son2023"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inference-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    frames: list[pd.DataFrame] = []
    for fname, season in STEEP_FILES:
        df = pd.read_parquet(args.inference_dir / fname)
        df["season"] = season
        df["pop"] = "steep"
        frames.append(df)
        print(f"steep {season}: rows={len(df)} stations={df.station_id.nunique()}")

    pf, _ = PLAIN_FILE
    pdf = pd.read_parquet(args.inference_dir / pf)
    pdf["season"] = _season_from_timestamp(pdf["timestamp"])
    pdf["pop"] = "plain"
    frames.append(pdf)
    print(f"plain (all seasons): rows={len(pdf)} stations={pdf.station_id.nunique()}")

    combined = pd.concat(frames, ignore_index=True)

    missing = [c for c in REQUIRED if c not in combined.columns]
    if missing:
        raise SystemExit(f"combined parquet missing required cols: {missing}")

    # Drop rows the loader would drop anyway (keeps cache pre-materialisation
    # aligned with what training will request).
    n0 = len(combined)
    combined = combined.dropna(subset=["speed_obs", "lat", "lon", "height_obs"])
    combined = combined[combined["speed_obs"] > 0.0].reset_index(drop=True)
    print(f"after drop NaN/zero-speed: {len(combined)}/{n0}")

    steep_sids = set(combined.loc[combined["pop"] == "steep", "station_id"])
    plain_sids = set(combined.loc[combined["pop"] == "plain", "station_id"])
    print(f"TOTAL rows={len(combined)} stations={combined.station_id.nunique()} "
          f"(steep={len(steep_sids)} plain={len(plain_sids)} "
          f"overlap={len(steep_sids & plain_sids)})")
    print("per-season counts:")
    print(combined.groupby(["pop", "season"]).size())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.output, index=False)
    print(f"WROTE {args.output} ({args.output.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
