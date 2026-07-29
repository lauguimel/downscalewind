"""
rematerialise_lon_neg_diag.py — DIAGNOSTIC re-cache of lon<0 grid.zarr.

After the `_resolve_dem_path` lon<0 fix (inference_input.py), the grid.zarr
caches for west-hemisphere stations contain a flat-ZERO terrain artefact and
must be rebuilt with the corrected terrain. This driver re-materialises a
station-filtered subset of `combined_steep_plain_v2.parquet`, FORCING overwrite
(it calls `parallel_materialise` directly — which always rmtrees the target —
with NO skip-cached logic), routing the per-season ERA5 store automatically.

DIAGNOSTIC SCOPE ONLY: pass `--stations-file` with the 23 held-out VAL lon<0
stations so we re-score the CURRENT model on TRUE terrain. The 99 train lon<0
stations are intentionally NOT re-cached here (that is for a later re-train).

Usage (one season per PBS task, or all seasons serially):
    python rematerialise_lon_neg_diag.py \
        --combined-parquet ~/dsw/data/inference/combined_steep_plain_v2.parquet \
        --stations-file ~/dsw/tmp/val_lon_neg_stations.txt \
        --era5-root ~/dsw/data/raw \
        --dem ~/dsw/data/raw/srtm_tiles/ \
        --worldcover ~/dsw/data/raw/worldcover_esa/ \
        --cache-dir /scratch/maitreje/dsw/phase_H_prime_M_I3_combined_cache \
        --n-workers 16
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

_SCRIPT = Path(__file__).resolve().parent
if str(_SCRIPT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT))

from infer_at_stations import parallel_materialise  # noqa: E402

logger = logging.getLogger("rematerialise_lon_neg")

# Season → ERA5 store basename (MAM/JJA/SON use the Mediterranean store;
# winter 2022-23 uses the dedicated DJF store). Matches materialise_combined_cache.
SEASON_STORES = {
    "mam2023": "era5_europe_hourly_mam2023_med.zarr",
    "jja2023": "era5_europe_hourly_jja2023_med.zarr",
    "son2023": "era5_europe_hourly_son2023_med.zarr",
    "winter2223": "era5_europe_winter2223.zarr",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined-parquet", type=Path, required=True)
    ap.add_argument("--stations-file", type=Path, required=True,
                    help="one station_id per line (the lon<0 subset to rebuild)")
    ap.add_argument("--era5-root", type=Path, required=True,
                    help="dir holding the per-season ERA5 stores")
    ap.add_argument("--dem", type=Path, required=True)
    ap.add_argument("--worldcover", type=Path, default=None)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--seasons", type=str, default=None,
                    help="comma list to restrict; default = all 4")
    ap.add_argument("--n-workers", type=int, default=16)
    ap.add_argument("--chunk-size", type=int, default=2000)
    ap.add_argument("--max-era5-delta-h", type=float, default=3.5)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    stations = [s.strip() for s in args.stations_file.read_text().splitlines()
                if s.strip()]
    sset = set(stations)
    logger.info("re-cache %d target stations from %s", len(stations),
                args.stations_file)

    df = pd.read_parquet(args.combined_parquet)
    if "season" not in df.columns:
        raise SystemExit("combined parquet has no 'season' column")
    df = df[df["station_id"].isin(sset)].reset_index(drop=True)
    df["timestamp_ns"] = pd.to_datetime(df["timestamp"]).astype("int64")
    logger.info("filtered to %d pairings across %d stations",
                len(df), df["station_id"].nunique())

    seasons = (args.seasons.split(",") if args.seasons
               else list(SEASON_STORES))
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    grand_built = grand_fail = 0
    t0 = time.time()
    for season in seasons:
        store = args.era5_root / SEASON_STORES[season]
        sub = df[df["season"] == season].reset_index(drop=True)
        if len(sub) == 0:
            logger.info("season=%s: 0 pairings — skip", season)
            continue
        if not store.exists():
            raise SystemExit(f"ERA5 store missing for {season}: {store}")
        logger.info("season=%s rows=%d store=%s", season, len(sub), store.name)
        built = fail = 0
        for start in range(0, len(sub), args.chunk_size):
            chunk = sub.iloc[start:start + args.chunk_size].reset_index(drop=True)
            res = parallel_materialise(
                chunk,
                era5_store=store,
                dem=args.dem,
                worldcover=args.worldcover,
                workdir=args.cache_dir,
                max_era5_delta_h=args.max_era5_delta_h,
                n_workers=args.n_workers,
            )
            built += len(res)
            fail += len(chunk) - len(res)
            logger.info("[%s] %d/%d built=%d fail=%d (%.1f s)",
                        season, start + len(chunk), len(sub), built, fail,
                        time.time() - t0)
        grand_built += built
        grand_fail += fail
        logger.info("DONE season=%s built=%d fail=%d", season, built, fail)

    logger.info("ALL DONE built=%d fail=%d in %.1f min",
                grand_built, grand_fail, (time.time() - t0) / 60.0)


if __name__ == "__main__":
    main()
