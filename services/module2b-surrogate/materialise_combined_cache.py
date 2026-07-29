"""
materialise_combined_cache.py — M_I3 cache pre-materialisation (the gatekeeper).

Pre-builds every grid.zarr that `train_v2_devine_style.py` will need for the
combined steep+plain dataset, INTO the shared training cache_dir, using the
CORRECT per-season ERA5 store. This is required because the loader accepts only
ONE `era5_store` but the 4 steep seasons live in disjoint temporal windows
(mam/jja/son/winter) → no single store covers them all.

Once this cache is fully populated, the two training runs use
`overwrite_cache=false`: every __getitem__ hits the cache, build_one is never
called, and the placeholder `era5_store` in the training YAML is irrelevant.

Reuses the already-validated `parallel_materialise` from infer_at_stations.py
(ProcessPool, per-pairing grid.zarr, drops failures). The cache path convention
(`<station_id>_<ts_tag>/grid.zarr`, ts_tag = ts.replace(':','').replace('-','')[:13])
is IDENTICAL between infer_at_stations.materialise_grid_zarr and
ObsCenteredDataset._cache_path, so the cache is reused verbatim by training.

Designed to run as a PBS array (one task per season) so the 4 seasons build in
parallel on 4 CPU nodes.

Usage (per season):
    python materialise_combined_cache.py \
        --combined-parquet ~/dsw/data/inference/combined_steep_plain_v2.parquet \
        --season jja2023 \
        --era5-store ~/dsw/data/raw/era5_europe_hourly_jja2023_med.zarr \
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

import numpy as np
import pandas as pd

_SCRIPT = Path(__file__).resolve().parent
if str(_SCRIPT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT))

from infer_at_stations import parallel_materialise  # noqa: E402

logger = logging.getLogger("materialise_combined")

# Season → ERA5 store basename, for an optional auto-resolve / sanity hint.
SEASON_STORES = {
    "mam2023": "era5_europe_hourly_mam2023_med.zarr",
    "jja2023": "era5_europe_hourly_jja2023_med.zarr",
    "son2023": "era5_europe_hourly_son2023_med.zarr",
    "winter2223": "era5_europe_winter2223.zarr",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined-parquet", type=Path, required=True)
    ap.add_argument("--season", type=str, required=True,
                    choices=list(SEASON_STORES))
    ap.add_argument("--era5-store", type=Path, required=True)
    ap.add_argument("--dem", type=Path, required=True)
    ap.add_argument("--worldcover", type=Path, default=None)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--n-workers", type=int, default=16)
    ap.add_argument("--chunk-size", type=int, default=4000,
                    help="rows per parallel_materialise call (progress + memory)")
    ap.add_argument("--max-era5-delta-h", type=float, default=3.5)
    ap.add_argument("--max-rows", type=int, default=None,
                    help="debug cap on number of rows")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    df = pd.read_parquet(args.combined_parquet)
    if "season" not in df.columns:
        raise SystemExit("combined parquet has no 'season' column "
                         "(rebuild with build_combined_steep_plain_parquet.py)")
    df = df[df["season"] == args.season].reset_index(drop=True)
    if args.max_rows is not None:
        df = df.head(args.max_rows).reset_index(drop=True)
    logger.info("season=%s rows=%d era5_store=%s", args.season, len(df),
                args.era5_store)

    # parallel_materialise needs a 'timestamp_ns' int column.
    df["timestamp_ns"] = pd.to_datetime(df["timestamp"]).astype("int64")

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # Skip rows already cached (idempotent re-runs).
    def _cached(row) -> bool:
        ts_iso = str(np.array(int(row.timestamp_ns)).astype("datetime64[ns]"))
        tag = ts_iso.replace(":", "").replace("-", "")[:13]
        return (args.cache_dir / f"{row.station_id}_{tag}" / "grid.zarr").exists()

    n_total = len(df)
    n_built = 0
    n_skipped = 0
    n_fail = 0
    t0 = time.time()
    for start in range(0, n_total, args.chunk_size):
        chunk = df.iloc[start:start + args.chunk_size]
        todo = chunk[~chunk.apply(_cached, axis=1)]
        n_skipped += len(chunk) - len(todo)
        if len(todo) == 0:
            continue
        todo = todo.reset_index(drop=True)
        res = parallel_materialise(
            todo,
            era5_store=args.era5_store,
            dem=args.dem,
            worldcover=args.worldcover,
            workdir=args.cache_dir,
            max_era5_delta_h=args.max_era5_delta_h,
            n_workers=args.n_workers,
        )
        n_built += len(res)
        n_fail += len(todo) - len(res)
        done = start + len(chunk)
        rate = n_built / max(1e-6, time.time() - t0)
        logger.info("[%s] %d/%d built=%d skip=%d fail=%d (%.1f built/s)",
                    args.season, done, n_total, n_built, n_skipped, n_fail, rate)

    logger.info("DONE season=%s built=%d skipped=%d failed=%d of %d in %.1f min",
                args.season, n_built, n_skipped, n_fail, n_total,
                (time.time() - t0) / 60.0)


if __name__ == "__main__":
    main()
