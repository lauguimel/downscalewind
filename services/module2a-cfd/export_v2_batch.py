"""
export_v2_batch.py — Batch / sharded export of campaign v2 to native grid.zarr.

For each (gold|silver) case in dataset_v2_status.csv, calls export_to_grid_zarr_v2
to produce <output_dir>/<site_id>_<case_name>/grid.zarr.

Sharded mode: each PBS array job processes its slice of (gold ∪ silver) cases.

Usage
-----
    python export_v2_batch.py \\
        --status-csv /scratch/.../manifests/dataset_v2_status.csv \\
        --sites-csv  ~/dsw/data/campaign/complex_terrain_v1/sites.csv \\
        --run-matrix ~/dsw/data/campaign/complex_terrain_v1/run_matrix.csv \\
        --campaign-dir /scratch/maitreje/dsw/complex_terrain_v1/sites \\
        --era5-dir   /scratch/maitreje/dsw/era5_campaign_v3 \\
        --output-dir /scratch/maitreje/dsw/training_v2 \\
        --tiers gold silver \\
        --time 300 \\
        --shard 1 --n-shards 20 \\
        --max-cases 3        # smoke
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--status-csv", type=Path, required=True)
    ap.add_argument("--sites-csv", type=Path, required=True)
    ap.add_argument("--run-matrix", type=Path, required=True)
    ap.add_argument("--campaign-dir", type=Path, required=True,
                    help="Parent dir holding sites/<site_id>/case_tsNNN/")
    ap.add_argument("--era5-dir", type=Path, required=True,
                    help="Parent dir holding era5_<site_id>.zarr")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--tiers", nargs="+", default=["gold", "silver"],
                    choices=["gold", "silver"])
    ap.add_argument("--time", default=None,
                    help="Override OF time dir. Default: read solve_iter from status CSV per-case "
                         "(handles early_converged cases at iter ≠ 300).")
    ap.add_argument("--include-turb", nargs="*", default=[])
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--n-shards", type=int, default=None)
    ap.add_argument("--max-cases", type=int, default=None,
                    help="Process at most N cases (smoke test).")
    ap.add_argument("--cases", nargs="*", default=None,
                    help="Restrict to these specific 'site_id/case_tsNNN' (smoke).")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    # Make sibling export module importable
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import export_to_grid_zarr_v2 as exp  # noqa: E402

    # ── 1. Sites metadata (lat/lon) ────────────────────────────────────────
    sites: dict[str, dict] = {}
    with open(args.sites_csv) as f:
        for row in csv.DictReader(f):
            sites[row["site_id"]] = {
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "group": row.get("group", ""),
            }
    logger.info("Loaded %d sites from %s", len(sites), args.sites_csv)

    # ── 2. Run matrix (case_ts<idx> → timestamp) ───────────────────────────
    runs: dict[str, list[str]] = defaultdict(list)
    with open(args.run_matrix) as f:
        for row in csv.DictReader(f):
            runs[row["site_id"]].append(row["timestamp"])
    logger.info("Loaded run_matrix: %d sites with timestamps", len(runs))

    # ── 3. Status CSV → list of (site_id, case_name, solve_iter) ───────────
    cases: list[tuple[str, str, str]] = []
    with open(args.status_csv) as f:
        for row in csv.DictReader(f):
            if row["tier"] not in args.tiers:
                continue
            cases.append((row["site_id"], row["case_name"], row.get("solve_iter") or "300"))
    cases.sort()
    logger.info("Total cases to export (tier in %s): %d", args.tiers, len(cases))

    # ── 4. Shard slicing ───────────────────────────────────────────────────
    if args.shard is not None and args.n_shards is not None:
        if args.shard < 1 or args.shard > args.n_shards:
            ap.error(f"--shard {args.shard} invalid for --n-shards {args.n_shards}")
        cases = [c for i, c in enumerate(cases) if i % args.n_shards == (args.shard - 1)]
        logger.info("Shard %d/%d: %d cases", args.shard, args.n_shards, len(cases))

    if args.cases:
        wanted = set(args.cases)
        cases = [c for c in cases if f"{c[0]}/{c[1]}" in wanted]
        logger.info("Restricted to --cases (%d/%d matched).", len(cases), len(wanted))

    if args.max_cases is not None:
        cases = cases[: args.max_cases]
        logger.info("Limit to first %d cases (smoke).", len(cases))

    # ── 5. Export loop ─────────────────────────────────────────────────────
    n_ok, n_skip, n_fail = 0, 0, 0
    for site_id, case_name, solve_iter in cases:
        if site_id not in sites:
            logger.warning("Skip %s/%s — no entry in sites.csv", site_id, case_name)
            n_fail += 1
            continue
        site = sites[site_id]

        # case_tsNNN → timestamp from run_matrix order
        try:
            idx = int(case_name.replace("case_ts", ""))
        except ValueError:
            logger.warning("Skip %s/%s — unparseable case index", site_id, case_name)
            n_fail += 1
            continue
        if site_id not in runs or idx >= len(runs[site_id]):
            logger.warning("Skip %s/%s — no timestamp at index %d", site_id, case_name, idx)
            n_fail += 1
            continue
        timestamp = runs[site_id][idx]

        case_dir = args.campaign_dir / site_id / case_name
        if not case_dir.is_dir():
            logger.warning("Skip %s/%s — case dir missing", site_id, case_name)
            n_fail += 1
            continue

        era5_zarr = args.era5_dir / f"era5_{site_id}.zarr"
        if not era5_zarr.is_dir():
            logger.warning("Skip %s/%s — ERA5 zarr missing: %s", site_id, case_name, era5_zarr)
            n_fail += 1
            continue

        out = args.output_dir / f"{site_id}_{case_name}" / "grid.zarr"
        if args.skip_existing and out.exists():
            n_skip += 1
            continue

        time_arg = args.time if args.time else solve_iter
        # Inline call to export_to_grid_zarr_v2.main()
        # We rebuild argv to reuse its argparse machinery.
        argv = [
            "export_to_grid_zarr_v2.py",
            "--case-dir", str(case_dir),
            "--site-id", site_id,
            "--site-lat", str(site["lat"]),
            "--site-lon", str(site["lon"]),
            "--era5-zarr", str(era5_zarr),
            "--timestamp", timestamp,
            "--time", time_arg,
            "--output", str(out),
        ]
        if args.include_turb:
            argv += ["--include-turb", *args.include_turb]

        old_argv = sys.argv
        sys.argv = argv
        try:
            rc = exp.main()
        except SystemExit as e:
            rc = int(e.code) if e.code is not None else 0
        except Exception as e:
            import traceback
            logger.error("FAIL %s/%s: %s\n%s", site_id, case_name, type(e).__name__,
                         traceback.format_exc())
            n_fail += 1
            continue
        finally:
            sys.argv = old_argv

        if rc == 0:
            n_ok += 1
        else:
            n_fail += 1

    logger.info("=== batch done: ok=%d  skip=%d  fail=%d ===", n_ok, n_skip, n_fail)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
