"""merge_freeze_shards.py — combine N partial CSVs into final freeze deliverables.

Reads:
  <output-dir>/dataset_v2_status_shard_001_of_NNN.csv
  <output-dir>/dataset_v2_status_shard_002_of_NNN.csv
  ...
  <output-dir>/dataset_v2_splits.yaml   (already written by shard 1)
  <sites-csv>                            (for site metadata)

Writes:
  <output-dir>/dataset_v2_status.csv     (concatenated)
  <output-dir>/dataset_v2_manifest.yaml  (aggregated counts)
  <output-dir>/dataset_v2_qa_summary.md  (aggregated QA)

Usage:
  python merge_freeze_shards.py \
      --output-dir /scratch/maitreje/dsw/complex_terrain_v1/manifests \
      --sites-csv ~/dsw/data/campaign/complex_terrain_v1/sites.csv \
      --n-shards 20
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import click

from freeze_dataset_v2 import (
    write_manifest_yaml,
    write_qa_summary,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option("--output-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--sites-csv", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--n-shards", type=int, required=True)
def main(output_dir: Path, sites_csv: Path, n_shards: int):
    sites_meta = {}
    with open(sites_csv) as f:
        for row in csv.DictReader(f):
            sites_meta[row["site_id"]] = row

    # Load splits (written by shard 1)
    import yaml
    splits_path = output_dir / "dataset_v2_splits.yaml"
    if not splits_path.exists():
        raise click.UsageError(f"Missing {splits_path} (shard 1 should have written it)")
    splits_yaml = yaml.safe_load(splits_path.read_text())
    splits = {}
    for split, site_ids in splits_yaml.items():
        for sid in site_ids:
            splits[sid] = split

    # Load all shard CSVs
    all_records: list[dict] = []
    missing = []
    for k in range(1, n_shards + 1):
        shard_csv = output_dir / f"dataset_v2_status_shard_{k:03d}_of_{n_shards:03d}.csv"
        if not shard_csv.exists():
            missing.append(shard_csv.name)
            continue
        with open(shard_csv) as f:
            for row in csv.DictReader(f):
                row["flags"] = [s.strip() for s in row.get("flags", "").split(";") if s.strip()]
                all_records.append(row)
        logger.info("Loaded %s", shard_csv.name)

    if missing:
        raise click.UsageError(f"Missing {len(missing)} shard CSVs: {missing[:5]}...")

    logger.info("Merged %d records from %d shards", len(all_records), n_shards)

    # Write concatenated status CSV
    final_csv = output_dir / "dataset_v2_status.csv"
    fieldnames = [
        "case_id", "site_id", "case_name", "group", "lat", "lon",
        "case_status", "solve_iter", "tier", "split",
        "max_speed", "mean_speed", "T_min", "T_max",
        "q_min", "q_max", "k_min",
        "final_Ux", "n_iterations",
        "n_flags", "flags",
    ]
    with open(final_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in all_records:
            row = dict(r)
            row["flags"] = "; ".join(r.get("flags", []))
            row["n_flags"] = len(r.get("flags", []))
            writer.writerow(row)
    logger.info("Wrote %s (%d rows)", final_csv, len(all_records))

    write_manifest_yaml(all_records, sites_meta, splits, output_dir / "dataset_v2_manifest.yaml")
    write_qa_summary(all_records, sites_meta, output_dir / "dataset_v2_qa_summary.md")


if __name__ == "__main__":
    main()
