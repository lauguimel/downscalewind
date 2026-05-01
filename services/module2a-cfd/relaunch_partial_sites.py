"""relaunch_partial_sites.py — extract partial sites from frozen status CSV.

Reads dataset_v2_status.csv produced by freeze_dataset_v2.py, identifies sites
with at least one missing or rejected case (i.e. n_solved < n_expected), and
writes:
  - sites_list_partial.txt   (one site_id per line, for PBS array indexing)
  - campaign_v2_phase2_partial.pbs (PBS array sized to the partial list)

Usage:
    python relaunch_partial_sites.py \
        --status-csv /scratch/maitreje/dsw/complex_terrain_v1/manifests/dataset_v2_status.csv \
        --campaign-dir ~/dsw/data/campaign/complex_terrain_v1 \
        --pbs-template ~/dsw/configs/hpc/campaign_v2_phase2_solve.pbs \
        --output-pbs ~/dsw/configs/hpc/campaign_v2_phase2_partial.pbs
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import click


@click.command()
@click.option("--status-csv", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--campaign-dir", type=click.Path(path_type=Path), required=True)
@click.option("--pbs-template", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output-pbs", type=click.Path(path_type=Path), required=True)
@click.option("--n-expected", type=int, default=15, help="Expected cases per site")
@click.option("--min-n-solved", type=int, default=0,
              help="Only relaunch sites with at least N already solved (filters out systemically broken sites)")
def main(
    status_csv: Path,
    campaign_dir: Path,
    pbs_template: Path,
    output_pbs: Path,
    n_expected: int,
    min_n_solved: int,
) -> None:
    by_site = defaultdict(lambda: {"solved": 0, "total": 0})
    with open(status_csv) as f:
        for row in csv.DictReader(f):
            sid = row["site_id"]
            by_site[sid]["total"] += 1
            if row["case_status"] in ("solved", "early_converged"):
                by_site[sid]["solved"] += 1

    partial_sites = sorted([
        sid for sid, c in by_site.items()
        if c["solved"] < n_expected and c["solved"] >= min_n_solved
    ])

    print(f"Total sites scanned: {len(by_site)}")
    print(f"Sites needing relaunch (n_solved >= {min_n_solved} and < {n_expected}): {len(partial_sites)}")
    if not partial_sites:
        print("No partial sites — nothing to relaunch.")
        return

    sites_list = campaign_dir.expanduser() / "sites_list_partial.txt"
    sites_list.parent.mkdir(parents=True, exist_ok=True)
    sites_list.write_text("\n".join(partial_sites) + "\n")
    print(f"Wrote {sites_list} ({len(partial_sites)} sites)")

    template = pbs_template.read_text()
    patched = template.replace(
        "sites_list.txt", "sites_list_partial.txt"
    ).replace(
        "#PBS -J 1-820", f"#PBS -J 1-{len(partial_sites)}"
    ).replace(
        "#PBS -N v2_solve", "#PBS -N v2_solve_partial"
    )
    output_pbs = output_pbs.expanduser()
    output_pbs.write_text(patched)
    print(f"Wrote {output_pbs}")
    print(f"\nNext: qsub {output_pbs}")


if __name__ == "__main__":
    main()
