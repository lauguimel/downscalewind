"""
freeze_dataset_v2.py — Phase 0: scan, classify, QA, and freeze campaign v2 dataset.

Scans the campaign directory on Aqua, classifies each site and case,
applies physical QA checks, assigns quality tiers (gold/silver/rejected),
and produces the deliverables for training.

Deliverables
------------
  dataset_v2_status.csv   — per-case status and QA flags
  dataset_v2_manifest.yaml — campaign metadata, counts, thresholds
  dataset_v2_splits.yaml   — geographic train/val/test splits (by site)
  dataset_v2_qa_summary.md — human-readable summary

Usage (on Aqua)
-----
    cd ~/dsw/services/module2a-cfd
    python freeze_dataset_v2.py \
        --campaign-dir /scratch/maitreje/dsw/complex_terrain_v1 \
        --sites-csv ~/dsw/data/campaign/complex_terrain_v1/sites.csv \
        --n-iter 300 \
        --output-dir /scratch/maitreje/dsw/complex_terrain_v1/manifests
"""
from __future__ import annotations

import csv
import json
import logging
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical QA thresholds
# ---------------------------------------------------------------------------
QA_THRESHOLDS = {
    "max_speed_ms": 50.0,
    "T_min_K": 200.0,
    "T_max_K": 350.0,
    "q_min_kgkg": 0.0,
    "q_max_kgkg": 0.05,
    "k_min": 0.0,
    "Ux_residual_converged": 1e-2,
}

# ---------------------------------------------------------------------------
# Site status classification
# ---------------------------------------------------------------------------

SITE_STATUS_MESH_FAILED = "mesh_failed"
SITE_STATUS_COMPLETE = "complete"
SITE_STATUS_PARTIAL = "partial"
SITE_STATUS_EMPTY = "empty"


def classify_site(site_dir: Path, n_iter: int, n_expected: int = 15) -> dict:
    """Classify a site based on mesh and solve status.

    Returns dict with: site_id, site_status, n_solved, n_failed, cases[].
    """
    site_id = site_dir.name
    mesh_dir = site_dir / "mesh"

    result = {
        "site_id": site_id,
        "has_mesh": False,
        "site_status": SITE_STATUS_EMPTY,
        "n_solved": 0,
        "n_failed": 0,
        "cases": [],
    }

    if not (mesh_dir / "constant" / "polyMesh" / "points").exists():
        result["site_status"] = SITE_STATUS_MESH_FAILED
        return result

    result["has_mesh"] = True

    case_dirs = sorted(site_dir.glob("case_ts*"))
    if not case_dirs:
        result["site_status"] = SITE_STATUS_EMPTY
        return result

    for case_dir in case_dirs:
        case_name = case_dir.name
        case_info = {
            "case_id": f"{site_id}/{case_name}",
            "site_id": site_id,
            "case_name": case_name,
        }

        final_field = case_dir / str(n_iter) / "U"
        if final_field.exists():
            case_info["case_status"] = "solved"
            case_info["solve_iter"] = n_iter
            result["n_solved"] += 1
        else:
            last_iter = _find_last_timestep(case_dir)
            if last_iter is not None and last_iter > 0:
                case_info["case_status"] = "early_converged"
                case_info["solve_iter"] = last_iter
                result["n_solved"] += 1
            else:
                case_info["case_status"] = "diverged"
                case_info["solve_iter"] = 0
                result["n_failed"] += 1

        result["cases"].append(case_info)

    if result["n_solved"] == n_expected:
        result["site_status"] = SITE_STATUS_COMPLETE
    elif result["n_solved"] > 0:
        result["site_status"] = SITE_STATUS_PARTIAL
    else:
        result["site_status"] = SITE_STATUS_EMPTY

    return result


def _find_last_timestep(case_dir: Path) -> int | None:
    """Find the last written timestep directory containing U field."""
    candidates = []
    for d in case_dir.iterdir():
        if d.is_dir() and d.name.isdigit() and (d / "U").exists():
            candidates.append(int(d.name))
    return max(candidates) if candidates else None


# ---------------------------------------------------------------------------
# Per-case QA (reads OpenFOAM fields directly)
# ---------------------------------------------------------------------------

def qa_case_from_foam(case_dir: Path, solve_iter: int) -> dict:
    """Run physical QA on a solved case by reading OpenFOAM field files.

    Returns dict with field stats and QA flags.
    """
    qa = {"flags": [], "field_stats": {}}
    time_dir = case_dir / str(solve_iter)

    if not time_dir.exists():
        qa["flags"].append("missing_time_dir")
        return qa

    u_path = time_dir / "U"
    if u_path.exists():
        try:
            u_data = _read_foam_vector_field(u_path)
            speed = np.sqrt(np.sum(u_data**2, axis=1))
            max_speed = float(np.max(speed))
            mean_speed = float(np.mean(speed))
            qa["field_stats"]["max_speed"] = max_speed
            qa["field_stats"]["mean_speed"] = mean_speed
            if max_speed > QA_THRESHOLDS["max_speed_ms"]:
                qa["flags"].append(f"max_speed={max_speed:.1f}")
        except Exception as e:
            qa["flags"].append(f"U_read_error={e}")

    t_path = time_dir / "T"
    if t_path.exists():
        try:
            t_data = _read_foam_scalar_field(t_path)
            t_min, t_max = float(np.min(t_data)), float(np.max(t_data))
            qa["field_stats"]["T_min"] = t_min
            qa["field_stats"]["T_max"] = t_max
            if t_min < QA_THRESHOLDS["T_min_K"]:
                qa["flags"].append(f"T_min={t_min:.1f}")
            if t_max > QA_THRESHOLDS["T_max_K"]:
                qa["flags"].append(f"T_max={t_max:.1f}")
        except Exception as e:
            qa["flags"].append(f"T_read_error={e}")

    q_path = time_dir / "q"
    if q_path.exists():
        try:
            q_data = _read_foam_scalar_field(q_path)
            q_min, q_max = float(np.min(q_data)), float(np.max(q_data))
            qa["field_stats"]["q_min"] = q_min
            qa["field_stats"]["q_max"] = q_max
            if q_min < QA_THRESHOLDS["q_min_kgkg"]:
                qa["flags"].append(f"q_min={q_min:.4f}")
            if q_max > QA_THRESHOLDS["q_max_kgkg"]:
                qa["flags"].append(f"q_max={q_max:.4f}")
        except Exception as e:
            qa["flags"].append(f"q_read_error={e}")

    k_path = time_dir / "k"
    if k_path.exists():
        try:
            k_data = _read_foam_scalar_field(k_path)
            k_min = float(np.min(k_data))
            qa["field_stats"]["k_min"] = k_min
            if k_min < QA_THRESHOLDS["k_min"]:
                qa["flags"].append(f"k_min={k_min:.4e}")
        except Exception as e:
            qa["flags"].append(f"k_read_error={e}")

    log_path = case_dir / "log.simpleFoam"
    if log_path.exists():
        residuals = _parse_final_residuals(log_path)
        qa["field_stats"].update(residuals)
        ux_res = residuals.get("final_Ux")
        if ux_res is not None and ux_res > QA_THRESHOLDS["Ux_residual_converged"]:
            qa["flags"].append(f"Ux_residual={ux_res:.2e}")

    return qa


def qa_case_from_zarr(zarr_path: Path, case_idx: int) -> dict:
    """Run physical QA on a case from a stacked site Zarr."""
    import zarr as zarr_lib

    qa = {"flags": [], "field_stats": {}}

    try:
        store = zarr_lib.open(str(zarr_path), mode="r")
    except Exception as e:
        qa["flags"].append(f"zarr_open_error={e}")
        return qa

    if "U" in store:
        U = np.asarray(store["U"][case_idx])
        speed = np.sqrt(np.sum(U**2, axis=-1))
        max_speed = float(np.nanmax(speed))
        mean_speed = float(np.nanmean(speed))
        qa["field_stats"]["max_speed"] = max_speed
        qa["field_stats"]["mean_speed"] = mean_speed
        if max_speed > QA_THRESHOLDS["max_speed_ms"]:
            qa["flags"].append(f"max_speed={max_speed:.1f}")

    if "T" in store:
        T = np.asarray(store["T"][case_idx])
        t_min, t_max = float(np.nanmin(T)), float(np.nanmax(T))
        qa["field_stats"]["T_min"] = t_min
        qa["field_stats"]["T_max"] = t_max
        if t_min < QA_THRESHOLDS["T_min_K"]:
            qa["flags"].append(f"T_min={t_min:.1f}")
        if t_max > QA_THRESHOLDS["T_max_K"]:
            qa["flags"].append(f"T_max={t_max:.1f}")

    if "q" in store:
        q = np.asarray(store["q"][case_idx])
        q_min, q_max = float(np.nanmin(q)), float(np.nanmax(q))
        qa["field_stats"]["q_min"] = q_min
        qa["field_stats"]["q_max"] = q_max
        if q_min < QA_THRESHOLDS["q_min_kgkg"]:
            qa["flags"].append(f"q_min={q_min:.4f}")
        if q_max > QA_THRESHOLDS["q_max_kgkg"]:
            qa["flags"].append(f"q_max={q_max:.4f}")

    if "k" in store:
        k = np.asarray(store["k"][case_idx])
        k_min = float(np.nanmin(k))
        qa["field_stats"]["k_min"] = k_min
        if k_min < QA_THRESHOLDS["k_min"]:
            qa["flags"].append(f"k_min={k_min:.4e}")

    return qa


# ---------------------------------------------------------------------------
# OpenFOAM field readers (lightweight, no PyFoam dependency)
# ---------------------------------------------------------------------------

def _read_foam_scalar_field(path: Path) -> np.ndarray:
    """Read an OpenFOAM scalar internalField."""
    text = path.read_text(errors="replace")
    m = re.search(r"internalField\s+nonuniform\s+List<scalar>\s*\n(\d+)\s*\n\(", text)
    if not m:
        m = re.search(r"internalField\s+uniform\s+([\d.eE+-]+)", text)
        if m:
            return np.array([float(m.group(1))])
        raise ValueError(f"Cannot parse scalar field: {path}")

    n = int(m.group(1))
    start = m.end()
    end = text.index(")", start)
    values = np.fromstring(text[start:end], sep="\n", count=n)
    if len(values) != n:
        raise ValueError(f"Expected {n} values, got {len(values)}: {path}")
    return values


def _read_foam_vector_field(path: Path) -> np.ndarray:
    """Read an OpenFOAM vector internalField."""
    text = path.read_text(errors="replace")
    m = re.search(r"internalField\s+nonuniform\s+List<vector>\s*\n(\d+)\s*\n\(", text)
    if not m:
        raise ValueError(f"Cannot parse vector field: {path}")

    n = int(m.group(1))
    start = m.end()
    end = text.index("\n)", start)
    block = text[start:end]

    values = re.findall(r"\(([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\)", block)
    if len(values) != n:
        raise ValueError(f"Expected {n} vectors, got {len(values)}: {path}")
    return np.array(values, dtype=np.float64)


def _parse_final_residuals(log_path: Path) -> dict:
    """Parse final initial residuals from log.simpleFoam."""
    result = {}
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return result

    for var in ("Ux", "Uy", "Uz", "k", "epsilon", "omega", "T", "q"):
        matches = re.findall(
            rf"Solving for {var}, Initial residual = ([\d.eE+-]+)", text
        )
        if matches:
            result[f"final_{var}"] = float(matches[-1])

    time_steps = re.findall(r"^Time = (\d+)", text, re.MULTILINE)
    if time_steps:
        result["n_iterations"] = int(time_steps[-1])

    return result


# ---------------------------------------------------------------------------
# Quality tier assignment
# ---------------------------------------------------------------------------

def assign_tier(case_status: str, qa_flags: list[str]) -> str:
    """Assign gold / silver / rejected based on status and QA flags."""
    if case_status in ("diverged", "mesh_failed"):
        return "rejected"

    critical_flags = [f for f in qa_flags if any(
        f.startswith(p) for p in ("max_speed=", "T_min=", "T_max=", "q_min=", "q_max=",
                                   "U_read_error", "missing_time_dir")
    )]

    if critical_flags:
        return "rejected"

    mild_flags = [f for f in qa_flags if any(
        f.startswith(p) for p in ("k_min=", "Ux_residual=", "q_read_error", "k_read_error")
    )]

    if mild_flags:
        return "silver"

    return "gold"


# ---------------------------------------------------------------------------
# Geographic splits
# ---------------------------------------------------------------------------

def compute_splits(
    sites_df: list[dict],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, str]:
    """Assign train/val/test splits by site, stratified by group.

    Returns {site_id: split}.
    """
    rng = np.random.default_rng(seed)

    groups = defaultdict(list)
    for row in sites_df:
        groups[row["group"]].append(row["site_id"])

    splits = {}
    for group, site_ids in sorted(groups.items()):
        n = len(site_ids)
        perm = rng.permutation(n)
        n_train = max(1, int(n * train_frac))
        n_val = max(1, int(n * val_frac))

        for i in perm[:n_train]:
            splits[site_ids[i]] = "train"
        for i in perm[n_train:n_train + n_val]:
            splits[site_ids[i]] = "val"
        for i in perm[n_train + n_val:]:
            splits[site_ids[i]] = "test"

    return splits


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_status_csv(records: list[dict], output: Path) -> None:
    """Write per-case status CSV."""
    fieldnames = [
        "case_id", "site_id", "case_name", "group", "lat", "lon",
        "case_status", "solve_iter", "tier", "split",
        "max_speed", "mean_speed", "T_min", "T_max",
        "q_min", "q_max", "k_min",
        "final_Ux", "n_iterations",
        "n_flags", "flags",
    ]

    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            row = dict(rec)
            row["flags"] = "; ".join(rec.get("flags", []))
            row["n_flags"] = len(rec.get("flags", []))
            writer.writerow(row)

    logger.info("Status CSV: %s (%d rows)", output, len(records))


def write_manifest_yaml(
    records: list[dict],
    sites_meta: dict,
    splits: dict[str, str],
    output: Path,
) -> None:
    """Write dataset manifest YAML."""
    tier_counts = Counter(r["tier"] for r in records)
    status_counts = Counter(r["case_status"] for r in records)
    split_counts = Counter(r["split"] for r in records if r["tier"] != "rejected")
    group_counts = Counter(r["group"] for r in records if r["tier"] != "rejected")

    n_sites_usable = len(set(
        r["site_id"] for r in records if r["tier"] in ("gold", "silver")
    ))

    manifest = {
        "schema_version": 1,
        "dataset": "complex_terrain_v2",
        "frozen_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "qa_thresholds": QA_THRESHOLDS,
        "counts": {
            "total_cases": len(records),
            "gold": tier_counts.get("gold", 0),
            "silver": tier_counts.get("silver", 0),
            "rejected": tier_counts.get("rejected", 0),
            "sites_usable": n_sites_usable,
            "sites_total": len(sites_meta),
        },
        "by_status": dict(status_counts),
        "by_split": dict(split_counts),
        "by_group": dict(group_counts),
        "splits": {
            "method": "geographic_stratified_by_group",
            "seed": 42,
            "fractions": {"train": 0.70, "val": 0.15, "test": 0.15},
            "n_sites": {
                split: len([s for s, sp in splits.items() if sp == split])
                for split in ("train", "val", "test")
            },
        },
    }

    with open(output, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

    logger.info("Manifest YAML: %s", output)


def write_splits_yaml(splits: dict[str, str], output: Path) -> None:
    """Write splits YAML (site_id → split mapping)."""
    grouped = {"train": [], "val": [], "test": []}
    for site_id, split in sorted(splits.items()):
        grouped[split].append(site_id)

    with open(output, "w") as f:
        yaml.dump(grouped, f, default_flow_style=False)

    logger.info("Splits YAML: %s", output)


def write_qa_summary(records: list[dict], sites_meta: dict, output: Path) -> None:
    """Write human-readable QA summary."""
    tier_counts = Counter(r["tier"] for r in records)
    status_counts = Counter(r["case_status"] for r in records)
    group_tier = defaultdict(lambda: Counter())
    for r in records:
        group_tier[r["group"]][r["tier"]] += 1

    flag_counts = Counter()
    for r in records:
        for f in r.get("flags", []):
            flag_type = f.split("=")[0]
            flag_counts[flag_type] += 1

    lines = [
        "# Dataset v2 QA Summary",
        f"",
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Case counts",
        "",
        f"| Status | Count |",
        f"|--------|------:|",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"| {status} | {count} |")

    lines += [
        "",
        "## Quality tiers",
        "",
        f"| Tier | Count | % |",
        f"|------|------:|--:|",
    ]
    total = len(records)
    for tier in ("gold", "silver", "rejected"):
        c = tier_counts.get(tier, 0)
        pct = 100 * c / total if total > 0 else 0
        lines.append(f"| {tier} | {c} | {pct:.1f}% |")

    lines += [
        "",
        "## By group × tier",
        "",
        "| Group | Gold | Silver | Rejected | Total |",
        "|-------|-----:|-------:|---------:|------:|",
    ]
    for group in sorted(group_tier.keys()):
        ct = group_tier[group]
        g, s, r = ct.get("gold", 0), ct.get("silver", 0), ct.get("rejected", 0)
        lines.append(f"| {group} | {g} | {s} | {r} | {g+s+r} |")

    if flag_counts:
        lines += [
            "",
            "## QA flag breakdown",
            "",
            "| Flag | Count |",
            "|------|------:|",
        ]
        for flag, count in flag_counts.most_common():
            lines.append(f"| {flag} | {count} |")

    lines.append("")
    output.write_text("\n".join(lines))
    logger.info("QA summary: %s", output)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--campaign-dir", type=click.Path(exists=True, path_type=Path), required=True,
    help="Campaign root on Aqua (e.g. /scratch/maitreje/dsw/complex_terrain_v1)",
)
@click.option(
    "--sites-csv", type=click.Path(exists=True, path_type=Path), required=True,
    help="sites.csv with site_id, lat, lon, group, etc.",
)
@click.option("--n-iter", type=int, default=300, help="Expected final iteration.")
@click.option("--n-timestamps", type=int, default=15, help="Expected timestamps per site.")
@click.option(
    "--output-dir", type=click.Path(path_type=Path), default=None,
    help="Output directory for deliverables (default: <campaign-dir>/manifests).",
)
@click.option("--qa-fields/--no-qa-fields", default=True,
              help="Read OF fields for QA (slower but thorough).")
@click.option("--max-sites", type=int, default=None,
              help="Limit to N sites (for testing).")
def main(
    campaign_dir: Path,
    sites_csv: Path,
    n_iter: int,
    n_timestamps: int,
    output_dir: Path | None,
    qa_fields: bool,
    max_sites: int | None,
):
    """Phase 0: freeze campaign v2 dataset."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if output_dir is None:
        output_dir = campaign_dir / "manifests"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load site metadata
    sites_meta = {}
    with open(sites_csv) as f:
        for row in csv.DictReader(f):
            sites_meta[row["site_id"]] = row
    logger.info("Loaded %d sites from %s", len(sites_meta), sites_csv)

    # Compute splits
    sites_list = [{"site_id": k, "group": v["group"]} for k, v in sites_meta.items()]
    splits = compute_splits(sites_list)
    logger.info("Splits: %s", Counter(splits.values()))

    # Discover and classify sites
    sites_dir = campaign_dir / "sites"
    site_dirs = sorted(d for d in sites_dir.iterdir() if d.is_dir() and d.name.startswith("ct_"))
    logger.info("Found %d site directories", len(site_dirs))

    if max_sites is not None:
        site_dirs = site_dirs[:max_sites]

    all_records = []
    site_status_counts = Counter()

    for i, site_dir in enumerate(site_dirs):
        if (i + 1) % 100 == 0 or i == 0:
            logger.info("[%d/%d] %s", i + 1, len(site_dirs), site_dir.name)

        site_result = classify_site(site_dir, n_iter, n_timestamps)
        site_status_counts[site_result["site_status"]] += 1

        meta = sites_meta.get(site_dir.name, {})

        for case_info in site_result["cases"]:
            record = {
                "case_id": case_info["case_id"],
                "site_id": case_info["site_id"],
                "case_name": case_info["case_name"],
                "group": meta.get("group", "unknown"),
                "lat": meta.get("lat", ""),
                "lon": meta.get("lon", ""),
                "case_status": case_info["case_status"],
                "solve_iter": case_info.get("solve_iter", 0),
                "split": splits.get(case_info["site_id"], "unknown"),
                "flags": [],
            }

            if qa_fields and case_info["case_status"] in ("solved", "early_converged"):
                case_dir = site_dir / case_info["case_name"]
                qa = qa_case_from_foam(case_dir, case_info["solve_iter"])
                record["flags"] = qa["flags"]
                record.update(qa["field_stats"])

            record["tier"] = assign_tier(case_info["case_status"], record["flags"])
            all_records.append(record)

        if not site_result["cases"] and site_result["site_status"] == SITE_STATUS_MESH_FAILED:
            all_records.append({
                "case_id": f"{site_dir.name}/mesh_failed",
                "site_id": site_dir.name,
                "case_name": "mesh_failed",
                "group": meta.get("group", "unknown"),
                "lat": meta.get("lat", ""),
                "lon": meta.get("lon", ""),
                "case_status": "mesh_failed",
                "solve_iter": 0,
                "split": splits.get(site_dir.name, "unknown"),
                "tier": "rejected",
                "flags": ["mesh_failed"],
            })

    # Log summary
    tier_counts = Counter(r["tier"] for r in all_records)
    logger.info("Site status: %s", dict(site_status_counts))
    logger.info("Case tiers: gold=%d, silver=%d, rejected=%d",
                tier_counts.get("gold", 0), tier_counts.get("silver", 0),
                tier_counts.get("rejected", 0))

    # Write deliverables
    write_status_csv(all_records, output_dir / "dataset_v2_status.csv")
    write_manifest_yaml(all_records, sites_meta, splits, output_dir / "dataset_v2_manifest.yaml")
    write_splits_yaml(splits, output_dir / "dataset_v2_splits.yaml")
    write_qa_summary(all_records, sites_meta, output_dir / "dataset_v2_qa_summary.md")

    # Print summary
    print()
    print("=" * 60)
    print("PHASE 0 — DATASET v2 FREEZE")
    print("=" * 60)
    print(f"  Sites scanned:   {len(site_dirs)}")
    for status, count in sorted(site_status_counts.items()):
        print(f"    {status:20s} {count}")
    print(f"  Cases total:     {len(all_records)}")
    print(f"    gold:          {tier_counts.get('gold', 0)}")
    print(f"    silver:        {tier_counts.get('silver', 0)}")
    print(f"    rejected:      {tier_counts.get('rejected', 0)}")
    usable = tier_counts.get("gold", 0) + tier_counts.get("silver", 0)
    print(f"  Usable for training: {usable}")
    print(f"  Outputs in: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
