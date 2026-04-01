"""
qa_campaign_batch.py — Batch QA for campaign CFD results.

Checks convergence, field outliers, and coverage statistics.
Designed to run on UGA where campaign data lives.

Usage:
    python qa_campaign_batch.py \
        --campaign-dir /home/guillaume/dsw/campaign_9k \
        --output qa_report.json

    # Logs only (skip Zarr reading — fast)
    python qa_campaign_batch.py \
        --campaign-dir /home/guillaume/dsw/campaign_9k \
        --logs-only --output qa_report.json
"""
from __future__ import annotations

import csv
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import click
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
RESIDUAL_THRESHOLDS = {
    "Ux": 1e-4,
    "Uy": 1e-4,
    "Uz": 1e-4,
    "k": 1e-4,
    "epsilon": 1e-4,
    "T": 1e-5,
    "q": 1e-5,
}
DEFAULT_RESIDUAL_THRESHOLD = 1e-4

FIELD_LIMITS = {
    "max_speed": 50.0,        # m/s — flag if max(|U|) exceeds
    "T_min": 200.0,           # K
    "T_max": 350.0,           # K
    "q_min": 0.0,             # kg/kg
    "q_max": 0.05,            # kg/kg
    "k_min": 0.0,             # m2/s2 — negative TKE is unphysical
}


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def parse_log_simplefoam(log_path: Path) -> dict:
    """Parse log.simpleFoam for residuals, iteration count, and continuity.

    Returns dict with:
        n_iterations: int
        residuals: {var: [initial_res_per_iter]}
        final_residuals: {var: float}
        continuity_local: [float]  (sum local per iter)
        converged: bool
        converged_vars: {var: bool}
    """
    result = {
        "n_iterations": 0,
        "residuals": {},
        "final_residuals": {},
        "continuity_local": [],
        "converged": False,
        "converged_vars": {},
    }

    try:
        text = log_path.read_text(errors="replace")
    except OSError as e:
        logger.warning("Cannot read %s: %s", log_path, e)
        return result

    # Count time steps (= iterations for steady-state simpleFoam)
    time_steps = re.findall(r"^Time = (\d+)", text, re.MULTILINE)
    result["n_iterations"] = int(time_steps[-1]) if time_steps else 0

    # Residuals per variable
    pat_res = r"Solving for (\w+), Initial residual = ([\d.e+-]+), Final residual = ([\d.e+-]+), No Iterations (\d+)"
    data_init = defaultdict(list)
    data_final = defaultdict(list)
    for var, init_r, final_r, _ in re.findall(pat_res, text):
        data_init[var].append(float(init_r))
        data_final[var].append(float(final_r))

    result["residuals"] = {k: v for k, v in data_init.items()}

    # Final residual = last initial residual (standard OpenFOAM convention)
    for var, vals in data_init.items():
        result["final_residuals"][var] = vals[-1]

    # Continuity errors
    pat_cont = r"time step continuity errors : sum local = ([\d.e+-]+)"
    result["continuity_local"] = [float(v) for v in re.findall(pat_cont, text)]

    # Convergence check: all monitored variables below threshold
    for var, final_val in result["final_residuals"].items():
        threshold = RESIDUAL_THRESHOLDS.get(var, DEFAULT_RESIDUAL_THRESHOLD)
        result["converged_vars"][var] = final_val < threshold

    # Overall converged = Ux converged (primary indicator)
    if "Ux" in result["converged_vars"]:
        result["converged"] = result["converged_vars"]["Ux"]

    return result


# ---------------------------------------------------------------------------
# Zarr field checks (per-site stacked Zarr)
# ---------------------------------------------------------------------------

def check_fields_zarr(zarr_path: Path) -> list[dict]:
    """Check field outliers from a per-site stacked Zarr.

    Returns list of per-case dicts with field stats and flags.
    """
    import zarr

    try:
        store = zarr.open(str(zarr_path), mode="r")
    except Exception as e:
        logger.warning("Cannot open Zarr %s: %s", zarr_path, e)
        return []

    results = []

    # U: (n_cases, n_cells, 3)
    U = np.asarray(store["U"]) if "U" in store else None
    T = np.asarray(store["T"]) if "T" in store else None
    q = np.asarray(store["q"]) if "q" in store else None
    k = np.asarray(store["k"]) if "k" in store else None

    n_cases = U.shape[0] if U is not None else 0

    for i in range(n_cases):
        case_info = {"case_idx": i, "flags": []}

        if U is not None:
            speed = np.sqrt(np.sum(U[i] ** 2, axis=-1))
            case_info["max_speed"] = float(np.nanmax(speed))
            case_info["mean_speed"] = float(np.nanmean(speed))
            if case_info["max_speed"] > FIELD_LIMITS["max_speed"]:
                case_info["flags"].append(
                    f"max_speed={case_info['max_speed']:.1f} > {FIELD_LIMITS['max_speed']}"
                )

        if T is not None:
            t_min = float(np.nanmin(T[i]))
            t_max = float(np.nanmax(T[i]))
            case_info["T_min"] = t_min
            case_info["T_max"] = t_max
            if t_min < FIELD_LIMITS["T_min"]:
                case_info["flags"].append(f"T_min={t_min:.1f} < {FIELD_LIMITS['T_min']}")
            if t_max > FIELD_LIMITS["T_max"]:
                case_info["flags"].append(f"T_max={t_max:.1f} > {FIELD_LIMITS['T_max']}")

        if q is not None:
            q_min = float(np.nanmin(q[i]))
            q_max = float(np.nanmax(q[i]))
            case_info["q_min"] = q_min
            case_info["q_max"] = q_max
            if q_min < FIELD_LIMITS["q_min"]:
                case_info["flags"].append(f"q_min={q_min:.4f} < {FIELD_LIMITS['q_min']}")
            if q_max > FIELD_LIMITS["q_max"]:
                case_info["flags"].append(f"q_max={q_max:.4f} > {FIELD_LIMITS['q_max']}")

        if k is not None:
            k_min = float(np.nanmin(k[i]))
            case_info["k_min"] = k_min
            if k_min < FIELD_LIMITS["k_min"]:
                case_info["flags"].append(f"k_min={k_min:.4e} < {FIELD_LIMITS['k_min']}")

        results.append(case_info)

    return results


def find_site_zarr(site_dir: Path) -> Path | None:
    """Find the stacked Zarr for a site (multiple naming conventions)."""
    # Primary: site_XXXXX.zarr (from run_multisite_campaign.py)
    candidate = site_dir / f"{site_dir.name}.zarr"
    if candidate.exists():
        return candidate
    # Fallback: stacked.zarr
    candidate = site_dir / "stacked.zarr"
    if candidate.exists():
        return candidate
    # Fallback: fields.zarr
    candidate = site_dir / "fields.zarr"
    if candidate.exists():
        return candidate
    return None


# ---------------------------------------------------------------------------
# Coverage statistics
# ---------------------------------------------------------------------------

def compute_coverage(progress: list[dict], inflow_data: dict) -> dict:
    """Compute coverage stats from campaign progress + inflow metadata."""
    stats = {
        "n_sites": 0,
        "n_sites_ok": 0,
        "n_cases_solved": 0,
        "n_cases_failed": 0,
        "lat_range": [None, None],
        "lon_range": [None, None],
        "wind_dir_histogram": {},
        "wind_speed_histogram": {},
    }

    lats, lons = [], []
    wind_dirs, wind_speeds = [], []

    for entry in progress:
        stats["n_sites"] += 1
        if entry.get("zarr_ok"):
            stats["n_sites_ok"] += 1
        stats["n_cases_solved"] += entry.get("n_solved", 0)
        stats["n_cases_failed"] += entry.get("n_failed", 0)

        lat = entry.get("lat")
        lon = entry.get("lon")
        if lat is not None:
            lats.append(float(lat))
        if lon is not None:
            lons.append(float(lon))

    if lats:
        stats["lat_range"] = [min(lats), max(lats)]
    if lons:
        stats["lon_range"] = [min(lons), max(lons)]

    # Wind dir/speed from inflow metadata stored in Zarr
    for site_id, data in inflow_data.items():
        for wd in data.get("wind_dirs", []):
            wind_dirs.append(wd)
        for ws in data.get("wind_speeds", []):
            wind_speeds.append(ws)

    # Bin wind directions into 16 sectors (22.5 deg each)
    if wind_dirs:
        dirs = np.array(wind_dirs)
        bins = np.arange(0, 361, 22.5)
        counts, _ = np.histogram(dirs, bins=bins)
        for i, count in enumerate(counts):
            label = f"{bins[i]:.0f}-{bins[i+1]:.0f}"
            stats["wind_dir_histogram"][label] = int(count)

    # Bin wind speeds into 2 m/s bins
    if wind_speeds:
        speeds = np.array(wind_speeds)
        bins = np.arange(0, max(speeds) + 2, 2)
        counts, _ = np.histogram(speeds, bins=bins)
        for i, count in enumerate(counts):
            label = f"{bins[i]:.0f}-{bins[i+1]:.0f}"
            stats["wind_speed_histogram"][label] = int(count)

    return stats


# ---------------------------------------------------------------------------
# Main QA pipeline
# ---------------------------------------------------------------------------

def discover_sites(campaign_dir: Path) -> list[Path]:
    """Find all site_XXXXX directories."""
    return sorted([
        d for d in campaign_dir.iterdir()
        if d.is_dir() and d.name.startswith("site_")
    ])


def qa_site(site_dir: Path, logs_only: bool = False) -> dict:
    """Run QA on a single site. Returns per-case QA records.

    Campaign structure: site_XXXXX/site_XXXXX.zarr (stacked Zarr with
    U[n_ts, n_cells, 3], T/q/k[n_ts, n_cells], meta/{wind_dir, u_hub, case_id}).
    Logs are cleaned up after export — QA is primarily Zarr-based.
    """
    site_id = site_dir.name
    records = []
    inflow_meta = {"wind_dirs": [], "wind_speeds": []}

    zarr_path = find_site_zarr(site_dir)
    if zarr_path is None:
        return {"records": [], "inflow_meta": inflow_meta}

    if logs_only:
        # With stacked Zarr campaigns, logs are deleted after export.
        # logs_only mode just counts cases from meta.
        try:
            import zarr
            store = zarr.open(str(zarr_path), mode="r")
            if "meta" in store and "case_id" in store["meta"]:
                n_cases = store["meta"]["case_id"].shape[0]
                for i in range(n_cases):
                    records.append({
                        "site_id": site_id,
                        "case_id": f"{site_id}/ts{i:03d}",
                        "case_idx": i,
                        "converged": None,
                        "flags": [],
                        "flagged": False,
                    })
            if "meta" in store:
                if "wind_dir" in store["meta"]:
                    wd = np.asarray(store["meta"]["wind_dir"])
                    inflow_meta["wind_dirs"] = wd[~np.isnan(wd)].tolist()
                if "u_hub" in store["meta"]:
                    ws = np.asarray(store["meta"]["u_hub"])
                    inflow_meta["wind_speeds"] = ws[~np.isnan(ws)].tolist()
        except Exception as e:
            logger.warning("Cannot read %s: %s", zarr_path, e)
        return {"records": records, "inflow_meta": inflow_meta}

    # Full QA: read fields from stacked Zarr
    zarr_results = check_fields_zarr(zarr_path)

    # Extract inflow metadata
    try:
        import zarr
        store = zarr.open(str(zarr_path), mode="r")
        meta = store.get("meta", {})
        case_ids = np.asarray(meta["case_id"]) if "case_id" in meta else None
        wind_dirs = np.asarray(meta["wind_dir"]) if "wind_dir" in meta else None
        u_hubs = np.asarray(meta["u_hub"]) if "u_hub" in meta else None

        if wind_dirs is not None:
            inflow_meta["wind_dirs"] = wind_dirs[~np.isnan(wind_dirs)].tolist()
        if u_hubs is not None:
            inflow_meta["wind_speeds"] = u_hubs[~np.isnan(u_hubs)].tolist()
    except Exception:
        case_ids = None

    for i, zr in enumerate(zarr_results):
        cid = str(case_ids[i]) if case_ids is not None and i < len(case_ids) else f"ts{i:03d}"
        record = {
            "site_id": site_id,
            "case_id": f"{site_id}/{cid}",
            "case_idx": i,
            "converged": None,  # no logs available in stacked campaigns
            "max_speed": zr.get("max_speed"),
            "mean_speed": zr.get("mean_speed"),
            "T_min": zr.get("T_min"),
            "T_max": zr.get("T_max"),
            "q_min": zr.get("q_min"),
            "q_max": zr.get("q_max"),
            "k_min": zr.get("k_min"),
            "field_flags": zr.get("flags", []),
        }
        record["flags"] = list(record["field_flags"])
        record["flagged"] = len(record["flags"]) > 0
        records.append(record)

    return {"records": records, "inflow_meta": inflow_meta}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_csv(records: list[dict], output_path: Path) -> None:
    """Write QA results as CSV (one row per case)."""
    if not records:
        return

    fieldnames = [
        "case_id", "site_id", "case_idx",
        "n_iterations", "converged",
        "final_residual_Ux", "final_residual_cont",
        "max_speed", "mean_speed",
        "T_min", "T_max", "q_min", "q_max", "k_min",
        "flagged", "flags",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            row = dict(rec)
            # Serialize flags list to string
            row["flags"] = "; ".join(rec.get("flags", []))
            writer.writerow(row)


def print_summary(records: list[dict], coverage: dict) -> None:
    """Print human-readable summary to stdout."""
    n_total = len(records)
    n_converged = sum(1 for r in records if r.get("converged") is True)
    n_not_converged = sum(1 for r in records if r.get("converged") is False)
    n_no_log = sum(1 for r in records if r.get("converged") is None)
    n_flagged = sum(1 for r in records if r.get("flagged"))

    # Count specific field flags
    flag_counts = defaultdict(int)
    for r in records:
        for f in r.get("flags", []):
            # Group by flag type (before '=')
            flag_type = f.split("=")[0] if "=" in f else f
            flag_counts[flag_type] += 1

    print("=" * 60)
    print("QA CAMPAIGN SUMMARY")
    print("=" * 60)
    print(f"  Total cases:      {n_total}")
    print(f"  Converged:        {n_converged}")
    print(f"  Not converged:    {n_not_converged}")
    print(f"  No log available: {n_no_log}")
    print(f"  Flagged:          {n_flagged}")
    print()

    if flag_counts:
        print("  Flag breakdown:")
        for flag_type, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
            print(f"    {flag_type}: {count}")
        print()

    print(f"  Sites total:      {coverage.get('n_sites', '?')}")
    print(f"  Sites with Zarr:  {coverage.get('n_sites_ok', '?')}")
    print(f"  Cases solved:     {coverage.get('n_cases_solved', '?')}")
    print(f"  Cases failed:     {coverage.get('n_cases_failed', '?')}")
    print(f"  Lat range:        {coverage.get('lat_range', '?')}")
    print(f"  Lon range:        {coverage.get('lon_range', '?')}")

    # Top flagged cases
    flagged = [r for r in records if r.get("flagged")]
    if flagged:
        print()
        print(f"  Top flagged cases (showing up to 20):")
        for r in flagged[:20]:
            flags_str = "; ".join(r.get("flags", []))
            print(f"    {r['case_id']}: {flags_str}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--campaign-dir", type=click.Path(exists=True, path_type=Path), required=True,
    help="Campaign root directory (e.g. /home/guillaume/dsw/campaign_9k).",
)
@click.option(
    "--output", type=click.Path(path_type=Path), default=None,
    help="Output JSON report path (default: <campaign-dir>/qa_report.json).",
)
@click.option(
    "--logs-only", is_flag=True, default=False,
    help="Skip Zarr field checks (faster, log parsing only).",
)
@click.option(
    "--max-sites", type=int, default=None,
    help="Limit to first N sites (for quick testing).",
)
def main(campaign_dir: Path, output: Path | None, logs_only: bool, max_sites: int | None):
    """Batch QA for campaign CFD results."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if output is None:
        output = campaign_dir / "qa_report.json"

    # Load campaign progress if available
    progress_path = campaign_dir / "campaign_progress.json"
    progress = []
    if progress_path.exists():
        with open(progress_path) as f:
            progress = json.load(f)
        logger.info("Loaded campaign_progress.json: %d entries", len(progress))

    # Discover sites
    sites = discover_sites(campaign_dir)
    logger.info("Found %d site directories", len(sites))

    if max_sites is not None:
        sites = sites[:max_sites]
        logger.info("Limited to %d sites (--max-sites)", max_sites)

    # Run QA on each site
    all_records = []
    all_inflow = {}

    for i, site_dir in enumerate(sites):
        if (i + 1) % 50 == 0 or i == 0:
            logger.info("[%d/%d] Processing %s", i + 1, len(sites), site_dir.name)

        result = qa_site(site_dir, logs_only=logs_only)
        all_records.extend(result["records"])
        if result["inflow_meta"]["wind_dirs"]:
            all_inflow[site_dir.name] = result["inflow_meta"]

    logger.info("QA complete: %d case records from %d sites", len(all_records), len(sites))

    # Coverage statistics
    coverage = compute_coverage(progress, all_inflow)

    # Write JSON report
    report = {
        "campaign_dir": str(campaign_dir),
        "n_sites": len(sites),
        "n_cases": len(all_records),
        "logs_only": logs_only,
        "coverage": coverage,
        "thresholds": {
            "residuals": RESIDUAL_THRESHOLDS,
            "fields": FIELD_LIMITS,
        },
        "summary": {
            "n_converged": sum(1 for r in all_records if r.get("converged") is True),
            "n_not_converged": sum(1 for r in all_records if r.get("converged") is False),
            "n_flagged": sum(1 for r in all_records if r.get("flagged")),
        },
        "cases": all_records,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("JSON report: %s", output)

    # Write CSV
    csv_path = output.with_suffix(".csv")
    write_csv(all_records, csv_path)
    logger.info("CSV report: %s", csv_path)

    # Print summary
    print_summary(all_records, coverage)


if __name__ == "__main__":
    main()
