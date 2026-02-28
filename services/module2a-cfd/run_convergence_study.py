"""
run_convergence_study.py — Mesh convergence study for CFD pipeline

Tests multiple horizontal resolutions × context variants on the canonical
validation case (Perdigão SW, neutral stability).

Variants
--------
  context_cells=1 : Variante 0 — 25×25 km pipeline test (<1 min, blockMesh only)
  context_cells=3 : Variante A — 75×75 km (25 km buffer each side)
  context_cells=5 : Variante B — 125×125 km (50 km buffer each side)

Usage
-----
    python run_convergence_study.py \
        --case-id 2017-05-15T12:00 \
        --resolutions-m 10000 5000 1000 500 \
        --context-cells 1 3 5 \
        --vertical-variants 20 30 50 \
        --era5 data/raw/era5_perdigao.zarr \
        --srtm data/raw/srtm_perdigao_30m.tif \
        --z0map data/raw/z0_perdigao.tif \
        --output data/processed/convergence_study/ \
        --n-cores 8
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# Ensure local module directory is on the path (handles 'module2a-cfd' hyphen)
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[2]))

logger = logging.getLogger(__name__)

# Default resolutions to test (horizontal, metres)
DEFAULT_RESOLUTIONS_M = [500, 250, 100]
# Default context variants (start with 1×1 single ERA5 cell)
DEFAULT_CONTEXT_CELLS = [1]
# Default vertical layer counts (None = auto from resolution/AR)
DEFAULT_VERTICAL_VARIANTS = [None]


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single(
    case_id: str,
    resolution_m: float,
    context_cells: int,
    n_z: int,
    era5_zarr: Path,
    srtm_tif: Path | None,
    z0_tif: Path | None,
    site_cfg: dict,
    output_root: Path,
    n_cores: int = 8,
) -> dict:
    """Run one CFD case and return result metrics.

    Steps
    -----
    1. prepare_inflow  — ERA5 → inlet profile JSON
    2. generate_mesh   — SRTM → case directory + templates
    3. openfoam_runner — blockMesh + snappyHexMesh + solver
    4. export_cfd      — OpenFOAM results → Zarr + CSV

    Returns
    -------
    dict with keys: case_label, resolution_m, context_cells, n_z,
                    n_cells, max_non_ortho, max_skewness,
                    cpu_time_s, ok, error
    """
    from prepare_inflow import prepare_inflow
    from generate_mesh import generate_mesh
    from openfoam_runner import OpenFOAMRunner
    from export_cfd import export_cfd

    safe_id    = case_id.replace(":", "_")  # colons break Docker volume mounts
    nz_str = f"nz{n_z}" if n_z is not None else "nzauto"
    case_label = f"{safe_id}_{int(resolution_m)}m_{context_cells}x{context_cells}_{nz_str}"
    case_dir   = output_root / "cases" / case_label

    site_lat = site_cfg["site"]["coordinates"]["latitude"]
    site_lon = site_cfg["site"]["coordinates"]["longitude"]

    result = {
        "case_label":    case_label,
        "resolution_m":  resolution_m,
        "context_cells": context_cells,
        "n_z":           n_z,
        "n_cells":       0,
        "max_non_ortho": None,
        "max_skewness":  None,
        "cpu_time_s":    None,
        "ok":            False,
        "error":         None,
    }

    t0 = time.perf_counter()

    try:
        # Step 1 — inflow profile
        inflow_json = output_root / "inflow" / f"{case_id.replace(':', '_')}.json"
        if not inflow_json.exists():
            logger.info("[%s] Preparing inflow profile…", case_label)
            prepare_inflow(
                era5_zarr=era5_zarr,
                timestamp=case_id,
                site_lat=site_lat,
                site_lon=site_lon,
                z0_tif=z0_tif,
                output_json=inflow_json,
            )
        else:
            logger.info("[%s] Inflow profile already exists, reusing.", case_label)

        # Step 2 — generate mesh + case templates
        logger.info("[%s] Generating mesh…", case_label)
        geom = generate_mesh(
            site_cfg=site_cfg,
            resolution_m=resolution_m,
            context_cells=context_cells,
            output_dir=case_dir,
            srtm_tif=srtm_tif,
            inflow_json=inflow_json,
            n_z=n_z,
        )
        result["n_cells"] = geom["n_x"] * geom["n_y"] * geom["n_z"]

        # Step 3 — OpenFOAM
        logger.info("[%s] Running OpenFOAM…", case_label)
        runner = OpenFOAMRunner(case_dir, n_cores=n_cores)
        # Always run snappyHexMesh: it provides distance-based refinement
        # (fine resolution near terrain, coarse aloft)
        quality = runner.run_case(
            solver="simpleFoam",
            skip_snappy=False,
            inflow_json=inflow_json,
        )
        result["n_cells"]       = quality.n_cells if quality.n_cells > 0 else result["n_cells"]
        result["max_non_ortho"] = quality.max_non_ortho
        result["max_skewness"]  = quality.max_skewness

        # Step 4 — export
        logger.info("[%s] Exporting results…", case_label)
        towers_yaml = Path(__file__).parents[2] / "configs" / "sites" / "perdigao_towers.yaml"
        export_cfd(
            case_dir=case_dir,
            towers_yaml=towers_yaml,
            site_cfg=site_cfg,
            case_id=case_label,
            output_dir=output_root / "results",
            metadata={
                "resolution_m":  resolution_m,
                "context_cells": context_cells,
                "n_z":           n_z,
                "era5_case_id":  case_id,
            },
        )

        result["cpu_time_s"] = time.perf_counter() - t0
        result["ok"]         = True
        logger.info("[%s] Done in %.0f s", case_label, result["cpu_time_s"])

    except Exception as exc:
        result["cpu_time_s"] = time.perf_counter() - t0
        result["error"]      = str(exc)
        logger.error("[%s] FAILED after %.0f s: %s", case_label, result["cpu_time_s"], exc)

    return result


# ---------------------------------------------------------------------------
# Convergence study loop
# ---------------------------------------------------------------------------

def run_convergence_study(
    case_id: str,
    resolutions_m: list[float],
    context_cells_list: list[int],
    vertical_variants: list[int | None],
    era5_zarr: Path,
    srtm_tif: Path | None,
    z0_tif: Path | None,
    site_cfg: dict,
    output_root: Path,
    n_cores: int = 8,
) -> list[dict]:
    """Run the full convergence study matrix.

    Loops over resolutions_m × context_cells_list.
    n_z is auto-computed from resolution and AR target (None = auto).

    Returns
    -------
    List of result dicts from run_single().
    """
    results = []

    logger.info("=== Horizontal convergence study ===")
    for res_m in resolutions_m:
        for ctx in context_cells_list:
            r = run_single(
                case_id=case_id,
                resolution_m=res_m,
                context_cells=ctx,
                n_z=None,  # auto from resolution / AR target
                era5_zarr=era5_zarr,
                srtm_tif=srtm_tif,
                z0_tif=z0_tif,
                site_cfg=site_cfg,
                output_root=output_root,
                n_cores=n_cores,
            )
            results.append(r)

    return results


# ---------------------------------------------------------------------------
# Write results summary
# ---------------------------------------------------------------------------

def write_results(results: list[dict], output_root: Path) -> None:
    """Write convergence study results to CSV and JSON."""
    output_root.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = output_root / "convergence_results.csv"
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
    logger.info("Results CSV: %s", csv_path)

    # JSON (full detail)
    json_path = output_root / "convergence_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results JSON: %s", json_path)

    # Summary to stdout
    _print_summary(results)


def _print_summary(results: list[dict]) -> None:
    ok     = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]

    print(f"\n{'=' * 60}")
    print(f"Convergence study: {len(ok)} OK, {len(failed)} FAILED")
    print(f"{'=' * 60}")
    print(f"{'Label':<40} {'Res[m]':>8} {'Ctx':>4} {'nZ':>4} "
          f"{'Cells':>10} {'NonOrtho':>10} {'CPU[s]':>8} {'OK':>4}")
    print("-" * 90)
    for r in results:
        flag = "OK" if r["ok"] else "FAIL"
        no   = f"{r['max_non_ortho']:.1f}" if r["max_non_ortho"] is not None else "—"
        cpu  = f"{r['cpu_time_s']:.0f}" if r["cpu_time_s"] is not None else "—"
        print(f"{r['case_label'][:40]:<40} "
              f"{r['resolution_m']:>8.0f} "
              f"{r['context_cells']:>4d} "
              f"{str(r['n_z'] or 'auto'):>4} "
              f"{r['n_cells']:>10,d} "
              f"{no:>10} "
              f"{cpu:>8} "
              f"{flag:>4}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run mesh convergence study for Module 2A CFD pipeline"
    )
    parser.add_argument("--case-id",    required=True,
                        help="Canonical case timestamp (e.g. 2017-05-15T12:00)")
    parser.add_argument("--resolutions-m", nargs="+", type=float,
                        default=DEFAULT_RESOLUTIONS_M)
    parser.add_argument("--context-cells", nargs="+", type=int,
                        default=DEFAULT_CONTEXT_CELLS, choices=[1, 3, 5])
    parser.add_argument("--vertical-variants", nargs="+", type=int,
                        default=DEFAULT_VERTICAL_VARIANTS)
    parser.add_argument("--era5",    required=True,
                        help="ERA5 zarr store")
    parser.add_argument("--srtm",    default=None,
                        help="SRTM GeoTIFF (default: data/raw/srtm_perdigao_30m.tif)")
    parser.add_argument("--z0map",   default=None,
                        help="z0 raster (GeoTIFF)")
    parser.add_argument("--site",    default="perdigao")
    parser.add_argument("--output",  required=True,
                        help="Output directory for cases + results")
    parser.add_argument("--n-cores", type=int, default=8)
    args = parser.parse_args()

    cfg_path = (
        Path(__file__).parents[2]
        / "configs" / "sites" / f"{args.site}.yaml"
    )
    with open(cfg_path) as f:
        site_cfg = yaml.safe_load(f)

    srtm_tif = args.srtm
    if srtm_tif is None:
        candidate = (
            Path(__file__).parents[2]
            / "data" / "raw" / f"srtm_{args.site}_30m.tif"
        )
        srtm_tif = candidate if candidate.exists() else None

    results = run_convergence_study(
        case_id=args.case_id,
        resolutions_m=args.resolutions_m,
        context_cells_list=args.context_cells,
        vertical_variants=args.vertical_variants,
        era5_zarr=Path(args.era5),
        srtm_tif=Path(srtm_tif) if srtm_tif else None,
        z0_tif=Path(args.z0map) if args.z0map else None,
        site_cfg=site_cfg,
        output_root=Path(args.output),
        n_cores=args.n_cores,
    )

    write_results(results, Path(args.output))
