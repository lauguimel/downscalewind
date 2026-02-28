#!/usr/bin/env python
"""ingest_data.py — Sequential data ingestion for Module 2A.

Runs three ingestion phases in order:
  1. ERA5 IOP pressure levels (CDS API)
  2. Perdigão tower observations (manual NetCDF → Zarr)
  3. COP-DEM GLO-30 terrain (AWS Open Data, no account needed)

Usage:
    python notebooks/ingest_data.py
    python notebooks/ingest_data.py --start 2017-05 --end 2017-06
    python notebooks/ingest_data.py --skip-era5
    python notebooks/ingest_data.py --skip-era5 --skip-obs   # terrain only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
RAW       = ROOT / "data" / "raw"
ERA5_ZARR = RAW / "era5_perdigao.zarr"
OBS_ZARR  = RAW / "perdigao_obs.zarr"
SRTM_TIF  = RAW / "srtm_perdigao_30m.tif"
OBS_RAW   = RAW / "perdigao_obs_raw"
INGEST    = ROOT / "services" / "data-ingestion"


def _run(cmd: list[str], label: str) -> None:
    print(f"\n⏳  {label}…")
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode == 0:
        print(f"✅  {label}")
    else:
        print(f"❌  {label}  (exit {r.returncode})")
        sys.exit(r.returncode)


def _size(path: Path) -> str:
    if not path.exists():
        return "❌ absent"
    total = (
        sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        if path.is_dir()
        else path.stat().st_size
    )
    return f"✅ {total / 1e6:.0f} Mo"


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest ERA5 + observations + terrain")
    p.add_argument("--start",      default="2017-05", help="ERA5 start month (YYYY-MM)")
    p.add_argument("--end",        default="2017-06", help="ERA5 end month (YYYY-MM)")
    p.add_argument("--skip-era5",  action="store_true", help="Skip ERA5 download")
    p.add_argument("--skip-obs",   action="store_true", help="Skip observation conversion")
    p.add_argument("--skip-srtm",  action="store_true", help="Skip COP-DEM download")
    args = p.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    OBS_RAW.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Module 2A — Data ingestion")
    print("=" * 60)

    # ── 1. ERA5 ──────────────────────────────────────────────────────────────
    print("\n── 1. ERA5 IOP (Copernicus CDS) ─────────────────────────────")
    if args.skip_era5:
        print("⏭   ERA5 : skipped (--skip-era5)")
    elif ERA5_ZARR.exists():
        print(f"✅  ERA5 : already present  →  {ERA5_ZARR.relative_to(ROOT)}")
    else:
        cdsapirc = Path.home() / ".cdsapirc"
        if not cdsapirc.exists():
            print(
                "❌  ERA5 : ~/.cdsapirc not found.\n"
                "    Create an account at https://cds.climate.copernicus.eu then run:\n\n"
                "    cat > ~/.cdsapirc << 'EOF'\n"
                "    url: https://cds.climate.copernicus.eu/api\n"
                "    key: YOUR-KEY-HERE\n"
                "    EOF\n\n"
                "    Then re-run this script."
            )
            sys.exit(1)
        _run(
            [
                sys.executable,
                str(INGEST / "ingest_era5.py"),
                "--site", "perdigao",
                "--start", args.start,
                "--end", args.end,
                "--output", str(ERA5_ZARR),
            ],
            f"ERA5 {args.start} → {args.end}",
        )

    # ── 2. Observations ───────────────────────────────────────────────────────
    print("\n── 2. Perdigão tower observations ───────────────────────────")
    if args.skip_obs:
        print("⏭   Observations : skipped (--skip-obs)")
    elif OBS_ZARR.exists():
        print(f"✅  Observations : already present  →  {OBS_ZARR.relative_to(ROOT)}")
    else:
        nc_files = list(OBS_RAW.glob("*.nc")) + list(OBS_RAW.glob("*.nc4"))
        if not nc_files:
            print(
                f"⚠️   No NetCDF files found in {OBS_RAW.relative_to(ROOT)}\n"
                "    Download NCAR/EOL ISFS QC 5-min data:\n"
                "      https://data.eol.ucar.edu/project/Perdigao\n"
                "      → Flux → NCAR-EOL QC 5-min ISFS\n"
                "      → Request files isfs_qc_tiltcor_YYYYMMDD.nc (2017-05-01 to 2017-07-01)\n"
                f"    Place the .nc files in {OBS_RAW.relative_to(ROOT)}/ then re-run."
            )
            # non-fatal: continue to next phase
        else:
            _run(
                [
                    sys.executable,
                    str(INGEST / "ingest_perdigao_obs.py"),
                    "--site", "perdigao",
                    "--raw-dir", str(OBS_RAW),
                    "--output", str(OBS_ZARR),
                ],
                f"Observations ({len(nc_files)} NetCDF files → Zarr)",
            )

    # ── 3. COP-DEM terrain ────────────────────────────────────────────────────
    print("\n── 3. COP-DEM GLO-30 terrain (AWS Open Data) ────────────────")
    if args.skip_srtm:
        print("⏭   COP-DEM : skipped (--skip-srtm)")
    elif SRTM_TIF.exists():
        mb = SRTM_TIF.stat().st_size / 1e6
        print(f"✅  COP-DEM : already present ({mb:.0f} Mo)  →  {SRTM_TIF.relative_to(ROOT)}")
    else:
        _run(
            [
                sys.executable,
                str(INGEST / "ingest_srtm.py"),
                "--site", "perdigao",
                "--output", str(SRTM_TIF),
            ],
            "COP-DEM 30m",
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  ERA5 zarr        : {_size(ERA5_ZARR)}")
    print(f"  Observations     : {_size(OBS_ZARR)}")
    print(f"  COP-DEM GeoTIFF  : {_size(SRTM_TIF)}")

    all_ok = ERA5_ZARR.exists() and OBS_ZARR.exists() and SRTM_TIF.exists()
    if all_ok:
        print("\n✅  All data ready.")
        print("    Next step: python notebooks/workflow_module2.py")
    else:
        print("\n⚠️   Some data is missing — complete the steps above and re-run.")


if __name__ == "__main__":
    main()
