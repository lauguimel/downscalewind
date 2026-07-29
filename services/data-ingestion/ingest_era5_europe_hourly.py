"""
ingest_era5_europe_hourly.py — ERA5 hourly Europe bbox download (with d2m).

Phase G mission M_G6.5: produce an ERA5 hourly store covering the Europe
bbox (FR/ES/PT primarily) that includes `d2m` (dewpoint 2m), which is
mandatory for `src.dataset_v2.build_era5_baseline_tensor(mode='surface')`
in the surrogate v2 pipeline. The legacy `data/raw/era5_europe.zarr`:
  - is at 6h cadence (constant prediction in 6h blocks at inference time),
  - is missing `d2m`.

This script fixes both issues for Phase G inference (M_G7).

Design notes
------------
- bbox is CLI argument "S,W,N,E" (no sites.csv dependency).
- Variables match dataset_v2 expectations:
    pressure: {u, v, t, q}   (10 plevels)  — z dropped (unused by surrogate
              v2 surface mode, but kept by default to remain consistent
              with the legacy schema; can be disabled via --no-z).
    surface : {t2m, d2m, u10, v10}
- Cadence: hourly (24 time steps/day).
- Monthly batching to stay under CDS request budget (~120 GB/req).
- Resumable per month via partial NetCDF files.
- Output Zarr format = v2 (consumer compatibility with Aqua zarr 2.18).

Usage
-----
Smoke (1 day, small bbox):
    python ingest_era5_europe_hourly.py \\
        --output data/raw/era5_europe_hourly_smoke.zarr \\
        --start 2023-01 --end 2023-01 \\
        --bbox "36,-9.5,52,10" --smoke

Production (6 years, FR/ES/PT bbox):
    python ingest_era5_europe_hourly.py \\
        --output data/raw/era5_europe_hourly.zarr \\
        --start 2018-01 --end 2023-12 \\
        --bbox "36,-9.5,52,10"
"""

from __future__ import annotations

import calendar
import logging
import os
import sys
from pathlib import Path

import click
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.logging_config import get_logger

log = get_logger("ingest_era5_europe_hourly")


# ── Constants ────────────────────────────────────────────────────────────────

# Canonical 10 plevels for surrogate v2 (verified against
# /scratch/maitreje/dsw/training_v2/<case>/grid.zarr input/era5_pressure_levels
# on 2026-05-26 — engineer.md said [...,150] but the trained surrogate v2 used
# [...,600,...,200] so we align to the actual training data).
PRESSURE_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200]

# dataset_v2 surrogate v2 expects {u, v, t, q} on pressure levels and
# {t2m, d2m, u10, v10} on surface (cf. engineer.md, dataset_v2_vit.py).
PRESSURE_VARIABLES = [
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "specific_humidity",
    "geopotential",  # not used by surrogate v2 surface mode but kept for schema parity
]
SURFACE_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "2m_dewpoint_temperature",
]
CDS_TO_SHORT = {
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "geopotential": "z",
    "temperature": "t",
    "specific_humidity": "q",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "100m_u_component_of_wind": "u100",
    "100m_v_component_of_wind": "v100",
    "2m_temperature": "t2m",
    "2m_dewpoint_temperature": "d2m",
}
HOURS_1H = [f"{h:02d}:00" for h in range(24)]


# ── bbox parsing ─────────────────────────────────────────────────────────────

def parse_bbox(bbox_str: str) -> dict:
    """Parse 'S,W,N,E' string → bbox dict.

    Example: '36,-9.5,52,10' → {south:36, west:-9.5, north:52, east:10}.
    """
    parts = [p.strip() for p in bbox_str.split(",")]
    if len(parts) != 4:
        raise click.BadParameter(
            f"bbox must be 'S,W,N,E' (4 values), got {len(parts)}: {bbox_str!r}"
        )
    south, west, north, east = (float(p) for p in parts)
    if south >= north:
        raise click.BadParameter(f"bbox: south ({south}) must be < north ({north})")
    if west >= east:
        raise click.BadParameter(f"bbox: west ({west}) must be < east ({east})")
    return {"south": south, "west": west, "north": north, "east": east}


# ── CDS client ───────────────────────────────────────────────────────────────

def build_cds_client():
    """Build CDS client. Fails clean if no auth available.

    Order: explicit CDS_API_KEY env var → fallback to ~/.cdsapirc.
    """
    try:
        import cdsapi
    except ImportError as e:
        log.error("cdsapi not installed. `pip install cdsapi`.")
        raise SystemExit(1) from e

    api_key = os.environ.get("CDS_API_KEY")
    api_url = os.environ.get("CDS_API_URL")
    if api_key:
        log.info("Using CDS_API_KEY from environment")
        return cdsapi.Client(url=api_url, key=api_key, quiet=True)
    cdsapirc = Path.home() / ".cdsapirc"
    if cdsapirc.exists():
        log.info("Using ~/.cdsapirc credentials")
        return cdsapi.Client(quiet=True)
    log.error(
        "No CDS credentials found. Set CDS_API_KEY env var or create ~/.cdsapirc"
    )
    raise SystemExit(2)


# ── NetCDF helpers (mirror ingest_era5_europe.py pattern) ────────────────────

def _nc_is_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1024:
        return False
    try:
        import xarray as xr
        with xr.open_dataset(path) as _ds:
            _ds.load()
        return True
    except Exception as exc:
        log.warning(
            "  %s exists but is not a valid NetCDF (%s) — will redownload",
            path.name, type(exc).__name__,
        )
        return False


def _iter_months(start: str, end: str):
    y0, m0 = int(start[:4]), int(start[5:7])
    y1, m1 = int(end[:4]), int(end[5:7])
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


def download_month(
    client,
    year: int,
    month: int,
    bbox: dict,
    cache_dir: Path,
    pressure_vars: list[str],
    surface_vars: list[str],
    days: list[str] | None = None,
    surface_only: bool = False,
):
    """Download one month of ERA5 hourly for the bbox. Resumable.

    If `days` is None, full month; otherwise restrict to given list of '01' style.
    If `surface_only` is True, the pressure-level request is skipped entirely
    (returns pl_path=None) — used by the light single-level stores (e.g. 100 m
    wind for the FuXi-CFD benchmark).
    """
    n_days = calendar.monthrange(year, month)[1]
    if days is None:
        days = [f"{d:02d}" for d in range(1, n_days + 1)]
    tag = f"{year}_{month:02d}"
    if len(days) < n_days:
        tag += f"_d{days[0]}-{days[-1]}"

    pl_path = cache_dir / f"_pl_hourly_{tag}.nc"
    sf_path = cache_dir / f"_sf_hourly_{tag}.nc"

    area = [bbox["north"], bbox["west"], bbox["south"], bbox["east"]]

    if surface_only:
        pl_path = None
    elif _nc_is_valid(pl_path):
        log.info(
            "  Pressure %s: skipping (valid NetCDF exists, %.0f MB)",
            tag, pl_path.stat().st_size / 1e6,
        )
    else:
        if pl_path.exists():
            log.info(
                "  Removing orphan partial %s (%.0f MB)",
                pl_path.name, pl_path.stat().st_size / 1e6,
            )
            pl_path.unlink()
        log.info("  Downloading pressure hourly %s ...", tag)
        pl_request = {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": pressure_vars,
            "pressure_level": [str(p) for p in PRESSURE_LEVELS],
            "year": str(year),
            "month": f"{month:02d}",
            "day": days,
            "time": HOURS_1H,
            "area": area,
        }
        client.retrieve("reanalysis-era5-pressure-levels", pl_request, str(pl_path))

    if _nc_is_valid(sf_path):
        log.info(
            "  Surface %s: skipping (valid NetCDF exists, %.0f MB)",
            tag, sf_path.stat().st_size / 1e6,
        )
    else:
        if sf_path.exists():
            log.info(
                "  Removing orphan partial %s (%.0f MB)",
                sf_path.name, sf_path.stat().st_size / 1e6,
            )
            sf_path.unlink()
        log.info("  Downloading surface hourly %s ...", tag)
        sf_request = {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": surface_vars,
            "year": str(year),
            "month": f"{month:02d}",
            "day": days,
            "time": HOURS_1H,
            "area": area,
        }
        client.retrieve("reanalysis-era5-single-levels", sf_request, str(sf_path))

    return pl_path, sf_path


def nc_to_arrays(
    pl_path: Path | None,
    sf_path: Path,
    with_z: bool = True,
    surface_only: bool = False,
    surface_short_names: list[str] | None = None,
) -> dict:
    """Load monthly NetCDF (pressure + surface) into numpy arrays.

    When `surface_only` is True, only the surface NetCDF is read and the
    coords/time come from the surface dataset. `surface_short_names` selects
    which surface variables to extract (defaults to {u10,v10,t2m,d2m}); only
    those present in the file are returned.
    """
    import xarray as xr

    sf_ds = xr.open_dataset(sf_path)
    # Force N→S lat order for consistency with shared schema.
    if sf_ds.latitude[0] < sf_ds.latitude[-1]:
        sf_ds = sf_ds.isel(latitude=slice(None, None, -1))

    out: dict = {}

    if surface_only:
        time_var = "valid_time" if "valid_time" in sf_ds.dims else "time"
        out["times"] = sf_ds[time_var].values.astype("datetime64[ns]")
        out["lats"] = sf_ds.latitude.values.astype(np.float32)
        out["lons"] = sf_ds.longitude.values.astype(np.float32)
        out["levels"] = np.array([], dtype=np.float32)
    else:
        pl_ds = xr.open_dataset(pl_path)
        if pl_ds.latitude[0] < pl_ds.latitude[-1]:
            pl_ds = pl_ds.isel(latitude=slice(None, None, -1))

        level_var = "pressure_level" if "pressure_level" in pl_ds.dims else "level"
        if level_var != "level":
            pl_ds = pl_ds.rename({level_var: "level"})

        time_var = "valid_time" if "valid_time" in pl_ds.dims else "time"
        out["times"] = pl_ds[time_var].values.astype("datetime64[ns]")
        out["lats"] = pl_ds.latitude.values.astype(np.float32)
        out["lons"] = pl_ds.longitude.values.astype(np.float32)
        out["levels"] = pl_ds.level.values.astype(np.float32)
        out["u"] = pl_ds.u.values.astype(np.float32)
        out["v"] = pl_ds.v.values.astype(np.float32)
        out["t"] = pl_ds.t.values.astype(np.float32)
        out["q"] = pl_ds.q.values.astype(np.float32)
        if with_z and "z" in pl_ds.data_vars:
            out["z"] = pl_ds.z.values.astype(np.float32)
        pl_ds.close()

    wanted = surface_short_names or ["u10", "v10", "t2m", "d2m"]
    out["surface_keys"] = []
    for short in wanted:
        if short not in sf_ds.data_vars:
            # ERA5 100 m winds land as u100/v100; fail loud if a requested
            # surface var is genuinely absent (bad CDS request).
            raise RuntimeError(
                f"{sf_path.name} is missing surface var '{short}' — check the "
                f"CDS request variable list (got {list(sf_ds.data_vars)})."
            )
        out[short] = sf_ds[short].values.astype(np.float32)
        out["surface_keys"].append(short)

    sf_ds.close()
    return out


# ── Zarr writer ──────────────────────────────────────────────────────────────

def write_zarr(
    out_path: Path,
    merged: dict,
    with_z: bool,
    surface_only: bool = False,
    surface_keys: list[str] | None = None,
) -> None:
    """Write merged arrays to Zarr v2 store (Aqua-compatible)."""
    import zarr

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        store = zarr.open_group(str(out_path), mode="w", zarr_format=2)
    except TypeError:
        # zarr <3.x: v2 is the only format
        store = zarr.open_group(str(out_path), mode="w")

    coords = store.create_group("coords")
    coords.create_array("time", data=merged["times"].astype("datetime64[ns]").astype(np.int64))
    coords["time"].attrs.update({
        "note": "UTC timestamps as int64 (datetime64[ns])",
        "cadence_hours": 1,
    })
    coords.create_array("lat", data=merged["lats"])
    coords["lat"].attrs.update({"long_name": "latitude", "units": "degrees_north"})
    coords.create_array("lon", data=merged["lons"])
    coords["lon"].attrs.update({"long_name": "longitude", "units": "degrees_east"})
    if not surface_only:
        coords.create_array("level", data=merged["levels"])
        coords["level"].attrs.update({"long_name": "pressure level", "units": "hPa"})

    if not surface_only:
        pressure = store.create_group("pressure")
        pres_vars = ["u", "v", "t", "q"]
        if with_z and "z" in merged:
            pres_vars.append("z")
        for var in pres_vars:
            pressure.create_array(var, data=merged[var])

    surface = store.create_group("surface")
    sf_vars = surface_keys or ["u10", "v10", "t2m", "d2m"]
    for var in sf_vars:
        surface.create_array(var, data=merged[var])

    if surface_only:
        title = "ERA5 hourly single-level — Europe bbox (100 m wind for FuXi-CFD benchmark)"
        created = "ingest_era5_europe_hourly.py --surface-only (FuXi-CFD 100m)"
    else:
        title = "ERA5 hourly — Europe bbox (with d2m for surrogate v2)"
        created = "ingest_era5_europe_hourly.py (Phase G M_G6.5)"
    store.attrs.update({
        "Conventions": "CF-1.9",
        "title": title,
        "source": "era5_hourly",
        "cadence_hours": 1,
        "created_by": created,
    })


# ── CLI ──────────────────────────────────────────────────────────────────────

@click.command()
@click.option(
    "--output",
    required=True,
    type=click.Path(path_type=Path),
    help="Output Zarr store path",
)
@click.option(
    "--start",
    required=True,
    help="Start month YYYY-MM (e.g. 2023-01)",
)
@click.option(
    "--end",
    required=True,
    help="End month YYYY-MM inclusive (e.g. 2023-03)",
)
@click.option(
    "--bbox",
    required=True,
    help="Bounding box 'S,W,N,E' in degrees. Default Europe: '36,-9.5,52,10'.",
)
@click.option(
    "--smoke",
    is_flag=True,
    default=False,
    help="Smoke mode: download only 1 day of the start month, with a small bbox kept as-is.",
)
@click.option(
    "--no-z",
    is_flag=True,
    default=False,
    help="Skip geopotential (z) on pressure levels (saves ~20%% data).",
)
@click.option(
    "--surface-only",
    is_flag=True,
    default=False,
    help="Skip pressure-level download entirely; write a light single-level "
         "store. Use with --surface-vars to choose which surface fields.",
)
@click.option(
    "--surface-vars",
    default=None,
    help="Comma-separated CDS single-level variable names to fetch (overrides "
         "the default {u10,v10,t2m,d2m}). E.g. "
         "'100m_u_component_of_wind,100m_v_component_of_wind' for FuXi-CFD.",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to store intermediate NetCDFs (default: <output_parent>/_cache_hourly).",
)
@click.option(
    "--keep-nc",
    is_flag=True,
    default=False,
    help="Keep intermediate NetCDF files after Zarr write.",
)
@click.option(
    "--max-days-per-req",
    type=int,
    default=31,
    help="Split a month into N-day chunks to stay under CDS 2026 size limit "
         "(typical: 16 = halve a month). Default 31 = no split (legacy).",
)
def main(output, start, end, bbox, smoke, no_z, surface_only, surface_vars,
         cache_dir, keep_nc, max_days_per_req):
    """Ingest ERA5 hourly for the Europe bbox, including d2m.

    Exit criteria (M_G6.5):
      - Zarr store written with pressure/{u,v,t,q[,z]} on 10 plevels and
        surface/{t2m,d2m,u10,v10}.
      - coords/time hourly cadence (Δt=1h).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    bbox_d = parse_bbox(bbox)
    log.info(
        "Europe hourly bbox: N=%.2f W=%.2f S=%.2f E=%.2f",
        bbox_d["north"], bbox_d["west"], bbox_d["south"], bbox_d["east"],
    )
    log.info(
        "Approx grid: %d × %d points (0.25° resolution)",
        int((bbox_d["north"] - bbox_d["south"]) / 0.25) + 1,
        int((bbox_d["east"] - bbox_d["west"]) / 0.25) + 1,
    )

    months = list(_iter_months(start, end))
    if not months:
        raise click.UsageError(f"empty month range {start} → {end}")
    if smoke:
        months = months[:1]
        log.info("SMOKE mode: only 1 day of %s-%02d", *months[0])

    cache = cache_dir or output.parent / "_cache_hourly"
    cache.mkdir(parents=True, exist_ok=True)

    client = build_cds_client()

    pressure_vars = list(PRESSURE_VARIABLES)
    if no_z:
        pressure_vars = [v for v in pressure_vars if v != "geopotential"]

    # Resolve surface variable list (CDS long names) + their short names.
    if surface_vars:
        surface_var_list = [s.strip() for s in surface_vars.split(",") if s.strip()]
    else:
        surface_var_list = list(SURFACE_VARIABLES)
    surface_short = []
    for cds_name in surface_var_list:
        if cds_name not in CDS_TO_SHORT:
            raise click.BadParameter(
                f"unknown surface variable {cds_name!r}; known: "
                f"{sorted(CDS_TO_SHORT)}"
            )
        surface_short.append(CDS_TO_SHORT[cds_name])
    log.info(
        "Mode: %s | surface vars=%s (short=%s)",
        "SURFACE-ONLY" if surface_only else "pressure+surface",
        surface_var_list, surface_short,
    )

    nc_files = []
    for y, m in months:
        n_days = calendar.monthrange(y, m)[1]
        if smoke:
            day_chunks = [["01"]]
        elif max_days_per_req >= n_days:
            day_chunks = [None]  # full month in one request
        else:
            # Split into chunks of `max_days_per_req` days (e.g. 16 → two halves
            # for a 31-day month).
            day_chunks = []
            for start_d in range(1, n_days + 1, max_days_per_req):
                end_d = min(n_days, start_d + max_days_per_req - 1)
                day_chunks.append([f"{d:02d}" for d in range(start_d, end_d + 1)])
        for days in day_chunks:
            pl_path, sf_path = download_month(
                client, y, m, bbox_d, cache,
                pressure_vars=pressure_vars,
                surface_vars=surface_var_list,
                days=days,
                surface_only=surface_only,
            )
            nc_files.append((pl_path, sf_path))

    log.info("Loading and concatenating %d month NetCDF pairs...", len(nc_files))
    per_month = [
        nc_to_arrays(
            pl, sf, with_z=not no_z,
            surface_only=surface_only, surface_short_names=surface_short,
        )
        for pl, sf in nc_files
    ]

    merged = {
        "times": np.concatenate([d["times"] for d in per_month]),
        "lats": per_month[0]["lats"],
        "lons": per_month[0]["lons"],
        "levels": per_month[0]["levels"],
    }
    if surface_only:
        var_keys = list(surface_short)
    else:
        var_keys = ["u", "v", "t", "q"] + list(surface_short)
        if (not no_z) and all("z" in d for d in per_month):
            var_keys.append("z")
    for var in var_keys:
        merged[var] = np.concatenate([d[var] for d in per_month], axis=0)

    n_t = len(merged["times"])
    log.info(
        "Final shape: %d times × %d levels × %d × %d",
        n_t, len(merged["levels"]), len(merged["lats"]), len(merged["lons"]),
    )

    # Sanity: confirm hourly cadence
    if n_t > 1:
        dt_ns = np.diff(merged["times"].astype("datetime64[ns]").astype(np.int64))
        dt_hours = dt_ns / 1e9 / 3600.0
        uniq = np.unique(np.round(dt_hours).astype(int))
        log.info("Cadence Δt (hours): unique=%s", uniq.tolist())
        if not (len(uniq) == 1 and uniq[0] == 1):
            log.warning(
                "Unexpected cadence: expected 1h everywhere, got %s",
                uniq.tolist(),
            )

    log.info("Writing Zarr v2 store: %s", output)
    write_zarr(
        output, merged, with_z=not no_z,
        surface_only=surface_only, surface_keys=surface_short,
    )
    log.info("Zarr write done")

    if not keep_nc:
        for pl, sf in nc_files:
            if pl is not None:
                pl.unlink(missing_ok=True)
            sf.unlink(missing_ok=True)
        log.info("Removed intermediate NetCDF files")

    # Final smoke verification: re-open and report shapes
    import zarr
    g = zarr.open_group(str(output), mode="r")
    log.info("Verification — coords/time shape=%s", g["coords/time"].shape)
    if not surface_only:
        log.info("Verification — pressure vars: %s", sorted(list(g["pressure"].array_keys())))
    log.info("Verification — surface vars : %s", sorted(list(g["surface"].array_keys())))
    log.info("DONE — exit criterion met for %s", output)


if __name__ == "__main__":
    main()
