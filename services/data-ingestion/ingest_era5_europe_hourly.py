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

PRESSURE_LEVELS = [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150]

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
):
    """Download one month of ERA5 hourly for the bbox. Resumable.

    If `days` is None, full month; otherwise restrict to given list of '01' style.
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

    if _nc_is_valid(pl_path):
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


def nc_to_arrays(pl_path: Path, sf_path: Path, with_z: bool = True) -> dict:
    """Load monthly NetCDF (pressure + surface) into numpy arrays."""
    import xarray as xr

    pl_ds = xr.open_dataset(pl_path)
    sf_ds = xr.open_dataset(sf_path)

    # Force N→S lat order for consistency with shared schema.
    if pl_ds.latitude[0] < pl_ds.latitude[-1]:
        pl_ds = pl_ds.isel(latitude=slice(None, None, -1))
        sf_ds = sf_ds.isel(latitude=slice(None, None, -1))

    level_var = "pressure_level" if "pressure_level" in pl_ds.dims else "level"
    if level_var != "level":
        pl_ds = pl_ds.rename({level_var: "level"})

    time_var = "valid_time" if "valid_time" in pl_ds.dims else "time"
    times = pl_ds[time_var].values.astype("datetime64[ns]")

    out = {
        "times": times,
        "lats": pl_ds.latitude.values.astype(np.float32),
        "lons": pl_ds.longitude.values.astype(np.float32),
        "levels": pl_ds.level.values.astype(np.float32),
        "u": pl_ds.u.values.astype(np.float32),
        "v": pl_ds.v.values.astype(np.float32),
        "t": pl_ds.t.values.astype(np.float32),
        "q": pl_ds.q.values.astype(np.float32),
    }
    if with_z and "z" in pl_ds.data_vars:
        out["z"] = pl_ds.z.values.astype(np.float32)

    out["u10"] = sf_ds.u10.values.astype(np.float32)
    out["v10"] = sf_ds.v10.values.astype(np.float32)
    out["t2m"] = sf_ds.t2m.values.astype(np.float32)
    if "d2m" not in sf_ds.data_vars:
        raise RuntimeError(
            f"{sf_path.name} is missing d2m — CDS request must include "
            "'2m_dewpoint_temperature'."
        )
    out["d2m"] = sf_ds.d2m.values.astype(np.float32)

    pl_ds.close()
    sf_ds.close()
    return out


# ── Zarr writer ──────────────────────────────────────────────────────────────

def write_zarr(out_path: Path, merged: dict, with_z: bool) -> None:
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
    coords.create_array("level", data=merged["levels"])
    coords["level"].attrs.update({"long_name": "pressure level", "units": "hPa"})

    pressure = store.create_group("pressure")
    pres_vars = ["u", "v", "t", "q"]
    if with_z and "z" in merged:
        pres_vars.append("z")
    for var in pres_vars:
        pressure.create_array(var, data=merged[var])

    surface = store.create_group("surface")
    for var in ["u10", "v10", "t2m", "d2m"]:
        surface.create_array(var, data=merged[var])

    store.attrs.update({
        "Conventions": "CF-1.9",
        "title": "ERA5 hourly — Europe bbox (with d2m for surrogate v2)",
        "source": "era5_hourly",
        "cadence_hours": 1,
        "created_by": "ingest_era5_europe_hourly.py (Phase G M_G6.5)",
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
def main(output, start, end, bbox, smoke, no_z, cache_dir, keep_nc):
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

    nc_files = []
    for y, m in months:
        days = ["01"] if smoke else None  # smoke: only the first day
        pl_path, sf_path = download_month(
            client, y, m, bbox_d, cache,
            pressure_vars=pressure_vars,
            surface_vars=SURFACE_VARIABLES,
            days=days,
        )
        nc_files.append((pl_path, sf_path))

    log.info("Loading and concatenating %d month NetCDF pairs...", len(nc_files))
    per_month = [nc_to_arrays(pl, sf, with_z=not no_z) for pl, sf in nc_files]

    merged = {
        "times": np.concatenate([d["times"] for d in per_month]),
        "lats": per_month[0]["lats"],
        "lons": per_month[0]["lons"],
        "levels": per_month[0]["levels"],
    }
    var_keys = ["u", "v", "t", "q", "u10", "v10", "t2m", "d2m"]
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
    write_zarr(output, merged, with_z=not no_z)
    log.info("Zarr write done")

    if not keep_nc:
        for pl, sf in nc_files:
            pl.unlink(missing_ok=True)
            sf.unlink(missing_ok=True)
        log.info("Removed intermediate NetCDF files")

    # Final smoke verification: re-open and report shapes
    import zarr
    g = zarr.open_group(str(output), mode="r")
    log.info("Verification — coords/time shape=%s", g["coords/time"].shape)
    log.info("Verification — pressure vars: %s", sorted(list(g["pressure"].array_keys())))
    log.info("Verification — surface vars : %s", sorted(list(g["surface"].array_keys())))
    log.info("DONE — exit criterion met for %s", output)


if __name__ == "__main__":
    main()
