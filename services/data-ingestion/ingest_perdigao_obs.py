"""
ingest_perdigao_obs.py — NCAR/EOL ISFS QC 5-min observations for Perdigão 2017.

Expects files downloaded from:
  https://data.eol.ucar.edu/project/Perdigao
  Dataset: NCAR-EOL Quality Controlled 5-minute ISFS Surface Flux Data,
           Geographic Coordinate, Tilt Corrected

Files: isfs_qc_tiltcor_YYYYMMDD.nc (one per day, 2017-05-01 to 2017-07-01)
       288 time steps/day at 5-min resolution, 41 sites

Variable naming convention:
  u_{h}m_{site}     — East wind component at height h (m), site name
  v_{h}m_{site}     — North wind component
  u_u__{h}m_{site}  — Variance of u → u_std = sqrt(abs(u_u__))
  T_{h}m_{site}     — Temperature (°C)
  RH_{h}m_{site}    — Relative humidity (%)

Coordinates:
  latitude_{site}, longitude_{site}, altitude_{site}  — scalar per site

Output Zarr schema:
  perdigao_obs.zarr/
    sites/
      u      [time, site_id, height]  float32  m/s East
      v      [time, site_id, height]  float32  m/s North
      u_std  [time, site_id, height]  float32  m/s (turbulence)
      T      [time, site_id, height]  float32  °C
      RH     [time, site_id, height]  float32  %
    coords/
      time       [time]      int64   ns since epoch UTC
      site_id    [site_id]   bytes   site name (e.g. b"tnw01")
      height_m   [height]    float32 m above ground
      lat        [site_id]   float32 degrees North
      lon        [site_id]   float32 degrees East
      altitude_m [site_id]   float32 m ASL

Usage:
    python ingest_perdigao_obs.py \\
        --site perdigao \\
        --raw-dir ../../data/raw/perdigao_obs_raw/ \\
        --output ../../data/raw/perdigao_obs.zarr
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import click
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.logging_config import get_logger

log = get_logger("ingest_perdigao_obs")

# IOP period (Intensive Observation Period)
IOP_START = np.datetime64("2017-05-01T00:00", "ns")
IOP_END   = np.datetime64("2017-06-15T23:59", "ns")

# Values above this threshold are ISFS FillValue (1e37f)
FILL_VALUE_THRESHOLD = 1e36


# ── ISFS parser ───────────────────────────────────────────────────────────────

def _parse_isfs_daily(raw_dir: Path) -> dict:
    """
    Parse NCAR/EOL ISFS daily NetCDF files.

    Discovers sites and heights from variable names (regex on u_{h}m_{site}).
    Extracts u, v, u_std (= sqrt(|u_u__|)), T, RH for all site × height combinations.

    Returns:
        dict with keys:
          sites     — list of str, sorted site names
          heights   — float32 array of discovered heights [m]
          times     — datetime64[ns] array [n_times]
          u         — float32 [n_times, n_sites, n_heights]
          v         — float32 [n_times, n_sites, n_heights]
          u_std     — float32 [n_times, n_sites, n_heights]
          T         — float32 [n_times, n_sites, n_heights]  °C
          RH        — float32 [n_times, n_sites, n_heights]  %
          lat       — float32 [n_sites]
          lon       — float32 [n_sites]
          altitude_m — float32 [n_sites]
    """
    try:
        import xarray as xr
    except ImportError:
        log.error("xarray required — install with: pip install xarray")
        sys.exit(1)

    # Locate files
    nc_files = sorted(raw_dir.glob("isfs_qc_tiltcor_*.nc"))
    if not nc_files:
        nc_files = sorted(raw_dir.glob("*.nc")) + sorted(raw_dir.glob("*.nc4"))
    if not nc_files:
        raise FileNotFoundError(f"No NetCDF files found in {raw_dir}")

    log.info("Opening ISFS files", extra={
        "n_files": len(nc_files),
        "first": nc_files[0].name,
        "last": nc_files[-1].name,
    })

    # Open and concatenate daily files without dask (not installed in this env)
    datasets = [xr.open_dataset(str(f), engine="netcdf4") for f in nc_files]
    ds = xr.concat(datasets, dim="time")

    # Discover sites and available heights from variable names
    # Pattern: u_{height}m_{site}  e.g. u_10m_tnw01
    var_pattern = re.compile(r"^u_(\d+)m_(.+)$")
    t_pattern   = re.compile(r"^[Tt]_(\d+)m_(.+)$")
    rh_pattern  = re.compile(r"^[Rr][Hh]_(\d+)m_(.+)$")
    sites_heights: dict[str, set[int]] = {}
    t_available: set[str] = set()   # variable names with T data
    rh_available: set[str] = set()  # variable names with RH data
    for var in ds.data_vars:
        m = var_pattern.match(var)
        if m:
            height = int(m.group(1))
            site = m.group(2)
            sites_heights.setdefault(site, set()).add(height)
        if t_pattern.match(var):
            t_available.add(var)
        if rh_pattern.match(var):
            rh_available.add(var)

    if not sites_heights:
        sample_vars = list(ds.data_vars)[:15]
        raise ValueError(
            f"No u_{{h}}m_{{site}} variables found.\n"
            f"Sample variables: {sample_vars}"
        )

    sites = sorted(sites_heights.keys())
    all_heights: set[int] = set()
    for h_set in sites_heights.values():
        all_heights.update(h_set)
    heights = sorted(all_heights)

    log.info("Discovered sites and heights", extra={
        "n_sites": len(sites),
        "heights_m": heights,
        "n_times": len(ds.time),
        "n_T_vars": len(t_available),
        "n_RH_vars": len(rh_available),
    })

    times = ds.time.values.astype("datetime64[ns]")
    n_times   = len(times)
    n_sites   = len(sites)
    n_heights = len(heights)

    # Pre-allocate output arrays
    u_arr     = np.full((n_times, n_sites, n_heights), np.nan, dtype=np.float32)
    v_arr     = np.full((n_times, n_sites, n_heights), np.nan, dtype=np.float32)
    u_std_arr = np.full((n_times, n_sites, n_heights), np.nan, dtype=np.float32)
    t_arr     = np.full((n_times, n_sites, n_heights), np.nan, dtype=np.float32)
    rh_arr    = np.full((n_times, n_sites, n_heights), np.nan, dtype=np.float32)
    lats      = np.full(n_sites, np.nan, dtype=np.float32)
    lons      = np.full(n_sites, np.nan, dtype=np.float32)
    alts      = np.full(n_sites, np.nan, dtype=np.float32)

    def _load_scalar(name: str) -> float:
        """Load scalar or 1-element variable, return float (NaN if not found)."""
        if name not in ds.data_vars and name not in ds.coords:
            return np.nan
        arr = np.atleast_1d(ds[name].values)
        return float(arr[0])

    def _load_1d(name: str) -> np.ndarray | None:
        """Load 1D time series, apply FillValue mask, return float32 or None."""
        if name not in ds.data_vars:
            return None
        vals = ds[name].values.astype(np.float32)
        vals[vals > FILL_VALUE_THRESHOLD] = np.nan
        return vals

    for s_idx, site in enumerate(sites):
        # Coordinates (scalar variables per site)
        lats[s_idx] = _load_scalar(f"latitude_{site}")
        lons[s_idx] = _load_scalar(f"longitude_{site}")
        alts[s_idx] = _load_scalar(f"altitude_{site}")

        for h_idx, h in enumerate(heights):
            u_vals = _load_1d(f"u_{h}m_{site}")
            if u_vals is not None:
                u_arr[:, s_idx, h_idx] = u_vals

            v_vals = _load_1d(f"v_{h}m_{site}")
            if v_vals is not None:
                v_arr[:, s_idx, h_idx] = v_vals

            uu_vals = _load_1d(f"u_u__{h}m_{site}")
            if uu_vals is not None:
                # u_std = sqrt(|variance|) — abs() handles float precision < 0
                u_std_arr[:, s_idx, h_idx] = np.sqrt(np.abs(uu_vals))

            # Temperature — try T_ then t_ (case variations in ISFS files)
            t_vals = _load_1d(f"T_{h}m_{site}") or _load_1d(f"t_{h}m_{site}")
            if t_vals is not None:
                t_arr[:, s_idx, h_idx] = t_vals

            # Relative humidity — try RH_ then rh_ then Rh_
            rh_vals = (_load_1d(f"RH_{h}m_{site}")
                       or _load_1d(f"rh_{h}m_{site}")
                       or _load_1d(f"Rh_{h}m_{site}"))
            if rh_vals is not None:
                rh_arr[:, s_idx, h_idx] = rh_vals

        n_valid = int(np.sum(np.isfinite(u_arr[:, s_idx, :])))
        log.debug("Site extracted", extra={"site": site, "n_valid_u": n_valid})

    ds.close()
    for d in datasets:
        d.close()

    log.info("ISFS parsing complete", extra={
        "n_times": n_times,
        "n_sites": n_sites,
        "n_heights": n_heights,
        "total_valid_u": int(np.sum(np.isfinite(u_arr))),
    })

    return {
        "sites":      sites,
        "heights":    np.array(heights, dtype=np.float32),
        "times":      times,
        "u":          u_arr,
        "v":          v_arr,
        "u_std":      u_std_arr,
        "T":          t_arr,
        "RH":         rh_arr,
        "lat":        lats,
        "lon":        lons,
        "altitude_m": alts,
    }


# ── Temporal aggregation ──────────────────────────────────────────────────────

def _aggregate_to_30min(
    times: np.ndarray, data_dict: dict[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Aggregate 5-min data to 30-min means.

    Works for any array shape [n_times, ...] — averages along axis 0.

    Args:
        times:     datetime64[ns] array of length n_times
        data_dict: dict of arrays with shape [n_times, ...]

    Returns:
        (times_30min, data_30min_dict)
    """
    t_start = times[0]
    t_end   = times[-1]
    dt_30min = np.timedelta64(30, "m")

    n_slots = int((t_end - t_start) / dt_30min) + 1
    times_30min = np.array(
        [t_start + i * dt_30min for i in range(n_slots)],
        dtype="datetime64[ns]",
    )

    data_30min: dict[str, np.ndarray] = {}
    for varname, arr in data_dict.items():
        shape_out = (n_slots,) + arr.shape[1:]
        out = np.full(shape_out, np.nan, dtype=np.float32)
        for i, t_center in enumerate(times_30min):
            t_lo = t_center - dt_30min // 2
            t_hi = t_center + dt_30min // 2
            mask = (times >= t_lo) & (times < t_hi)
            if mask.any():
                with np.errstate(all="ignore"):  # silence nanmean of all-NaN slice
                    out[i] = np.nanmean(arr[mask], axis=0)
        data_30min[varname] = out

    return times_30min, data_30min


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--site",       required=True,
              help="Site identifier (e.g. perdigao)")
@click.option("--raw-dir",
              default=str(Path(__file__).resolve().parents[2] / "data" / "raw" / "perdigao_obs_raw"),
              show_default=True,
              help="Directory containing isfs_qc_tiltcor_*.nc files")
@click.option("--output",
              default=str(Path(__file__).resolve().parents[2] / "data" / "raw" / "perdigao_obs.zarr"),
              show_default=True,
              help="Output Zarr store path")
@click.option("--config-dir",
              default=str(Path(__file__).resolve().parents[2] / "configs" / "sites"),
              show_default=True,
              help="Directory for site config YAML files")
@click.option("--iop-only/--no-iop-only", default=True, show_default=True,
              help="Restrict output to IOP period (2017-05-01 to 2017-06-15)")
def main(site: str, raw_dir: str, output: str, config_dir: str, iop_only: bool):
    """
    Convert NCAR/EOL ISFS NetCDF observations to a Zarr store.

    Reads isfs_qc_tiltcor_YYYYMMDD.nc files from --raw-dir,
    aggregates to 30-min means, and writes perdigao_obs.zarr.
    Also generates {site}_towers.yaml with lat/lon/altitude of each site.

    Download source:
      https://data.eol.ucar.edu/project/Perdigao
      → Flux → NCAR-EOL QC 5-min ISFS → isfs_qc_tiltcor_*.nc
    """
    import zarr

    raw_path    = Path(raw_dir)
    output_path = Path(output)
    config_path = Path(config_dir)

    if not raw_path.exists():
        log.error("raw-dir not found", extra={"path": str(raw_path)})
        sys.exit(1)

    nc_count = len(list(raw_path.glob("*.nc")) + list(raw_path.glob("*.nc4")))
    if nc_count == 0:
        log.error(
            "No NetCDF files found. Download from NCAR/EOL:\n"
            "  https://data.eol.ucar.edu/project/Perdigao\n"
            "  → Flux → NCAR-EOL QC 5-min ISFS → isfs_qc_tiltcor_*.nc",
            extra={"raw_dir": str(raw_path)},
        )
        sys.exit(1)

    log.info("Starting ingestion", extra={"site": site, "n_nc_files": nc_count})

    # ── Parse ISFS files ─────────────────────────────────────────────────────
    data = _parse_isfs_daily(raw_path)

    sites   = data["sites"]
    heights = data["heights"]
    times   = data["times"]

    # ── Aggregate 5-min → 30-min ─────────────────────────────────────────────
    log.info("Aggregating to 30-min")
    times_30min, data_30min = _aggregate_to_30min(times, {
        "u":     data["u"],
        "v":     data["v"],
        "u_std": data["u_std"],
        "T":     data["T"],
        "RH":    data["RH"],
    })

    # ── Filter to IOP ─────────────────────────────────────────────────────────
    if iop_only:
        mask = (times_30min >= IOP_START) & (times_30min <= IOP_END)
        times_30min = times_30min[mask]
        for k in data_30min:
            data_30min[k] = data_30min[k][mask]

    n_times   = len(times_30min)
    n_sites   = len(sites)
    n_heights = len(heights)

    log.info("Writing Zarr store", extra={
        "output": str(output_path),
        "n_times": n_times,
        "n_sites": n_sites,
        "n_heights": n_heights,
        "heights_m": heights.tolist(),
    })

    # ── Write Zarr ────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(output_path), mode="w")

    # Coordinates group
    coords = root.require_group("coords")

    arr = coords.create_array("time", shape=(n_times,), dtype=np.int64,
                              chunks=(n_times,), overwrite=True)
    arr[:] = times_30min.astype(np.int64)
    arr.attrs.update({"long_name": "time UTC", "units": "ns since epoch"})

    # Store site IDs as fixed-length byte strings (S10 handles up to 10 chars)
    site_bytes = np.array([s.encode("ascii") for s in sites], dtype="S10")
    arr = coords.create_array("site_id", shape=(n_sites,), dtype="S10",
                              chunks=(n_sites,), overwrite=True)
    arr[:] = site_bytes
    arr.attrs.update({"long_name": "site identifier (ISFS)"})

    arr = coords.create_array("height_m", shape=(n_heights,), dtype=np.float32,
                              chunks=(n_heights,), overwrite=True)
    arr[:] = heights
    arr.attrs.update({"long_name": "height above ground", "units": "m"})

    arr = coords.create_array("lat", shape=(n_sites,), dtype=np.float32,
                              chunks=(n_sites,), overwrite=True)
    arr[:] = data["lat"]
    arr.attrs.update({"units": "degrees_north"})

    arr = coords.create_array("lon", shape=(n_sites,), dtype=np.float32,
                              chunks=(n_sites,), overwrite=True)
    arr[:] = data["lon"]
    arr.attrs.update({"units": "degrees_east"})

    arr = coords.create_array("altitude_m", shape=(n_sites,), dtype=np.float32,
                              chunks=(n_sites,), overwrite=True)
    arr[:] = data["altitude_m"]
    arr.attrs.update({"units": "m ASL"})

    # Sites group — [time, site, height]
    sites_grp = root.require_group("sites")
    shape  = (n_times, n_sites, n_heights)
    chunks = (min(240, n_times), 1, n_heights)

    arr = sites_grp.create_array("u", shape=shape, dtype=np.float32,
                                 chunks=chunks, overwrite=True)
    arr[...] = data_30min["u"]
    arr.attrs.update({"long_name": "U wind (East)", "units": "m s-1"})

    arr = sites_grp.create_array("v", shape=shape, dtype=np.float32,
                                 chunks=chunks, overwrite=True)
    arr[...] = data_30min["v"]
    arr.attrs.update({"long_name": "V wind (North)", "units": "m s-1"})

    arr = sites_grp.create_array("u_std", shape=shape, dtype=np.float32,
                                 chunks=chunks, overwrite=True)
    arr[...] = data_30min["u_std"]
    arr.attrs.update({
        "long_name": "u standard deviation (turbulence)",
        "units": "m s-1",
    })

    arr = sites_grp.create_array("T", shape=shape, dtype=np.float32,
                                 chunks=chunks, overwrite=True)
    arr[...] = data_30min["T"]
    arr.attrs.update({"long_name": "temperature", "units": "degC"})

    arr = sites_grp.create_array("RH", shape=shape, dtype=np.float32,
                                 chunks=chunks, overwrite=True)
    arr[...] = data_30min["RH"]
    arr.attrs.update({"long_name": "relative humidity", "units": "%"})

    # Global attributes
    root.attrs.update({
        "Conventions": "CF-1.9",
        "title": "Perdigão 2017 campaign — NCAR/EOL ISFS surface flux observations",
        "site": site,
        "iop_period": "2017-05-01 to 2017-06-15",
        "time_resolution": "30min (aggregated from 5min ISFS)",
        "source": "NCAR/EOL ISFS QC 5-min, tilt-corrected geographic coordinates",
        "reference": "Fernando et al. (2019), doi:10.1175/BAMS-D-17-0227.1",
        "n_sites": n_sites,
        "n_heights": n_heights,
        "heights_m": heights.tolist(),
    })

    # ── Write perdigao_towers.yaml ────────────────────────────────────────────
    config_path.mkdir(parents=True, exist_ok=True)
    towers_yaml_path = config_path / f"{site}_towers.yaml"

    towers_data: dict = {}
    for i, s in enumerate(sites):
        entry: dict = {}
        if np.isfinite(data["lat"][i]):
            entry["lat"] = round(float(data["lat"][i]), 6)
        if np.isfinite(data["lon"][i]):
            entry["lon"] = round(float(data["lon"][i]), 6)
        if np.isfinite(data["altitude_m"][i]):
            entry["altitude_m"] = round(float(data["altitude_m"][i]), 1)
        entry["heights_m"] = [int(h) for h in heights if
                               np.any(np.isfinite(data_30min["u"][:, i, list(heights).index(h)]))]
        towers_data[s] = entry

    with open(towers_yaml_path, "w") as f:
        yaml.dump({"towers": towers_data}, f, default_flow_style=False, sort_keys=True)

    log.info("Towers YAML written", extra={
        "path": str(towers_yaml_path),
        "n_towers": len(towers_data),
    })

    # ── Summary ───────────────────────────────────────────────────────────────
    n_valid_u = int(np.sum(np.isfinite(data_30min["u"])))
    n_total   = data_30min["u"].size
    fill_frac = 1.0 - n_valid_u / max(n_total, 1)

    log.info("Ingestion complete", extra={
        "output": str(output_path),
        "n_times": n_times,
        "n_sites": n_sites,
        "n_heights": n_heights,
        "n_valid_u": n_valid_u,
        "fill_fraction": round(fill_frac, 3),
        "iop_only": iop_only,
    })

    print(f"Done. Zarr store: {output_path}")
    print(f"  {n_times} timesteps × {n_sites} sites × {n_heights} heights")
    print(f"  Heights: {heights.tolist()} m")
    print(f"  Valid u: {n_valid_u}/{n_total} ({100*(1-fill_frac):.1f}%)")
    print(f"  Towers YAML: {towers_yaml_path}")


if __name__ == "__main__":
    main()
