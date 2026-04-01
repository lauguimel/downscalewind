"""
ingest_effis_fwi.py — Download EFFIS FWI historical data from CDS.

Source: Copernicus Climate Data Store
  Dataset: cems-fire-historical (alt: cems-fire-historical-v1)
  Product: ERA5-based fire danger reanalysis, ~0.25 deg, daily

Downloads daily Fire Weather Index (FWI) and optionally Initial Spread Index (ISI)
for the Iberian Peninsula, covering the Perdigao IOP and Pedrogao Grande fire (2017).

Output:
  data/raw/effis_fwi_iberia_2017.nc   — raw netCDF from CDS
  data/raw/effis_fwi_perdigao.csv     — FWI time series at Perdigao
  data/raw/effis_fwi_pedrogao.csv     — FWI time series at Pedrogao Grande

Usage:
    python ingest_effis_fwi.py --output ../../data/raw/effis_fwi_iberia_2017.nc
    python ingest_effis_fwi.py --start 2017-05-01 --end 2017-06-30
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.logging_config import get_logger

log = get_logger("ingest_effis_fwi")

# Key locations
PERDIGAO = {"name": "Perdigao", "lat": 39.716, "lon": -7.740}
PEDROGAO = {"name": "Pedrogao_Grande", "lat": 39.93, "lon": -8.23}

# Iberian Peninsula bounding box [N, W, S, E] (CDS convention)
IBERIA_AREA = [44, -10, 36, 4]

# Default period
DEFAULT_START = "2017-05-01"
DEFAULT_END = "2017-06-30"


# -- CDS download ------------------------------------------------------------

def _check_cdsapi():
    """Check that cdsapi is installed and configured."""
    try:
        import cdsapi  # noqa: F401
    except ImportError:
        log.error(
            "cdsapi not installed. Install with:\n"
            "  pip install cdsapi\n"
            "Then create ~/.cdsapirc with your CDS credentials:\n"
            "  url: https://cds.climate.copernicus.eu/api\n"
            "  key: <your-uid>:<your-api-key>"
        )
        sys.exit(1)

    cdsapirc = Path.home() / ".cdsapirc"
    if not cdsapirc.exists():
        log.error(
            "CDS API credentials not found. Create ~/.cdsapirc with:\n"
            "  url: https://cds.climate.copernicus.eu/api\n"
            "  key: <your-uid>:<your-api-key>\n"
            "Register at: https://cds.climate.copernicus.eu/user/register"
        )
        sys.exit(1)


def download_fwi(
    output_path: Path,
    start: str,
    end: str,
    variables: list[str],
    area: list[int],
) -> Path:
    """
    Download FWI data from CDS.

    Args:
        output_path: Path for the output netCDF file.
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).
        variables: List of CDS variable names.
        area: Bounding box [N, W, S, E].

    Returns:
        Path to the downloaded file.
    """
    import cdsapi

    start_dt = np.datetime64(start)
    end_dt = np.datetime64(end)

    # Build month/year lists from date range
    years = sorted(set(
        str(start_dt + np.timedelta64(i, "M"))[:4]
        for i in range(
            0,
            int((end_dt - start_dt) / np.timedelta64(1, "D")) // 28 + 2,
        )
    ))
    months = sorted(set(
        str(start_dt + np.timedelta64(i, "M"))[5:7]
        for i in range(
            0,
            int((end_dt - start_dt) / np.timedelta64(1, "D")) // 28 + 2,
        )
    ))
    days = [str(d).zfill(2) for d in range(1, 32)]

    log.info("Requesting CDS download", extra={
        "dataset": "cems-fire-historical",
        "variables": variables,
        "years": years,
        "months": months,
        "area": area,
    })

    # Dataset name: 'cems-fire-historical' (v4.1)
    # Alternative: 'cems-fire-historical-v1' if the above is deprecated
    c = cdsapi.Client()
    c.retrieve(
        "cems-fire-historical",
        {
            "product_type": "reanalysis",
            "variable": variables,
            "version": "4.1",
            "year": years,
            "month": months,
            "day": days,
            "area": area,
            "format": "netcdf",
        },
        str(output_path),
    )

    log.info("Download complete", extra={"path": str(output_path)})
    return output_path


# -- Extraction ---------------------------------------------------------------

def extract_point_timeseries(
    nc_path: Path,
    point: dict,
    csv_dir: Path,
) -> None:
    """
    Extract FWI time series at a single lat/lon point (nearest grid cell).

    Saves a CSV with columns: date, fwi [, isi].
    Prints summary statistics.
    """
    import xarray as xr

    ds = xr.open_dataset(str(nc_path))

    # Find the FWI variable — name varies across CDS versions
    fwi_candidates = ["fwi", "fire_weather_index", "FWI"]
    fwi_var = None
    for name in fwi_candidates:
        if name in ds.data_vars:
            fwi_var = name
            break
    if fwi_var is None:
        available = list(ds.data_vars)
        log.warning(
            "FWI variable not found, trying first data variable",
            extra={"available": available},
        )
        fwi_var = available[0] if available else None

    if fwi_var is None:
        log.error("No data variables found in netCDF")
        ds.close()
        return

    # Select nearest grid cell
    pt = ds.sel(
        latitude=point["lat"],
        longitude=point["lon"],
        method="nearest",
    )

    fwi_ts = pt[fwi_var].values
    times = pt.time.values

    # Build CSV dataframe
    import pandas as pd

    df = pd.DataFrame({"date": pd.to_datetime(times), "fwi": fwi_ts})

    # Also extract ISI if present
    isi_candidates = ["isi", "initial_spread_index", "ISI"]
    for name in isi_candidates:
        if name in ds.data_vars:
            df["isi"] = pt[name].values
            break

    df = df.set_index("date")

    # Summary stats
    print(f"\n{'='*60}")
    print(f"  {point['name']}  ({point['lat']:.3f}N, {point['lon']:.3f}E)")
    print(f"{'='*60}")
    print(f"  Period:  {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  N days:  {len(df)}")
    print(f"  FWI mean:   {df['fwi'].mean():.1f}")
    print(f"  FWI median: {df['fwi'].median():.1f}")
    print(f"  FWI max:    {df['fwi'].max():.1f}")
    print(f"  FWI > 30 (high):    {(df['fwi'] > 30).sum()} days")
    print(f"  FWI > 50 (extreme): {(df['fwi'] > 50).sum()} days")
    if "isi" in df.columns:
        print(f"  ISI mean:   {df['isi'].mean():.1f}")
        print(f"  ISI max:    {df['isi'].max():.1f}")

    # Pedrogao Grande fire date
    fire_date = "2017-06-17"
    if fire_date in df.index.strftime("%Y-%m-%d").values:
        fwi_fire = df.loc[fire_date, "fwi"]
        print(f"  FWI on {fire_date} (Pedrogao fire): {fwi_fire:.1f}")

    # Save CSV
    csv_path = csv_dir / f"effis_fwi_{point['name'].lower()}.csv"
    df.to_csv(csv_path, float_format="%.2f")
    log.info("CSV saved", extra={"path": str(csv_path), "n_rows": len(df)})
    print(f"  CSV: {csv_path}")

    ds.close()


# -- CLI ----------------------------------------------------------------------

@click.command()
@click.option(
    "--output",
    default=str(
        Path(__file__).resolve().parents[2] / "data" / "raw" / "effis_fwi_iberia_2017.nc"
    ),
    show_default=True,
    help="Output netCDF path",
)
@click.option("--start", default=DEFAULT_START, show_default=True,
              help="Start date (YYYY-MM-DD)")
@click.option("--end", default=DEFAULT_END, show_default=True,
              help="End date (YYYY-MM-DD)")
@click.option("--skip-download", is_flag=True, default=False,
              help="Skip CDS download, only extract from existing netCDF")
@click.option("--include-isi", is_flag=True, default=False,
              help="Also download Initial Spread Index")
def main(
    output: str,
    start: str,
    end: str,
    skip_download: bool,
    include_isi: bool,
):
    """
    Download EFFIS FWI reanalysis from CDS and extract time series.

    Downloads daily FWI (and optionally ISI) for the Iberian Peninsula,
    then extracts point time series at Perdigao and Pedrogao Grande.

    Requires cdsapi package and ~/.cdsapirc credentials.
    See: https://cds.climate.copernicus.eu/how-to-api
    """
    output_path = Path(output)
    csv_dir = output_path.parent

    # Build variable list
    variables = ["fire_weather_index"]
    if include_isi:
        variables.append("initial_spread_index")

    # Download from CDS
    if not skip_download:
        _check_cdsapi()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        download_fwi(output_path, start, end, variables, IBERIA_AREA)
    else:
        if not output_path.exists():
            log.error("netCDF file not found (use without --skip-download)",
                      extra={"path": str(output_path)})
            sys.exit(1)
        log.info("Skipping download, using existing file",
                 extra={"path": str(output_path)})

    # Extract point time series
    try:
        import xarray  # noqa: F401
        import pandas  # noqa: F401
    except ImportError:
        log.error("xarray and pandas required for extraction:\n"
                  "  pip install xarray pandas netcdf4")
        sys.exit(1)

    for point in [PERDIGAO, PEDROGAO]:
        extract_point_timeseries(output_path, point, csv_dir)

    print(f"\nDone. NetCDF: {output_path}")


if __name__ == "__main__":
    main()
