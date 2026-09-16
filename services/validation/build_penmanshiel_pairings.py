"""Build hourly turbine pairings from Penmanshiel SCADA (Cubico, Zenodo 8253010, CC BY 4.0).

Output schema mirrors `data/inference/multiheight_towers_v1.parquet` so the mast runner
(derived from `data/validation/crest_deficit/run_masts_M_K2.py`) can consume it unchanged:
    station_id, timestamp, lat, lon, elev, height_obs, u_obs, v_obs, speed_obs,
    season, pop, source, wd_borrowed
plus turbine-specific columns kept for the power-based validation:
    power_kw, ws_nacelle_std, ws_density_adj, ambient_t_c

Only on-the-hour 10-min averages are kept (timestamp minute == 0), matching the ERA5 hourly
cadence and the M_K2 protocol. Nacelle wind speed is rotor-perturbed: use `power_kw` as the
primary validation target, `speed_obs` as secondary.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import click
import numpy as np
import pandas as pd

RAW = Path("data/raw/penmanshiel")
COLS = {
    "Wind speed (m/s)": "speed_obs",
    "Wind speed, Standard deviation (m/s)": "ws_nacelle_std",
    "Density adjusted wind speed (m/s)": "ws_density_adj",
    "Wind direction (°)": "wd_deg",
    "Power (kW)": "power_kw",
    "Nacelle ambient temperature (°C)": "ambient_t_c",
}


def _read_turbine_csv(raw: bytes) -> pd.DataFrame:
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    header_idx = next(i for i, l in enumerate(lines) if l.startswith("# Date and time"))
    header = lines[header_idx].lstrip("# ")
    body = "\n".join([header] + lines[header_idx + 1:])
    df = pd.read_csv(io.StringIO(body), usecols=lambda c: c in COLS or c == "Date and time")
    df = df.rename(columns={**COLS, "Date and time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    return df


@click.command()
@click.option("--year", default=2018, show_default=True)
@click.option("--output", type=click.Path(path_type=Path),
              default=Path("data/inference/penmanshiel_turbines_v1.parquet"), show_default=True)
def main(year: int, output: Path) -> None:
    static = pd.read_csv(RAW / "Penmanshiel_WT_static.csv")
    static = static.rename(columns=lambda c: c.strip())
    meta = {row["Alternative Title"]: row for _, row in static.iterrows()}

    frames = []
    for zp in sorted(RAW.glob(f"Penmanshiel_SCADA_{year}_*.zip")):
        with zipfile.ZipFile(zp) as zf:
            for name in zf.namelist():
                if not name.startswith("Turbine_Data_"):
                    continue
                tid = "T" + name.split("_Penmanshiel_")[1].split("_")[0]  # e.g. T01
                if tid not in meta:
                    click.echo(f"skip {name}: no static row for {tid}")
                    continue
                df = _read_turbine_csv(zf.read(name))
                df = df[df.timestamp.dt.minute == 0].copy()
                m = meta[tid]
                df["station_id"] = f"penmanshiel_{tid}"
                df["lat"] = float(m["Latitude"])
                df["lon"] = float(m["Longitude"])
                df["elev"] = float(m["Elevation (m)"])
                df["height_obs"] = float(m["Hub Height (m)"])
                frames.append(df)
                click.echo(f"{tid}: {len(df)} hourly rows from {name}")
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["speed_obs"])
    # meteorological convention: direction the wind comes FROM; u = -ws*sin(wd), v = -ws*cos(wd)
    wd = np.deg2rad(out["wd_deg"].to_numpy(dtype=float))
    out["u_obs"] = -out["speed_obs"] * np.sin(wd)
    out["v_obs"] = -out["speed_obs"] * np.cos(wd)
    out["wd_borrowed"] = False
    out["season"] = f"y{year}"
    out["pop"] = "turbine_scada"
    out["source"] = "penmanshiel"
    cols = ["station_id", "timestamp", "lat", "lon", "elev", "height_obs", "u_obs", "v_obs",
            "speed_obs", "season", "pop", "source", "wd_borrowed",
            "power_kw", "ws_nacelle_std", "ws_density_adj", "ambient_t_c", "wd_deg"]
    out = out[cols].sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output, index=False)
    click.echo(f"wrote {output}: {len(out)} rows, {out.station_id.nunique()} turbines, "
               f"{out.timestamp.min()} → {out.timestamp.max()}, "
               f"mean WS {out.speed_obs.mean():.2f} m/s, mean P {out.power_kw.mean():.0f} kW")


if __name__ == "__main__":
    main()
