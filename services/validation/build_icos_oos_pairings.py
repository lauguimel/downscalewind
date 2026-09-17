"""Multi-height pairings for ICOS tall towers NEVER used in training (out-of-sample benchmark).

Reuses `build_icos` from services/module2b-surrogate/build_multiheight_pairings.py (same u/v
convention, same wd-borrowing rule, same JJA2020 window) with a different tower table, so the
resulting parquet has exactly the training-time schema. Heights above the surrogate top
(200 m AGL) and below 10 m are dropped. Towers: OXK, TOH, KRE, KIT (ICOS ATC meteo, CC BY 4.0).
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "services" / "module2b-surrogate"))
import build_multiheight_pairings as bmp  # noqa: E402

OOS_META = {
    "oxk": {"id": "OXK", "lat": 50.0300, "lon": 11.8083, "elev": 1022.0, "heights": [23.0, 90.0, 163.0]},
    "toh": {"id": "TOH", "lat": 51.8088, "lon": 10.5350, "elev": 801.0, "heights": [10.0, 76.0, 110.0, 147.0]},
    "kre": {"id": "KRE", "lat": 49.5720, "lon": 15.0800, "elev": 534.0, "heights": [10.0, 50.0, 125.0]},
    "kit": {"id": "KIT", "lat": 49.0915, "lon": 8.4249, "elev": 110.0, "heights": [30.0, 60.0, 100.0, 200.0]},
}


@click.command()
@click.option("--raw-dir", type=click.Path(path_type=Path), default=REPO / "data" / "raw")
@click.option("--output", type=click.Path(path_type=Path),
              default=REPO / "data" / "inference" / "icos_oos_towers_v1.parquet")
def main(raw_dir: Path, output: Path) -> None:
    bmp.ICOS_META = OOS_META
    df = bmp.build_icos(raw_dir)
    df["pop"] = "tower_icos_oos"
    ok = df.speed_obs.notna() & df.u_obs.notna() & (df.speed_obs > 0) & (df.speed_obs <= bmp.SPEED_MAX)
    df = df[ok].reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    click.echo(f"wrote {output}: {len(df)} rows")
    click.echo(df.groupby("station_id").speed_obs.agg(["size", "mean"]).round(2).to_string())


if __name__ == "__main__":
    main()
