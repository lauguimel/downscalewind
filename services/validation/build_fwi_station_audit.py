"""
Build a station-first FWI validation audit and an inference manifest.

The goal is to freeze where/when before scoring the downscaling model.  This
script uses independent SYNOP observations to select stations and fire-weather
days, then writes the station and timestamp manifests used by the UVT/rain/FWI
validation pipeline.

Inputs are downloaded from public Meteo-France/data.gouv endpoints and cached
under data/raw/mf_synop by default.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from shared.fwi import compute_fwi_series  # noqa: E402


STATIONS_URL = (
    "https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/OBS/SYNOP/"
    "postes_synop.geojson"
)
SYNOP_ARCHIVE_URL = (
    "https://donneespubliques.meteofrance.fr/donnees_libres/Txt/Synop/Archive/"
    "synop.{yyyymm}.csv.gz"
)

METRO_BBOX = (-6.5, 41.0, 10.5, 52.5)  # lon_min, lat_min, lon_max, lat_max
SYNOP_USECOLS = {"numer_sta", "date", "dd", "ff", "t", "td", "u", "rr24", "pres"}
NA_VALUES = ["mq", "", " ", "nan", "NaN"]


@dataclass(frozen=True)
class AuditConfig:
    year: int
    months: tuple[int, ...]
    score_months: tuple[int, ...]
    target_hour_utc: int
    max_hour_distance: int
    bbox: tuple[float, float, float, float]
    min_score_coverage: float
    max_sites: int
    max_timestamps_per_site: int
    max_per_region: int


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parse_bbox(value: str) -> tuple[float, float, float, float]:
    parts = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be lon_min,lat_min,lon_max,lat_max")
    return parts


def download_if_needed(url: str, path: Path, *, force: bool = False) -> Path:
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def load_synop_stations(cache_dir: Path, *, force: bool = False) -> pd.DataFrame:
    path = download_if_needed(STATIONS_URL, cache_dir / "postes_synop.geojson", force=force)
    payload = json.loads(path.read_text())
    rows = []
    for feature in payload.get("features", []):
        props = feature.get("properties", {})
        coords = feature.get("geometry", {}).get("coordinates", [np.nan, np.nan])
        station_id = str(props.get("Id", "")).strip()
        if not station_id:
            continue
        rows.append(
            {
                "station_id": f"MF_{int(station_id):05d}",
                "wmo_id": int(station_id),
                "name": str(props.get("Nom", "")).strip(),
                "lon": float(coords[0]),
                "lat": float(coords[1]),
                "alt_m": float(props.get("Altitude", np.nan)),
                "open_date": props.get("Date_ouverture"),
            }
        )
    return pd.DataFrame(rows).drop_duplicates("station_id")


def load_synop_month(year: int, month: int, cache_dir: Path, *, force: bool = False) -> pd.DataFrame:
    yyyymm = f"{year}{month:02d}"
    path = download_if_needed(
        SYNOP_ARCHIVE_URL.format(yyyymm=yyyymm),
        cache_dir / f"synop.{yyyymm}.csv.gz",
        force=force,
    )
    content = path.read_bytes()
    with gzip.GzipFile(fileobj=io.BytesIO(content)) as gz:
        df = pd.read_csv(
            gz,
            sep=";",
            usecols=lambda c: c in SYNOP_USECOLS,
            na_values=NA_VALUES,
            low_memory=False,
        )
    return df


def load_synop_observations(config: AuditConfig, cache_dir: Path, *, force: bool = False) -> pd.DataFrame:
    frames = [load_synop_month(config.year, m, cache_dir, force=force) for m in config.months]
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    raw["date_time"] = pd.to_datetime(raw["date"], format="%Y%m%d%H%M%S", errors="coerce")
    raw = raw.dropna(subset=["date_time", "numer_sta"]).copy()
    raw["wmo_id"] = pd.to_numeric(raw["numer_sta"], errors="coerce").astype("Int64")
    raw["station_id"] = raw["wmo_id"].map(lambda x: f"MF_{int(x):05d}" if pd.notna(x) else None)
    for col in ["dd", "ff", "t", "td", "u", "rr24", "pres"]:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
    return raw


def dewpoint_to_rh(t_c: np.ndarray, td_c: np.ndarray) -> np.ndarray:
    rh = 100.0 * np.exp(17.625 * td_c / (243.04 + td_c)) / np.exp(17.625 * t_c / (243.04 + t_c))
    return np.clip(rh, 0.0, 100.0)


def nearest_daily_noon(raw: pd.DataFrame, config: AuditConfig) -> pd.DataFrame:
    df = raw.copy()
    df["date"] = df["date_time"].dt.floor("D")
    df["hour_utc"] = df["date_time"].dt.hour
    df["hour_distance"] = (df["hour_utc"] - config.target_hour_utc).abs()
    df = df[df["hour_distance"] <= config.max_hour_distance]
    df = df.sort_values(["station_id", "date", "hour_distance", "date_time"])
    daily = df.groupby(["station_id", "date"], as_index=False).first()

    daily["t_obs_c"] = daily["t"] - 273.15
    daily["rh_obs_pct"] = daily["u"]
    missing_rh = daily["rh_obs_pct"].isna() & daily["td"].notna() & daily["t"].notna()
    if missing_rh.any():
        daily.loc[missing_rh, "rh_obs_pct"] = dewpoint_to_rh(
            daily.loc[missing_rh, "t_obs_c"].to_numpy(),
            (daily.loc[missing_rh, "td"] - 273.15).to_numpy(),
        )
    daily["wind_obs_ms"] = daily["ff"]
    daily["wind_dir_obs_deg"] = daily["dd"]
    daily["rain24_obs_mm"] = daily["rr24"].clip(lower=0.0)

    valid = (
        daily["t_obs_c"].between(-40.0, 55.0)
        & daily["rh_obs_pct"].between(0.0, 100.0)
        & daily["wind_obs_ms"].between(0.0, 70.0)
        & daily["rain24_obs_mm"].between(0.0, 500.0)
    )
    daily = daily[valid].copy()
    daily["month"] = daily["date"].dt.month
    daily["ws_obs_kmh"] = daily["wind_obs_ms"] * 3.6
    return daily


def compute_obs_fwi(daily: pd.DataFrame) -> pd.DataFrame:
    outputs = []
    for station_id, group in daily.sort_values(["station_id", "date"]).groupby("station_id"):
        g = group.copy()
        with np.errstate(all="ignore"):
            fwi_out = compute_fwi_series(
                t_c=g["t_obs_c"].to_numpy(),
                rh=g["rh_obs_pct"].to_numpy(),
                ws_kmh=g["ws_obs_kmh"].to_numpy(),
                rain_mm=g["rain24_obs_mm"].to_numpy(),
                months=g["month"].to_numpy(),
            )
        for key, values in fwi_out.items():
            g[f"{key}_obs"] = values
        outputs.append(g)
    if not outputs:
        return pd.DataFrame()
    out = pd.concat(outputs, ignore_index=True)
    return out[np.isfinite(out["fwi_obs"])].copy()


def region_tag(lat: float, lon: float) -> str:
    if 41.0 <= lat <= 43.5 and 8.0 <= lon <= 10.5:
        return "corsica"
    if 41.0 <= lat <= 45.5 and 2.0 <= lon <= 8.5:
        return "med_fr"
    if 42.0 <= lat <= 45.5 and -2.5 <= lon < 2.0:
        return "pyrenees_sw"
    if 44.0 <= lat <= 47.5 and 5.0 <= lon <= 8.5:
        return "alps"
    if lon <= -0.5 and lat >= 43.0:
        return "atlantic"
    if lat >= 45.5:
        return "north_control"
    return "other"


def build_station_inventory(
    fwi_daily: pd.DataFrame,
    stations: pd.DataFrame,
    config: AuditConfig,
) -> pd.DataFrame:
    score = fwi_daily[fwi_daily["month"].isin(config.score_months)].copy()
    if score.empty:
        return pd.DataFrame()

    n_score_days_possible = sum(
        pd.Period(f"{config.year}-{month:02d}").days_in_month for month in config.score_months
    )
    grouped = score.groupby("station_id")
    inv = grouped.agg(
        n_days_score=("date", "nunique"),
        max_fwi_obs=("fwi_obs", "max"),
        p95_fwi_obs=("fwi_obs", lambda x: float(np.nanpercentile(x, 95))),
        mean_fwi_obs=("fwi_obs", "mean"),
        n_fwi_gt_12=("fwi_obs", lambda x: int((x > 12.0).sum())),
        n_fwi_gt_21=("fwi_obs", lambda x: int((x > 21.0).sum())),
        n_fwi_gt_38=("fwi_obs", lambda x: int((x > 38.0).sum())),
        max_wind_obs_ms=("wind_obs_ms", "max"),
        max_t_obs_c=("t_obs_c", "max"),
        rain_score_sum_mm=("rain24_obs_mm", "sum"),
        dry_day_fraction=("rain24_obs_mm", lambda x: float((x <= 0.5).mean())),
    ).reset_index()
    inv["score_coverage"] = inv["n_days_score"] / float(n_score_days_possible)
    inv = inv.merge(stations, on="station_id", how="left")
    lon_min, lat_min, lon_max, lat_max = config.bbox
    inv = inv[inv["lat"].between(lat_min, lat_max) & inv["lon"].between(lon_min, lon_max)].copy()
    inv["region"] = [region_tag(lat, lon) for lat, lon in zip(inv["lat"], inv["lon"])]
    inv["candidate_score"] = (
        inv["p95_fwi_obs"]
        + 1.5 * inv["n_fwi_gt_21"]
        + 4.0 * inv["n_fwi_gt_38"]
        + 2.0 * inv["dry_day_fraction"]
        + 0.25 * inv["max_wind_obs_ms"]
    )
    inv["passes_qc"] = inv["score_coverage"] >= config.min_score_coverage
    return inv.sort_values(["passes_qc", "candidate_score"], ascending=[False, False])


def select_stations(inventory: pd.DataFrame, config: AuditConfig) -> pd.DataFrame:
    candidates = inventory[inventory["passes_qc"]].copy()
    candidates = candidates.sort_values("candidate_score", ascending=False)
    selected_rows = []
    region_counts: dict[str, int] = {}
    for _, row in candidates.iterrows():
        region = str(row["region"])
        if region_counts.get(region, 0) >= config.max_per_region:
            continue
        selected_rows.append(row)
        region_counts[region] = region_counts.get(region, 0) + 1
        if len(selected_rows) >= config.max_sites:
            break
    if not selected_rows:
        return pd.DataFrame(columns=candidates.columns)
    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    selected.insert(0, "selection_rank", np.arange(1, len(selected) + 1))
    return selected


def build_inference_manifest(
    fwi_daily: pd.DataFrame,
    selected: pd.DataFrame,
    config: AuditConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_ids = set(selected["station_id"].astype(str))
    scored = fwi_daily[
        fwi_daily["station_id"].isin(selected_ids) & fwi_daily["month"].isin(config.score_months)
    ].copy()
    meta_cols = [
        "selection_rank",
        "station_id",
        "wmo_id",
        "name",
        "lat",
        "lon",
        "alt_m",
        "region",
        "candidate_score",
    ]
    validation_days = scored.merge(
        selected[meta_cols],
        on="station_id",
        how="left",
        suffixes=("", "_station"),
    )
    for col in meta_cols:
        station_col = f"{col}_station"
        if station_col in validation_days.columns:
            validation_days[col] = validation_days[station_col]
            validation_days = validation_days.drop(columns=[station_col])

    manifest_rows = []
    for station_id, group in validation_days.groupby("station_id"):
        g = group.sort_values(["fwi_obs", "date"], ascending=[False, True]).head(
            config.max_timestamps_per_site
        )
        manifest_rows.append(g)
    manifest = pd.concat(manifest_rows, ignore_index=True) if manifest_rows else pd.DataFrame()
    if not manifest.empty:
        manifest = manifest.sort_values(["selection_rank", "fwi_obs"], ascending=[True, False]).copy()
        manifest["timestamp_utc"] = manifest["date"] + pd.to_timedelta(config.target_hour_utc, unit="h")
        manifest["case_id"] = [
            f"mf_synop_{int(wmo):05d}_{ts.strftime('%Y%m%dT%H%MZ')}"
            for wmo, ts in zip(manifest["wmo_id"], manifest["timestamp_utc"])
        ]
        keep = [
            "case_id",
            "selection_rank",
            "station_id",
            "wmo_id",
            "name",
            "region",
            "lat",
            "lon",
            "alt_m",
            "date",
            "timestamp_utc",
            "hour_utc",
            "t_obs_c",
            "rh_obs_pct",
            "wind_obs_ms",
            "wind_dir_obs_deg",
            "rain24_obs_mm",
            "ffmc_obs",
            "dmc_obs",
            "dc_obs",
            "isi_obs",
            "bui_obs",
            "fwi_obs",
        ]
        manifest = manifest[keep]
    return validation_days, manifest


def write_frame(df: pd.DataFrame, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(csv_path.with_suffix(".parquet"), index=False)
    except Exception:
        pass


def write_report(
    output_dir: Path,
    config: AuditConfig,
    raw: pd.DataFrame,
    fwi_daily: pd.DataFrame,
    inventory: pd.DataFrame,
    selected: pd.DataFrame,
    manifest: pd.DataFrame,
) -> None:
    top_cols = [
        "selection_rank",
        "station_id",
        "name",
        "region",
        "lat",
        "lon",
        "alt_m",
        "n_days_score",
        "score_coverage",
        "p95_fwi_obs",
        "max_fwi_obs",
        "n_fwi_gt_21",
        "n_fwi_gt_38",
    ]
    lines = [
        "# FWI station audit",
        "",
        f"- Source: Meteo-France SYNOP OMM, {config.year}.",
        f"- Computation months: {','.join(map(str, config.months))}.",
        f"- Scored fire-season months: {','.join(map(str, config.score_months))}.",
        f"- Daily weather sample: nearest SYNOP to {config.target_hour_utc:02d}:00 UTC "
        f"(max distance {config.max_hour_distance} h).",
        f"- Raw SYNOP rows: {len(raw):,}.",
        f"- Valid station-days with full FWI inputs: {len(fwi_daily):,}.",
        f"- Stations in inventory after bbox: {len(inventory):,}.",
        f"- Selected stations n: {len(selected):,}.",
        f"- Timestamps per station m: {config.max_timestamps_per_site}.",
        f"- Inference cases n x m: {len(manifest):,}.",
        "",
        "## Selected stations",
        "",
    ]
    if selected.empty:
        lines.append("No station passed QC.")
    else:
        try:
            lines.append(selected[top_cols].to_markdown(index=False, floatfmt=".3f"))
        except Exception:
            lines.append("```csv")
            lines.append(selected[top_cols].to_csv(index=False))
            lines.append("```")
    lines.extend(
        [
            "",
            "## Baseline availability audit",
            "",
            "| Product | Status for this manifest | Action |",
            "|---|---|---|",
            "| OBS station FWI | Ready in `validation_station_days.*` | Use as validation target. |",
            "| ERA5 FWI | Input files exist locally only for limited 2022 months; station extraction still needed | Build station extractor from CDS/ERA5 files. |",
            "| ERA5-Land FWI | Daily GEE grid exists for 2022 in `data/raw/downscalrain_grids/gee_2022/era5land_daily.nc` | Sample stations and compute FWI. |",
            "| IMERG corrected rain24 | Gridded IMERG/ERA5-Land inputs exist for 2022; dry-period correction calibrated | Sample grid, apply IMERG-first fire correction. |",
            "| AROME/ICON-D2 | Not present locally; useful for forecast demos, historical station validation needs archive access | Treat as optional high-res NWP baseline. |",
            "| GWIS/EFFIS | Ingestion script exists; 2022 file not present locally | Download CDS/GWIS FWI for selected stations. |",
            "| DownscaleWind UVT | Final checkpoints are on Aqua, not local | Run/submit `n x m` inference on Aqua or rsync checkpoint. |",
            "",
            "## Frozen outputs",
            "",
            "- `station_inventory.*`: all candidate SYNOP stations with OBS FWI metrics.",
            "- `selected_stations.*`: frozen station list.",
            "- `validation_station_days.*`: all selected station-days in scored months.",
            "- `inference_manifest.*`: top OBS-FWI days for UVT/rain/FWI inference.",
        ]
    )
    (output_dir / "audit_report.md").write_text("\n".join(lines) + "\n")


def run(config: AuditConfig, cache_dir: Path, output_dir: Path, *, force_download: bool = False) -> None:
    stations = load_synop_stations(cache_dir, force=force_download)
    raw = load_synop_observations(config, cache_dir, force=force_download)
    daily = nearest_daily_noon(raw, config)
    fwi_daily = compute_obs_fwi(daily)
    inventory = build_station_inventory(fwi_daily, stations, config)
    selected = select_stations(inventory, config)
    validation_days, manifest = build_inference_manifest(fwi_daily, selected, config)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_frame(inventory, output_dir / "station_inventory.csv")
    write_frame(selected, output_dir / "selected_stations.csv")
    write_frame(validation_days, output_dir / "validation_station_days.csv")
    write_frame(manifest, output_dir / "inference_manifest.csv")
    write_report(output_dir, config, raw, fwi_daily, inventory, selected, manifest)

    print(f"station_inventory={len(inventory)}")
    print(f"selected_stations={len(selected)}")
    print(f"inference_cases={len(manifest)}")
    print(f"output_dir={output_dir}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--months", default="1,2,3,4,5,6,7,8,9,10,11,12")
    parser.add_argument("--score-months", default="6,7,8,9")
    parser.add_argument("--target-hour-utc", type=int, default=12)
    parser.add_argument("--max-hour-distance", type=int, default=1)
    parser.add_argument("--bbox", type=parse_bbox, default=METRO_BBOX)
    parser.add_argument("--min-score-coverage", type=float, default=0.75)
    parser.add_argument("--max-sites", type=int, default=12)
    parser.add_argument("--max-timestamps-per-site", type=int, default=20)
    parser.add_argument("--max-per-region", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "data/raw/mf_synop")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/validation/fwi_station_audit_2022",
    )
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = AuditConfig(
        year=args.year,
        months=parse_int_list(args.months),
        score_months=parse_int_list(args.score_months),
        target_hour_utc=args.target_hour_utc,
        max_hour_distance=args.max_hour_distance,
        bbox=args.bbox,
        min_score_coverage=args.min_score_coverage,
        max_sites=args.max_sites,
        max_timestamps_per_site=args.max_timestamps_per_site,
        max_per_region=args.max_per_region,
    )
    run(config, args.cache_dir, args.output_dir, force_download=args.force_download)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
