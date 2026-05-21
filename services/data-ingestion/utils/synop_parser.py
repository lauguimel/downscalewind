"""Small parsing/cache helpers for Meteo-France SYNOP CSV ingestion."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from shared.logging_config import get_logger

log = get_logger("ingest_synop_meteofr")

SYNOP_ARCHIVE_URL = (
    "https://donneespubliques.meteofrance.fr/donnees_libres/Txt/Synop/"
    "Archive/synop.{yyyymm}.csv.gz"
)
POSTES_URL = (
    "https://donneespubliques.meteofrance.fr/donnees_libres/Txt/Synop/"
    "postesSynop.csv"
)
USER_AGENT = "downscalewind-ingest/0.1"
BBOX = {"lat_min": 41.0, "lat_max": 52.0, "lon_min": -5.0, "lon_max": 10.0}


@dataclass(frozen=True)
class Stations:
    ids: list[str]
    lat: np.ndarray
    lon: np.ndarray
    elev: np.ndarray


@dataclass(frozen=True)
class FetchResult:
    path: Path
    status: str
    size: int


def fetch_with_cache(url: str, path: Path) -> FetchResult:
    if path.exists() and path.stat().st_size > 0:
        return FetchResult(path, "cached", path.stat().st_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".part")
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(1, 4):
        try:
            log.info("Downloading source file", extra={
                "url": url, "path": str(path), "attempt": attempt,
            })
            with requests.get(url, headers=headers, timeout=60, stream=True) as resp:
                if resp.status_code == 404:
                    raise FileNotFoundError(f"404 Not Found: {url}")
                resp.raise_for_status()
                with tmp_path.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            size = tmp_path.stat().st_size
            if size <= 0:
                raise OSError(f"Downloaded empty file: {url}")
            tmp_path.replace(path)
            return FetchResult(path, "downloaded", size)
        except FileNotFoundError:
            if tmp_path.exists():
                tmp_path.unlink()
            raise
        except (OSError, requests.RequestException) as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt == 3:
                raise RuntimeError(f"Failed to download {url}: {exc}") from exc
            wait_s = 2 ** (attempt - 1)
            log.warning("Download failed, retrying", extra={
                "url": url, "attempt": attempt, "wait_s": wait_s, "error": str(exc),
            })
            time.sleep(wait_s)
    raise RuntimeError(f"Failed to download {url}")


def read_csv_with_fallback(path: Path, **kwargs: Any) -> pd.DataFrame:
    errors: list[str] = []
    for encoding in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise UnicodeDecodeError("utf-8", b"", 0, 1, "; ".join(errors))


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    return df


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    by_lower = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        col = by_lower.get(candidate.lower())
        if col is not None:
            return str(col)
    raise ValueError(f"Missing required column; tried {candidates}")


def _station_id(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(5) if text.isdigit() and len(text) < 5 else text


def _parse_float(value: Any) -> float:
    if pd.isna(value):
        return float("nan")
    text = str(value).strip().replace(",", ".")
    if not text or text.lower() == "mq":
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def load_mainland_stations(cache_dir: Path) -> tuple[Stations, FetchResult]:
    fetch = fetch_with_cache(POSTES_URL, cache_dir / "postesSynop.csv")
    df = _normalize_columns(read_csv_with_fallback(fetch.path, sep=";", dtype=str))
    id_col = _find_column(df, ("ID", "numer_sta", "station_id"))
    lat_col = _find_column(df, ("Latitude", "lat"))
    lon_col = _find_column(df, ("Longitude", "lon"))
    elev_col = _find_column(df, ("Altitude", "altitude", "elev"))
    out = pd.DataFrame({
        "id": df[id_col].map(_station_id),
        "lat": df[lat_col].map(_parse_float),
        "lon": df[lon_col].map(_parse_float),
        "elev": df[elev_col].map(_parse_float),
    }).dropna(subset=["id", "lat", "lon"])
    mask = (
        out["lat"].between(BBOX["lat_min"], BBOX["lat_max"])
        & out["lon"].between(BBOX["lon_min"], BBOX["lon_max"])
    )
    out = out.loc[mask].drop_duplicates("id").sort_values("id").reset_index(drop=True)
    if out.empty:
        raise ValueError("No mainland SYNOP stations after bbox filtering")
    stations = Stations(
        ids=out["id"].tolist(),
        lat=out["lat"].to_numpy(dtype=np.float32),
        lon=out["lon"].to_numpy(dtype=np.float32),
        elev=out["elev"].to_numpy(dtype=np.float32),
    )
    log.info("Loaded station list", extra={
        "status": fetch.status,
        "n_stations_mainland": len(stations.ids),
        "lat_bounds": [BBOX["lat_min"], BBOX["lat_max"]],
        "lon_bounds": [BBOX["lon_min"], BBOX["lon_max"]],
    })
    return stations, fetch


def load_month(path: Path, station_ids: set[str]) -> pd.DataFrame:
    df = read_csv_with_fallback(
        path, sep=";", na_values=["mq"],
        dtype={"numer_sta": str, "date": str}, low_memory=False,
    )
    df = _normalize_columns(df)
    required = ["numer_sta", "date", "ff", "dd", "t", "u"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    df = df[required].copy()
    df["numer_sta"] = df["numer_sta"].map(_station_id)
    df = df[df["numer_sta"].isin(station_ids)]
    if df.empty:
        return df
    for col in ("ff", "dd", "t", "u"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    date_text = df["date"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    df["timestamp"] = pd.to_datetime(
        date_text, format="%Y%m%d%H%M%S", utc=True, errors="coerce",
    )
    df = df.dropna(subset=["timestamp"])
    return df[["numer_sta", "timestamp", "ff", "dd", "t", "u"]]
