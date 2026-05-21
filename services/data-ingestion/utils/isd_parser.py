"""NOAA ISD-Lite cache, station filtering, and parser helpers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from .obs_zarr_writer import wind_to_uv

ISD_HISTORY_URL = "https://www1.ncdc.noaa.gov/pub/data/noaa/isd-history.csv"
ISD_HISTORY_URLS = [ISD_HISTORY_URL, "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"]
ISD_LITE_BASE_URL = "https://www1.ncdc.noaa.gov/pub/data/noaa/isd-lite"
ISD_LITE_BASE_URLS = [ISD_LITE_BASE_URL, "https://www.ncei.noaa.gov/pub/data/noaa/isd-lite"]
NOAA_COUNTRY_TO_ISO = {"FR": "FR", "SP": "ES", "PO": "PT"}
ISO_OR_NOAA_TO_NOAA = {"FR": "FR", "ES": "SP", "PT": "PO", "SP": "SP", "PO": "PO"}
ISD_LITE_COLUMNS = ["year", "mon", "day", "hour", "t", "td", "slp", "wdir", "wspd", "sky", "p1", "p6"]


class NoaaDownloadError(RuntimeError):
    """Raised for NOAA download failures that are not a cacheable 404."""


def fetch_isd_history(cache_dir: str | Path) -> pd.DataFrame:
    """Fetch and parse the NOAA ISD station history CSV using a disk cache."""
    path = Path(cache_dir) / "isd-history.csv"
    if not path.exists() or path.stat().st_size == 0:
        _download_first_available(ISD_HISTORY_URLS, path, allow_404=False)
    history = pd.read_csv(path, dtype=str, keep_default_na=False)
    history.columns = [str(col).strip().replace(" ", "_") for col in history.columns]
    return history


def filter_stations(
    history_df: pd.DataFrame,
    countries: Iterable[str],
    start_date: object,
    end_date: object,
) -> pd.DataFrame:
    """Filter active ISD stations for France, Spain, and Portugal.

    ``countries`` may be ISO codes (FR/ES/PT) or NOAA CTRY codes (FR/SP/PO).
    Returned ``country`` values are ISO two-letter codes for the unified OBS
    schema.
    """
    required = {"USAF", "WBAN", "STATION_NAME", "CTRY", "ICAO", "LAT", "LON", "ELEV(M)", "BEGIN", "END"}
    missing = sorted(required.difference(history_df.columns))
    if missing:
        raise ValueError(f"isd-history.csv missing required columns: {missing}")

    wanted_noaa = _normalize_countries(countries)
    start_ymd = _yyyymmdd_int(start_date)
    end_ymd = _yyyymmdd_int(end_date)

    hist = history_df.copy()
    for col in ("USAF", "WBAN", "CTRY", "STATION_NAME", "ICAO"):
        hist[col] = hist[col].astype(str).str.strip()
    begin = pd.to_numeric(hist["BEGIN"], errors="coerce")
    end = pd.to_numeric(hist["END"], errors="coerce")

    mask = hist["CTRY"].isin(wanted_noaa) & begin.notna() & end.notna()
    mask &= (begin.astype("int64") <= start_ymd) & (end.astype("int64") >= end_ymd)
    stations = hist.loc[mask].copy()
    if stations.empty:
        return _empty_station_frame()

    stations["usaf"] = stations["USAF"].map(lambda value: _clean_station_code(value, width=6))
    stations["wban"] = stations["WBAN"].map(lambda value: _clean_station_code(value, width=5))
    stations["lat"] = pd.to_numeric(stations["LAT"], errors="coerce")
    stations["lon"] = pd.to_numeric(stations["LON"], errors="coerce")
    stations["elev"] = pd.to_numeric(stations["ELEV(M)"], errors="coerce")
    stations["begin"] = begin.loc[stations.index].astype("int64")
    stations["end"] = end.loc[stations.index].astype("int64")
    stations = stations.loc[
        stations["usaf"].ne("")
        & stations["wban"].ne("")
        & np.isfinite(stations["lat"])
        & np.isfinite(stations["lon"])
    ].copy()
    stations = stations.drop_duplicates(subset=["usaf", "wban"], keep="first")
    if stations.empty:
        return _empty_station_frame()

    stations["station_id"] = [_station_id(usaf, wban) for usaf, wban in zip(stations["usaf"], stations["wban"])]
    stations["country"] = stations["CTRY"].map(NOAA_COUNTRY_TO_ISO)
    stations["station_name"] = stations["STATION_NAME"].astype(str).str.strip()
    stations["icao"] = stations["ICAO"].astype(str).str.strip()
    stations = stations.sort_values(["country", "usaf", "wban"]).reset_index(drop=True)
    return stations[
        ["usaf", "wban", "station_id", "country", "lat", "lon", "elev", "station_name", "icao", "begin", "end"]
    ]


def fetch_isd_lite_year(usaf: str, wban: str, year: int, cache_dir: str | Path) -> Path | None:
    """Fetch a station-year ISD-Lite gzip file into the cache.

    Missing files are negative-cached with a ``.missing`` marker to avoid
    repeated 404 requests in smoke and reruns.
    """
    year = int(year)
    filename = f"{usaf}-{wban}-{year}.gz"
    path = Path(cache_dir) / "isd-lite" / str(year) / filename
    missing_marker = path.with_suffix(path.suffix + ".missing")
    if path.exists() and path.stat().st_size > 0:
        return path
    if missing_marker.exists():
        return None

    urls = [f"{base}/{year}/{filename}" for base in ISD_LITE_BASE_URLS]
    ok = _download_first_available(urls, path, allow_404=True)
    if not ok:
        missing_marker.parent.mkdir(parents=True, exist_ok=True)
        missing_marker.write_text("\n".join(urls) + "\n", encoding="utf-8")
        return None
    return path


def parse_isd_lite_gz(path: str | Path) -> pd.DataFrame:
    """Parse an ISD-Lite gzip file into unified OBS variables."""
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        compression="gzip",
        na_values=["-9999"],
    )
    if df.empty:
        return _empty_obs_frame()
    if df.shape[1] < 9:
        raise ValueError(f"ISD-Lite file has {df.shape[1]} columns; expected at least 9")
    df = df.iloc[:, : len(ISD_LITE_COLUMNS)].copy()
    present_cols = ISD_LITE_COLUMNS[: df.shape[1]]
    df.columns = present_cols
    for col in ISD_LITE_COLUMNS[len(present_cols) :]:
        df[col] = np.nan

    parts = pd.DataFrame(
        {
            "year": pd.to_numeric(df["year"], errors="coerce"),
            "month": pd.to_numeric(df["mon"], errors="coerce"),
            "day": pd.to_numeric(df["day"], errors="coerce"),
            "hour": pd.to_numeric(df["hour"], errors="coerce"),
        }
    )
    time_utc = pd.to_datetime(parts, errors="coerce", utc=True)
    valid_time = ~time_utc.isna()

    t_c = pd.to_numeric(df["t"], errors="coerce") / 10.0
    td_c = pd.to_numeric(df["td"], errors="coerce") / 10.0
    wind_speed = pd.to_numeric(df["wspd"], errors="coerce") / 10.0
    wind_dir = pd.to_numeric(df["wdir"], errors="coerce")
    wind_speed = wind_speed.mask(wind_speed < 0)
    wind_dir = wind_dir.mask((wind_dir < 0) | (wind_dir > 360))
    t2m = t_c + 273.15
    rh = magnus_rh(t_c.to_numpy(dtype=np.float64), td_c.to_numpy(dtype=np.float64))
    u, v = wind_to_uv(
        wind_speed.to_numpy(dtype=np.float32),
        wind_dir.to_numpy(dtype=np.float32),
    )

    out = pd.DataFrame(
        {
            "time": time_utc.dt.tz_localize(None),
            "wind_speed": wind_speed.to_numpy(dtype=np.float32),
            "wind_dir": wind_dir.to_numpy(dtype=np.float32),
            "u": u,
            "v": v,
            "t2m": t2m.to_numpy(dtype=np.float32),
            "rh": rh,
        }
    ).loc[valid_time]
    if out.empty:
        return _empty_obs_frame()
    out = out.sort_values("time")
    return out.groupby("time", as_index=False).mean(numeric_only=True).astype(
        {col: np.float32 for col in ("wind_speed", "wind_dir", "u", "v", "t2m", "rh")}
    )


def magnus_rh(t_c: np.ndarray, td_c: np.ndarray) -> np.ndarray:
    """Relative humidity in percent from air and dew-point temperature in degC."""
    t = np.asarray(t_c, dtype=np.float64)
    td = np.asarray(td_c, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        es_t = 6.112 * np.exp(17.67 * t / (t + 243.5))
        e_td = 6.112 * np.exp(17.67 * td / (td + 243.5))
        rh = 100.0 * e_td / es_t
    rh[~np.isfinite(rh)] = np.nan
    return np.clip(rh, 0.0, 100.0).astype(np.float32)


def _download_to_cache(url: str, path: Path, *, allow_404: bool) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    last_error: str | None = None
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=60.0, stream=True)
            status = int(response.status_code)
            if status == 404 and allow_404:
                return False
            if status == 200:
                with tmp.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                tmp.replace(path)
                return True
            last_error = f"HTTP {status}"
            if status not in {429, 500, 502, 503, 504}:
                break
        except requests.RequestException as exc:
            last_error = str(exc)
        finally:
            try:
                response.close()  # type: ignore[name-defined]
            except Exception:
                pass
        if attempt < 2:
            time.sleep(2 ** (attempt + 1))
    try:
        tmp.unlink()
    except FileNotFoundError:
        pass
    raise NoaaDownloadError(f"NOAA download failed for {url}: {last_error}")


def _download_first_available(urls: Iterable[str], path: Path, *, allow_404: bool) -> bool:
    saw_404 = False
    last_error: NoaaDownloadError | None = None
    for url in urls:
        try:
            ok = _download_to_cache(url, path, allow_404=allow_404)
        except NoaaDownloadError as exc:
            last_error = exc
            continue
        if ok:
            return True
        saw_404 = True
    if saw_404 and allow_404:
        return False
    if last_error is not None:
        raise last_error
    return False


def _normalize_countries(countries: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for value in countries:
        code = str(value).strip().upper()
        if not code:
            continue
        if code not in ISO_OR_NOAA_TO_NOAA:
            raise ValueError(f"unsupported country code {value!r}; expected FR, ES, PT or FR, SP, PO")
        noaa_code = ISO_OR_NOAA_TO_NOAA[code]
        if noaa_code not in normalized:
            normalized.append(noaa_code)
    return normalized


def _yyyymmdd_int(value: object) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    ts = pd.Timestamp(value)
    return int(ts.strftime("%Y%m%d"))


def _clean_station_code(value: object, *, width: int) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if text.isdigit():
        return text.zfill(width)
    return text


def _station_id(usaf: str, wban: str) -> str:
    candidate = f"isd_{usaf}_{wban}"
    if len(candidate) <= 16:
        return candidate
    return f"isd_{usaf}{wban[-4:]}"[:16]


def _empty_station_frame() -> pd.DataFrame:
    cols = ["usaf", "wban", "station_id", "country", "lat", "lon", "elev", "station_name", "icao", "begin", "end"]
    return pd.DataFrame(columns=cols)


def _empty_obs_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["time", "wind_speed", "wind_dir", "u", "v", "t2m", "rh"])
