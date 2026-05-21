"""Parser for OGIMET decoded SYNOP HTML tables."""

from __future__ import annotations

import re
from io import StringIO

import numpy as np
import pandas as pd


CARDINAL_DEG = {
    "N": 0.0,
    "NNE": 22.5,
    "NE": 45.0,
    "ENE": 67.5,
    "E": 90.0,
    "ESE": 112.5,
    "SE": 135.0,
    "SSE": 157.5,
    "S": 180.0,
    "SSW": 202.5,
    "SW": 225.0,
    "WSW": 247.5,
    "W": 270.0,
    "WNW": 292.5,
    "NW": 315.0,
    "NNW": 337.5,
}


def parse_ogimet_html(html_text: str) -> pd.DataFrame:
    """Return columns: time, temp_c, dewpoint_c, wind_dir_deg, wind_speed_ms."""
    frames = _read_tables(html_text)
    errors: list[str] = []
    for frame in frames:
        try:
            parsed = _parse_frame(frame)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not parsed.empty:
            return parsed
    detail = "; ".join(errors[:3]) if errors else "no HTML tables found"
    raise ValueError(f"could not locate OGIMET decoded table: {detail}")


def relative_humidity_from_dewpoint(temp_c: np.ndarray, dewpoint_c: np.ndarray) -> np.ndarray:
    temp_c = temp_c.astype(np.float32)
    dewpoint_c = dewpoint_c.astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        es_t = 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5))
        es_td = 6.112 * np.exp(17.67 * dewpoint_c / (dewpoint_c + 243.5))
        rh = 100.0 * es_td / es_t
    return np.clip(rh, 0.0, 100.0).astype(np.float32)


def _read_tables(html_text: str) -> list[pd.DataFrame]:
    try:
        return pd.read_html(StringIO(html_text))
    except ValueError:
        return []


def _parse_frame(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df.columns = [_flat_col(c) for c in df.columns]
    df = df.dropna(how="all").reset_index(drop=True)
    if df.empty:
        raise ValueError("empty table")

    date_col = _find_col(df.columns, lambda n: "date" in n or n == "time")
    if date_col is None:
        raise ValueError("missing date/time column")
    time_col = _find_col(
        df.columns,
        lambda n: n in {"hour", "utc", "time"} or (("hour" in n or "time" in n) and "date" not in n),
        exclude={date_col},
    )
    times = _parse_times(df[date_col], df[time_col] if time_col else None)
    if times.notna().sum() == 0:
        raise ValueError("unparseable date/time column")

    temp_col = _find_col(
        df.columns,
        lambda n: ("temp" in n or n in {"t", "tc"}) and "dew" not in n and "max" not in n and "min" not in n,
    )
    dew_col = _find_col(df.columns, lambda n: "dew" in n or "td" in n)
    wspd_col = _find_col(df.columns, lambda n: "wind" in n and ("speed" in n or "ms" in n or "kt" in n or "kmh" in n))
    wdir_col = _find_col(df.columns, lambda n: "wind" in n and ("dir" in n or "direction" in n))
    wind_col = _find_col(df.columns, lambda n: "wind" in n)

    if temp_col is None and dew_col is None and wind_col is None:
        raise ValueError("table has date but no weather columns")

    out = pd.DataFrame({"time": times})
    out["temp_c"] = _series_float(df[temp_col]) if temp_col else np.nan
    out["dewpoint_c"] = _series_float(df[dew_col]) if dew_col else np.nan
    if wspd_col:
        out["wind_speed_ms"] = _series_float(df[wspd_col]) * _unit_factor(wspd_col)
    elif wind_col:
        out["wind_speed_ms"] = df[wind_col].map(_speed_from_wind_text).astype(np.float32)
    else:
        out["wind_speed_ms"] = np.nan
    if wdir_col:
        out["wind_dir_deg"] = df[wdir_col].map(_parse_direction).astype(np.float32)
    elif wind_col:
        out["wind_dir_deg"] = df[wind_col].map(_direction_from_wind_text).astype(np.float32)
    else:
        out["wind_dir_deg"] = np.nan

    out = out.dropna(subset=["time"]).sort_values("time")
    out = out.drop_duplicates(subset=["time"], keep="last")
    return out.reset_index(drop=True)


def _parse_times(date_values: pd.Series, hour_values: pd.Series | None) -> pd.Series:
    text = date_values.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    if hour_values is not None:
        hour = hour_values.astype(str).str.extract(r"(\d{1,2})(?::?(\d{2}))?", expand=True)
        hh = hour[0].fillna("").str.zfill(2)
        mm = hour[1].fillna("00").str.zfill(2)
        text = text + " " + hh + ":" + mm
    parsed = pd.to_datetime(text, errors="coerce", utc=True, dayfirst=True)
    return parsed.dt.tz_convert("UTC").dt.tz_localize(None)


def _series_float(series: pd.Series) -> pd.Series:
    return series.map(_parse_float).astype(np.float32)


def _parse_float(value) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip().replace(",", ".")
    if text in {"", "-", "--", "///", "nan", "None"}:
        return np.nan
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return np.nan
    val = float(match.group(0))
    return np.nan if val <= -98.0 else val


def _parse_direction(value) -> float:
    val = _parse_float(value)
    if np.isfinite(val):
        return val % 360.0
    token = re.sub(r"[^A-Za-z]", "", str(value)).upper()
    if token in {"VRB", "VAR", "CALM"}:
        return np.nan
    return CARDINAL_DEG.get(token, np.nan)


def _speed_from_wind_text(value) -> float:
    text = str(value)
    match = re.search(r"(?:/|\s)(\d+(?:[.,]\d+)?)\s*(kt|kts|m/s|ms|km/h|kmh)?", text, re.I)
    if not match:
        match = re.search(r"(\d+(?:[.,]\d+)?)\s*(kt|kts|m/s|ms|km/h|kmh)?", text, re.I)
    if not match:
        return np.nan
    speed = float(match.group(1).replace(",", "."))
    return speed * _unit_factor(match.group(2) or text)


def _direction_from_wind_text(value) -> float:
    text = str(value).strip()
    numeric = re.match(r"^\s*(\d{1,3})(?:\D|$)", text)
    if numeric:
        return float(numeric.group(1)) % 360.0
    cardinal = re.match(r"^\s*([NESW]{1,3})\b", text, re.I)
    if cardinal:
        return CARDINAL_DEG.get(cardinal.group(1).upper(), np.nan)
    return _parse_direction(text)


def _unit_factor(label: str) -> float:
    norm = _norm(label)
    if "kt" in norm or "knot" in norm:
        return 0.514444
    if "kmh" in norm or "kmhour" in norm:
        return 1.0 / 3.6
    return 1.0


def _find_col(columns, predicate, exclude: set[str] | None = None) -> str | None:
    exclude = exclude or set()
    for col in columns:
        if col in exclude:
            continue
        if predicate(_norm(col)):
            return col
    return None


def _flat_col(col) -> str:
    if isinstance(col, tuple):
        parts = [str(p) for p in col if p is not None and not str(p).startswith("Unnamed")]
        return " ".join(parts).strip()
    return str(col).strip()


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())
