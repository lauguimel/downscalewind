"""strata.py — Stratification helpers for OBS vs surrogate audit (M_G8).

Functions return small categorical columns appended to a copy of the input
DataFrame (no mutation of caller's frame). All helpers are stateless and
deterministic.

Strata definitions (per the M_G8 mandate slice):
- class_topo     : elev-based (plain / foothill / mountain / summit)
- height_bucket  : nearest in {10, 20, 50, 100} m AGL based on `height_obs`
- wind_class     : low (<3) / mid (3-7) / high (>7) m/s on `speed_obs`
- season         : DJF / MAM / JJA / SON via timestamp month
- era5_freshness : on_time / interpolated / far via |era5_time_delta_minutes|
- source         : passthrough (already in parquet)

`pairing_metrics` computes per-strata (or global) error stats:
  N, MAE, RMSE, bias, p10_ratio, p90_ratio, slope, intercept, R^2
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


# ─── Categorical assignment ────────────────────────────────────────────────

CLASS_TOPO_EDGES = [
    ("plain",    -1e6, 300.0),
    ("foothill",  300.0, 800.0),
    ("mountain",  800.0, 1500.0),
    ("summit",    1500.0, 1e6),
]

HEIGHT_BUCKETS = np.array([10.0, 20.0, 50.0, 100.0], dtype=np.float32)


def class_topo(elev: np.ndarray | pd.Series) -> np.ndarray:
    """elev (m ASL) → categorical class."""
    e = np.asarray(elev, dtype=np.float32)
    out = np.full(e.shape, "unknown", dtype=object)
    for name, lo, hi in CLASS_TOPO_EDGES:
        out[(e >= lo) & (e < hi)] = name
    return out


def height_bucket(height_obs: np.ndarray | pd.Series) -> np.ndarray:
    """Snap each height to the closest of {10, 20, 50, 100} m."""
    h = np.asarray(height_obs, dtype=np.float32)
    idx = np.argmin(np.abs(h[:, None] - HEIGHT_BUCKETS[None, :]), axis=1)
    return HEIGHT_BUCKETS[idx].astype(np.int32)


def wind_class(speed: np.ndarray | pd.Series) -> np.ndarray:
    s = np.asarray(speed, dtype=np.float32)
    out = np.full(s.shape, "low", dtype=object)
    out[s >= 3.0] = "mid"
    out[s >= 7.0] = "high"
    out[~np.isfinite(s)] = "unknown"
    return out


def season_from_ts(ts: pd.Series) -> np.ndarray:
    """timestamp → DJF/MAM/JJA/SON."""
    t = pd.to_datetime(ts, utc=True, errors="coerce")
    m = t.dt.month.to_numpy()
    out = np.full(m.shape, "unknown", dtype=object)
    out[np.isin(m, [12, 1, 2])] = "winter"
    out[np.isin(m, [3, 4, 5])] = "spring"
    out[np.isin(m, [6, 7, 8])] = "summer"
    out[np.isin(m, [9, 10, 11])] = "autumn"
    return out


def era5_freshness(delta_min: np.ndarray | pd.Series) -> np.ndarray:
    d = np.abs(np.asarray(delta_min, dtype=np.float32))
    out = np.full(d.shape, "far", dtype=object)
    out[d < 180.0] = "interpolated"
    out[d < 30.0] = "on_time"
    return out


def add_strata(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with strata columns added.

    Required input columns:
      elev, height_obs, speed_obs, timestamp, era5_time_delta_minutes
    """
    out = df.copy()
    out["class_topo"] = class_topo(out["elev"].to_numpy())
    out["height_bucket"] = height_bucket(out["height_obs"].to_numpy())
    out["wind_class"] = wind_class(out["speed_obs"].to_numpy())
    out["season"] = season_from_ts(out["timestamp"])
    out["era5_freshness"] = era5_freshness(out["era5_time_delta_minutes"].to_numpy())
    return out


# ─── Metric helpers ────────────────────────────────────────────────────────

def _affine_fit(obs: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    """Least-squares fit `pred = a * obs + b`, returning (a, b, R^2)."""
    obs = np.asarray(obs, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    m = np.isfinite(obs) & np.isfinite(pred)
    if m.sum() < 2:
        return float("nan"), float("nan"), float("nan")
    o, p = obs[m], pred[m]
    if np.std(o) < 1e-9:
        return float("nan"), float(np.mean(p)), float("nan")
    a, b = np.polyfit(o, p, 1)
    p_hat = a * o + b
    ss_res = np.sum((p - p_hat) ** 2)
    ss_tot = np.sum((p - np.mean(p)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(a), float(b), float(r2)


def pairing_metrics(
    obs: np.ndarray, pred: np.ndarray, *, ratio_clip: float = 1e-3,
) -> dict[str, float]:
    """Return all per-strata stats for one (obs, pred) sample.

    ratio_clip avoids division-by-zero spikes in low-wind buckets.
    """
    o = np.asarray(obs, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    m = np.isfinite(o) & np.isfinite(p)
    n = int(m.sum())
    out: dict[str, float] = {
        "N": n,
        "MAE": float("nan"),
        "RMSE": float("nan"),
        "bias": float("nan"),
        "p10_ratio": float("nan"),
        "p90_ratio": float("nan"),
        "slope": float("nan"),
        "intercept": float("nan"),
        "R2": float("nan"),
    }
    if n == 0:
        return out
    o = o[m]
    p = p[m]
    err = p - o
    out["MAE"] = float(np.mean(np.abs(err)))
    out["RMSE"] = float(np.sqrt(np.mean(err ** 2)))
    out["bias"] = float(np.mean(err))
    o_safe = np.where(np.abs(o) < ratio_clip, np.nan, o)
    ratio = p / o_safe
    finite = np.isfinite(ratio)
    if finite.any():
        out["p10_ratio"] = float(np.quantile(ratio[finite], 0.10))
        out["p90_ratio"] = float(np.quantile(ratio[finite], 0.90))
    a, b, r2 = _affine_fit(o, p)
    out["slope"] = a
    out["intercept"] = b
    out["R2"] = r2
    return out


def metrics_by_strata(
    df: pd.DataFrame,
    obs_col: str,
    pred_col: str,
    by: Sequence[str] | str,
) -> pd.DataFrame:
    """Group `df` by `by` columns and compute pairing_metrics per group.

    Returns a DataFrame with one row per group, the `by` columns first.
    """
    if isinstance(by, str):
        by = [by]
    rows: list[dict] = []
    for keys, grp in df.groupby(list(by), dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        m = pairing_metrics(grp[obs_col].to_numpy(), grp[pred_col].to_numpy())
        row: dict = dict(zip(by, keys))
        row.update(m)
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=list(by) + [
            "N", "MAE", "RMSE", "bias", "p10_ratio", "p90_ratio",
            "slope", "intercept", "R2",
        ])
    out = pd.DataFrame(rows)
    return out.sort_values(list(by)).reset_index(drop=True)
