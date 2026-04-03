"""
quantile_correction.py — Stratified quantile mapping for IMERG precipitation.

3-step bias correction:
  1. Drizzle threshold: IMERG < threshold → 0 (fixes false positive rate)
  2. Stratified Empirical QM: match IMERG CDF to station CDF per stratum
     (season × elevation band × climate zone)
  3. Terrain residual: XGBoost on residuals after QM (orographic fine-tuning)

Reference:
  Baez-Villanueva et al. (2020). RF-MEP for merging satellite precipitation.
  Derin et al. (2019). Evaluation of IMERG over complex terrain.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Stratification ───────────────────────────────────────────────────────────

SEASONS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}

ELEVATION_BANDS = [
    (0, 500, "low"),
    (500, 1500, "mid"),
    (1500, 9000, "high"),
]

# Simplified climate zones based on lat/lon (Mediterranean vs Atlantic vs Continental)
def _classify_climate(lat: float, lon: float) -> str:
    """Simple climate classification for Europe."""
    if lat < 44 and lon > -2:
        return "mediterranean"
    elif lat < 44 and lon <= -2:
        return "atlantic_south"
    elif lat < 50 and lon <= 5:
        return "atlantic_north"
    elif lat < 50 and lon > 5:
        return "continental"
    else:
        return "northern"


def get_stratum(month: int, elevation: float, lat: float, lon: float) -> str:
    """Assign a stratum key for stratified QM."""
    # Season
    season = "DJF"
    for s, months in SEASONS.items():
        if month in months:
            season = s
            break

    # Elevation band
    elev_band = "low"
    for lo, hi, band in ELEVATION_BANDS:
        if lo <= elevation < hi:
            elev_band = band
            break

    # Climate
    climate = _classify_climate(lat, lon)

    return f"{season}_{elev_band}_{climate}"


# ── Step 1: Drizzle threshold ────────────────────────────────────────────────

def apply_drizzle_threshold(rain: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Set IMERG values below threshold to 0 (fixes false positives)."""
    return np.where(rain < threshold, 0.0, rain)


# ── Step 2: Empirical Quantile Mapping ───────────────────────────────────────

@dataclass
class QMTransfer:
    """Stores the empirical CDF for one stratum."""
    quantiles: np.ndarray       # probability levels (0-1)
    imerg_values: np.ndarray    # IMERG quantile values
    station_values: np.ndarray  # station quantile values
    n_samples: int


def fit_qm_stratum(imerg: np.ndarray, station: np.ndarray,
                   n_quantiles: int = 100) -> QMTransfer:
    """Fit empirical QM for one stratum."""
    valid = np.isfinite(imerg) & np.isfinite(station)
    imerg_v = imerg[valid]
    station_v = station[valid]

    quantiles = np.linspace(0, 1, n_quantiles + 1)
    imerg_q = np.quantile(imerg_v, quantiles)
    station_q = np.quantile(station_v, quantiles)

    return QMTransfer(
        quantiles=quantiles,
        imerg_values=imerg_q,
        station_values=station_q,
        n_samples=int(valid.sum()),
    )


def apply_qm(rain: np.ndarray, transfer: QMTransfer) -> np.ndarray:
    """Apply empirical QM: map IMERG values through transfer function."""
    return np.interp(rain, transfer.imerg_values, transfer.station_values)


# ── Main class ───────────────────────────────────────────────────────────────

class StratifiedQMCorrector:
    """Stratified quantile mapping corrector for IMERG precipitation.

    Fits per-stratum QM transfer functions from paired IMERG-station data.
    """

    def __init__(self, drizzle_threshold: float = 0.5, n_quantiles: int = 100):
        self.drizzle_threshold = drizzle_threshold
        self.n_quantiles = n_quantiles
        self.transfers: dict[str, QMTransfer] = {}
        self.global_transfer: QMTransfer | None = None

    def fit(self, df: pd.DataFrame) -> None:
        """Fit QM from paired IMERG-station DataFrame.

        Required columns: rain_imerg, rain_station, month, elevation, lat, lon.
        """
        # Assign strata
        df = df.copy()
        df["stratum"] = df.apply(
            lambda r: get_stratum(r["month"], r["elevation"], r["lat"], r["lon"]),
            axis=1,
        )

        # Apply drizzle threshold to IMERG before fitting
        imerg_clean = apply_drizzle_threshold(df["rain_imerg"].values, self.drizzle_threshold)

        # Fit global (fallback)
        self.global_transfer = fit_qm_stratum(
            imerg_clean, df["rain_station"].values, self.n_quantiles
        )
        log.info("Global QM fitted on %d samples", self.global_transfer.n_samples)

        # Fit per stratum
        for stratum, group in df.groupby("stratum"):
            if len(group) < 50:  # need minimum samples for reliable QM
                log.debug("Stratum %s: %d samples (too few, using global)", stratum, len(group))
                continue

            imerg_s = apply_drizzle_threshold(group["rain_imerg"].values, self.drizzle_threshold)
            transfer = fit_qm_stratum(imerg_s, group["rain_station"].values, self.n_quantiles)
            self.transfers[stratum] = transfer
            log.info("Stratum %s: %d samples", stratum, transfer.n_samples)

        log.info("Fitted %d strata + global fallback", len(self.transfers))

    def predict(self, rain_imerg: np.ndarray, month: np.ndarray | int,
                elevation: float, lat: float, lon: float) -> np.ndarray:
        """Apply stratified QM correction.

        Args:
            rain_imerg: daily IMERG values (n_days,)
            month: month for each day (n_days,) or scalar
            elevation: station elevation (scalar)
            lat, lon: station location (scalar)

        Returns:
            corrected daily precipitation (n_days,)
        """
        rain_imerg = np.asarray(rain_imerg, dtype=np.float64)
        month = np.broadcast_to(month, rain_imerg.shape)

        # Apply drizzle threshold
        rain_clean = apply_drizzle_threshold(rain_imerg, self.drizzle_threshold)

        result = np.zeros_like(rain_clean)

        for i in range(len(rain_clean)):
            stratum = get_stratum(int(month[i]), elevation, lat, lon)
            transfer = self.transfers.get(stratum, self.global_transfer)
            result[i] = apply_qm(rain_clean[i:i+1], transfer)[0]

        return np.clip(result, 0, None)

    def save(self, path: Path) -> None:
        """Save fitted QM transfers to npz."""
        data = {}
        for key, tr in self.transfers.items():
            data[f"{key}_q"] = tr.quantiles
            data[f"{key}_imerg"] = tr.imerg_values
            data[f"{key}_station"] = tr.station_values
            data[f"{key}_n"] = np.array([tr.n_samples])

        if self.global_transfer:
            data["global_q"] = self.global_transfer.quantiles
            data["global_imerg"] = self.global_transfer.imerg_values
            data["global_station"] = self.global_transfer.station_values
            data["global_n"] = np.array([self.global_transfer.n_samples])

        data["drizzle_threshold"] = np.array([self.drizzle_threshold])
        data["n_quantiles"] = np.array([self.n_quantiles])
        data["strata_keys"] = np.array(list(self.transfers.keys()))

        np.savez(str(path), **data)
        log.info("Saved QM model to %s", path)

    @classmethod
    def load(cls, path: Path) -> "StratifiedQMCorrector":
        """Load fitted QM transfers from npz."""
        data = np.load(str(path), allow_pickle=True)

        obj = cls(
            drizzle_threshold=float(data["drizzle_threshold"][0]),
            n_quantiles=int(data["n_quantiles"][0]),
        )

        # Global
        if "global_q" in data:
            obj.global_transfer = QMTransfer(
                quantiles=data["global_q"],
                imerg_values=data["global_imerg"],
                station_values=data["global_station"],
                n_samples=int(data["global_n"][0]),
            )

        # Per-stratum
        for key in data.get("strata_keys", []):
            key = str(key)
            obj.transfers[key] = QMTransfer(
                quantiles=data[f"{key}_q"],
                imerg_values=data[f"{key}_imerg"],
                station_values=data[f"{key}_station"],
                n_samples=int(data[f"{key}_n"][0]),
            )

        return obj
