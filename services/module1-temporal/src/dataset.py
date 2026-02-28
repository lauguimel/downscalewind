"""
dataset.py — ERA5TemporalDataset pour l'interpolation temporelle 6h → 1h.

Charge les paires d'entrée (S0 à t₀, S1 à t₀+6h) depuis le store Zarr 6h,
et les vérités terrain horaires (t₀+1h à t₀+5h) depuis le store Zarr 1h.

Splits temporels (sur la date de S0) :
  train : 2016-01-01 → 2016-10-31 inclus
  val   : 2016-11-01 → 2016-12-31 inclus
  test  : 2017-05-01 → 2017-06-15 inclus (IOP Perdigão)

Notes sur la normalisation :
  Les NormStats sont calculées sur le split train uniquement, puis appliquées
  à tous les splits (train/val/test) — ne jamais calculer sur val/test.

Notes sur le crop :
  grid_size doit être impair et ≤ min(n_lat, n_lon). Le crop est centré sur
  le domaine. Pour grid_size=7 sur un domaine 7×7, tout le domaine est utilisé.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from .normalization import NormStats, VARIABLE_ORDER


Split = Literal["train", "val", "test"]

# Splits temporels — bornes en datetime64[h]
# Borne haute exclue : t0 < t_hi
SPLIT_RANGES: dict[Split, tuple[str, str]] = {
    "train": ("2016-01-01T00", "2016-11-01T00"),
    "val":   ("2016-11-01T00", "2017-01-01T00"),
    "test":  ("2017-05-01T00", "2017-06-16T00"),
}


class ERA5TemporalDataset(Dataset):
    """
    Dataset pour l'interpolation temporelle ERA5 6h → 1h.

    Pour chaque fenêtre temporelle de 6h valide :
      - S0      : snapshot à t₀ (6h store)
      - S1      : snapshot à t₀+6h (6h store)
      - targets : snapshots à t₀+1h, ..., t₀+5h (1h store)

    Tous les tenseurs sont normalisés en z-score via norm_stats.

    Args:
        zarr_6h:       Chemin vers le store Zarr ERA5 6h (entrées)
        zarr_1h:       Chemin vers le store Zarr ERA5 1h (cibles)
        norm_stats:    Statistiques de normalisation (calculées sur train)
        split:         'train', 'val', ou 'test'
        grid_size:     Côté de la fenêtre spatiale, impair ≤ min(n_lat, n_lon).
                       None = tout le domaine.
        n_intermediate: Nombre de pas intermédiaires (défaut: 5 pour 6h→1h)
    """

    def __init__(
        self,
        zarr_6h: str | Path,
        zarr_1h: str | Path,
        norm_stats: NormStats,
        split: Split = "train",
        grid_size: int | None = None,
        n_intermediate: int = 5,
    ) -> None:
        self.norm_stats = norm_stats
        self.split = split
        self.n_intermediate = n_intermediate

        # Ouvrir les stores en lecture
        self._store_6h = zarr.open_group(str(zarr_6h), mode="r")
        self._store_1h = zarr.open_group(str(zarr_1h), mode="r")

        # Les timestamps peuvent être stockés comme int64 (datetime64[ns] as int64)
        # ou comme datetime64[ns] selon la version zarr — on normalise dans les deux cas.
        def _to_datetime64h(arr: np.ndarray) -> np.ndarray:
            if arr.dtype.kind in ("i", "u"):
                # int64 : nanoseconds since epoch → datetime64[ns]
                return arr.astype("datetime64[ns]").astype("datetime64[h]")
            return arr.astype("datetime64[h]")

        times_6h = _to_datetime64h(self._store_6h["coords/time"][:])
        times_1h = _to_datetime64h(self._store_1h["coords/time"][:])

        # Index horaire : datetime64[h] → indice dans le store 1h
        self._hourly_idx: dict[np.datetime64, int] = {
            t: i for i, t in enumerate(times_1h)
        }

        # Bornes du split
        t_lo = np.datetime64(SPLIT_RANGES[split][0], "h")
        t_hi = np.datetime64(SPLIT_RANGES[split][1], "h")

        dt_6h = np.timedelta64(6, "h")
        dt_1h = np.timedelta64(1, "h")

        # Construction de l'index des fenêtres valides
        # Chaque entrée = (idx_s0, idx_s1, [idx_h1, ..., idx_h5])
        self._windows: list[tuple[int, int, list[int]]] = []

        for i in range(len(times_6h) - 1):
            t0 = times_6h[i]
            t1 = times_6h[i + 1]

            # t0 doit être dans le split
            if t0 < t_lo or t0 >= t_hi:
                continue

            # Vérifier que c'est une fenêtre de 6h exactement (pas de trou)
            if t1 != t0 + dt_6h:
                continue

            # Vérifier que tous les pas horaires intermédiaires sont disponibles
            hourly_idxs: list[int] = []
            valid = True
            for k in range(1, n_intermediate + 1):
                t_k = t0 + k * dt_1h
                if t_k not in self._hourly_idx:
                    valid = False
                    break
                hourly_idxs.append(self._hourly_idx[t_k])

            if valid:
                self._windows.append((i, i + 1, hourly_idxs))

        if len(self._windows) == 0:
            raise RuntimeError(
                f"Aucune fenêtre valide pour le split '{split}'. "
                f"Vérifiez que les stores couvrent la période {SPLIT_RANGES[split]}."
            )

        # Crop spatial centré
        n_lat = self._store_6h["pressure/u"].shape[2]
        n_lon = self._store_6h["pressure/u"].shape[3]

        if grid_size is None:
            self._lat_slice = slice(None)
            self._lon_slice = slice(None)
        else:
            if grid_size > n_lat or grid_size > n_lon:
                raise ValueError(
                    f"grid_size={grid_size} dépasse le domaine ({n_lat}×{n_lon})"
                )
            clat = n_lat // 2
            clon = n_lon // 2
            half = grid_size // 2
            self._lat_slice = slice(clat - half, clat - half + grid_size)
            self._lon_slice = slice(clon - half, clon - half + grid_size)

    def __len__(self) -> int:
        return len(self._windows)

    def _load_snapshot(self, store: zarr.Group, time_idx: int) -> np.ndarray:
        """
        Charge un snapshot complet depuis un store Zarr.

        Returns:
            Array float32 de shape (n_vars, n_levels, H, W) dans l'ordre VARIABLE_ORDER.
        """
        slices = (time_idx, slice(None), self._lat_slice, self._lon_slice)
        return np.stack(
            [store[f"pressure/{var}"][slices].astype(np.float32) for var in VARIABLE_ORDER],
            axis=0,
        )  # (V, L, H, W)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retourne (S0, S1, targets).

        Returns:
            S0      : Tensor (V, L, H, W) normalisé — snapshot entrée à t₀
            S1      : Tensor (V, L, H, W) normalisé — snapshot entrée à t₀+6h
            targets : Tensor (n_intermediate, V, L, H, W) normalisé — vérités terrain
        """
        idx_s0, idx_s1, hourly_idxs = self._windows[idx]

        snap0 = self._load_snapshot(self._store_6h, idx_s0)
        snap1 = self._load_snapshot(self._store_6h, idx_s1)

        snap0_norm = self.norm_stats.normalize_snapshot(snap0)
        snap1_norm = self.norm_stats.normalize_snapshot(snap1)

        target_snaps = [
            self.norm_stats.normalize_snapshot(
                self._load_snapshot(self._store_1h, h_idx)
            )
            for h_idx in hourly_idxs
        ]

        S0      = torch.from_numpy(snap0_norm)
        S1      = torch.from_numpy(snap1_norm)
        targets = torch.from_numpy(np.stack(target_snaps, axis=0))  # (T, V, L, H, W)

        return S0, S1, targets

    # ── Utilitaires ───────────────────────────────────────────────────────────

    @property
    def n_windows(self) -> int:
        return len(self._windows)

    def info(self) -> str:
        """Description courte du dataset pour les logs."""
        idx_s0, _, _ = self._windows[0]
        idx_sN, _, _ = self._windows[-1]
        t0_raw = self._store_6h["coords/time"][idx_s0]
        tN_raw = self._store_6h["coords/time"][idx_sN]
        def _to_h(v):
            a = np.array(v)
            return a.astype("datetime64[ns]").astype("datetime64[h]") if a.dtype.kind in ("i","u") else a.astype("datetime64[h]")
        t0_dt = _to_h(t0_raw)
        tN_dt = _to_h(tN_raw)
        return (
            f"ERA5TemporalDataset split={self.split!r} "
            f"windows={len(self._windows)} "
            f"period={t0_dt}→{tN_dt}"
        )
