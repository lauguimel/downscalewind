"""
normalization.py — Normalisation z-score par variable et niveau de pression.

Les statistiques sont calculées sur le jeu d'entraînement et stockées en JSON.
La normalisation est appliquée avant d'entrer dans le modèle et inversée
lors de l'évaluation ou de l'inférence.

Note sur l'advection : le modèle utilise les composantes u,v pour calculer
le déplacement physique. Si les données sont normalisées, les valeurs u/v
sont en unités de σ_u et σ_v. Le modèle reçoit des std_u et std_v pour
corriger le calcul du déplacement. Voir AdvectionResidualInterpolator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import zarr


# Ordre des variables — doit correspondre à l'ordre dans les tenseurs du modèle
VARIABLE_ORDER = ["u", "v", "z", "t", "q"]


class NormStats:
    """
    Statistiques de normalisation z-score par variable et niveau.

    Attributes:
        mean: dict[str, np.ndarray] — moyenne par variable, shape (n_levels,)
        std:  dict[str, np.ndarray] — écart-type par variable, shape (n_levels,)
    """

    def __init__(
        self,
        mean: dict[str, np.ndarray],
        std: dict[str, np.ndarray],
    ) -> None:
        self.mean = mean  # {var: (n_levels,)}
        self.std  = std   # {var: (n_levels,)}

    # ── Sérialisation ─────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Sauvegarde les statistiques au format JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "mean": {k: v.tolist() for k, v in self.mean.items()},
            "std":  {k: v.tolist() for k, v in self.std.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "NormStats":
        """Charge les statistiques depuis un fichier JSON."""
        with open(path) as f:
            data = json.load(f)
        mean = {k: np.array(v, dtype=np.float32) for k, v in data["mean"].items()}
        std  = {k: np.array(v, dtype=np.float32) for k, v in data["std"].items()}
        return cls(mean=mean, std=std)

    # ── Normalisation / dénormalisation ──────────────────────────────────────

    def normalize(self, x: np.ndarray, var: str) -> np.ndarray:
        """
        Normalise un array de forme (..., n_levels, H, W).

        Args:
            x:   Array numpy, axes : (..., n_levels, H, W)
            var: Nom de la variable ('u', 'v', 'z', 't', 'q')
        """
        mu  = self.mean[var][:, None, None]  # (n_levels, 1, 1)
        sig = self.std[var][:, None, None]
        return (x - mu) / (sig + 1e-8)

    def denormalize(self, x: np.ndarray, var: str) -> np.ndarray:
        """Inverse de normalize."""
        mu  = self.mean[var][:, None, None]
        sig = self.std[var][:, None, None]
        return x * (sig + 1e-8) + mu

    def normalize_tensor(
        self,
        x: torch.Tensor,    # (..., n_levels, H, W)
        var: str,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Version PyTorch de normalize."""
        dev = device or x.device
        mu  = torch.from_numpy(self.mean[var]).to(dev, x.dtype)[:, None, None]
        sig = torch.from_numpy(self.std[var]).to(dev, x.dtype)[:, None, None]
        return (x - mu) / (sig + 1e-8)

    def denormalize_tensor(
        self,
        x: torch.Tensor,
        var: str,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        dev = device or x.device
        mu  = torch.from_numpy(self.mean[var]).to(dev, x.dtype)[:, None, None]
        sig = torch.from_numpy(self.std[var]).to(dev, x.dtype)[:, None, None]
        return x * (sig + 1e-8) + mu

    def normalize_snapshot(
        self,
        snap: np.ndarray,  # (n_vars, n_levels, H, W) dans l'ordre VARIABLE_ORDER
    ) -> np.ndarray:
        """Normalise un snapshot complet (toutes les variables empilées)."""
        result = np.empty_like(snap)
        for i, var in enumerate(VARIABLE_ORDER):
            result[i] = self.normalize(snap[i], var)
        return result

    def denormalize_snapshot(
        self,
        snap: np.ndarray,  # (n_vars, n_levels, H, W)
    ) -> np.ndarray:
        """Dénormalise un snapshot complet."""
        result = np.empty_like(snap)
        for i, var in enumerate(VARIABLE_ORDER):
            result[i] = self.denormalize(snap[i], var)
        return result

    def get_wind_std(self) -> tuple[float, float]:
        """
        Retourne les std moyens de u et v (moyennés sur les niveaux).

        Utilisé par le modèle pour convertir les vitesses normalisées
        en déplacements physiques pour l'advection.
        """
        std_u = float(np.mean(self.std["u"]))
        std_v = float(np.mean(self.std["v"]))
        return std_u, std_v

    # ── Calcul depuis un store Zarr ───────────────────────────────────────────

    @classmethod
    def compute_from_zarr(
        cls,
        zarr_path: str | Path,
        time_slice: slice | None = None,
        variables: list[str] | None = None,
        chunk_size: int = 500,
    ) -> "NormStats":
        """
        Calcule les statistiques z-score depuis un store Zarr ERA5.

        Utilise un algorithme en ligne (Welford) pour éviter de charger toutes
        les données en mémoire.

        Args:
            zarr_path:   Chemin vers le store Zarr DownscaleWind
            time_slice:  Slice temporel pour le calcul (défaut : tout)
            variables:   Variables à traiter (défaut : VARIABLE_ORDER)
            chunk_size:  Taille des chunks de lecture en temps

        Returns:
            NormStats avec mean et std calculés sur les données
        """
        store = zarr.open_group(str(zarr_path), mode="r")
        vars_to_process = variables or VARIABLE_ORDER
        n_times = store["pressure/u"].shape[0]
        t_slice = time_slice or slice(0, n_times)
        t_start, t_end = t_slice.start or 0, t_slice.stop or n_times

        mean_dict: dict[str, np.ndarray] = {}
        std_dict:  dict[str, np.ndarray] = {}

        for var in vars_to_process:
            arr = store[f"pressure/{var}"]
            n_levels = arr.shape[1]

            # Algorithme de Welford en ligne par niveau
            count = np.zeros(n_levels, dtype=np.float64)
            M = np.zeros(n_levels, dtype=np.float64)   # moyenne courante
            S = np.zeros(n_levels, dtype=np.float64)   # variance * count courante

            t = t_start
            while t < t_end:
                t_next = min(t + chunk_size, t_end)
                chunk = arr[t:t_next].astype(np.float64)  # (chunk, L, H, W)
                # Aplatir H×W pour chaque niveau
                n_chunk = chunk.shape[0]
                HW = chunk.shape[2] * chunk.shape[3]

                for lvl in range(n_levels):
                    vals = chunk[:, lvl, :, :].reshape(-1)  # (chunk*H*W,)
                    for x in vals:
                        count[lvl] += 1
                        delta = x - M[lvl]
                        M[lvl] += delta / count[lvl]
                        delta2 = x - M[lvl]
                        S[lvl] += delta * delta2
                t = t_next

            mean_dict[var] = M.astype(np.float32)
            std_dict[var]  = np.sqrt(S / np.maximum(count - 1, 1)).astype(np.float32)

        return cls(mean=mean_dict, std=std_dict)


# ── Poids de la loss par variable ─────────────────────────────────────────────
# Appliqués sur les données normalisées (std ≈ 1 après normalisation).
# Ces poids reflètent l'importance relative pour l'application.

LOSS_WEIGHTS: dict[str, float] = {
    "u": 2.0,   # vent zonal — cible principale
    "v": 2.0,   # vent méridional — cible principale
    "z": 1.0,   # géopotentiel — cohérence dynamique
    "t": 1.0,   # température — cycle diurne
    "q": 0.5,   # humidité — moins critique pour outdoor
}


def build_loss_weight_tensor(
    n_levels: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Construit le tenseur de poids de loss de forme (1, n_vars, n_levels, 1, 1).

    Utilisé pour le weighted MSE dans train.py.
    """
    weights = torch.tensor(
        [LOSS_WEIGHTS[v] for v in VARIABLE_ORDER],
        dtype=torch.float32,
        device=device,
    )  # (n_vars,)
    # Expand pour broadcaster sur (B, V, L, H, W) : shape (1, V, 1, 1, 1)
    return weights.view(1, -1, 1, 1, 1)
