"""TorchScript inference wrapper for the FNO3D surrogate on CPU (ARM target).

Loads the traced model once at app startup. Thread-safe via a lock because
TorchScript models are not safe to call from multiple threads simultaneously.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import torch


class FNOEngine:
    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self._model = torch.jit.load(str(model_path), map_location="cpu")
        self._model.eval()
        self._lock = threading.Lock()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Run a single forward pass.

        Parameters
        ----------
        x : (7, ny, nx, nz) float32, already normalized

        Returns
        -------
        (5, ny, nx, nz) float32, the 5-channel residual (u,v,w,T,q) normalized
        """
        if x.ndim != 4 or x.shape[0] != 7:
            raise ValueError(f"expected (7, ny, nx, nz), got {x.shape}")
        tensor = torch.from_numpy(x).unsqueeze(0).float()  # (1,7,ny,nx,nz)
        with self._lock, torch.no_grad():
            out = self._model(tensor)
        return out.squeeze(0).numpy()
