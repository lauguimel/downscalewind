"""
dataset_wind9k.py — PyTorch dataset for 9k wind downscaling campaign.

Loads grid.zarr cases with strict geographic split:
  - Each site is exclusively in train OR val OR test
  - Split is read from dataset.yaml (pre-assigned per case)

Grid schema (128x128x32):
    input/terrain    (128, 128)    — elevation [m]
    input/z0         (128, 128)    — roughness [m]
    input/era5/{u,v,T,q,k} (32,)  — ERA5 inflow profiles
    target/U         (128, 128, 32, 3) — CFD velocity
    target/T         (128, 128, 32)    — CFD temperature
    target/q         (128, 128, 32)    — CFD humidity
    residual/U       (128, 128, 32, 3) — velocity residual (CFD - ERA5)
    residual/T       (128, 128, 32)    — T residual
    residual/q       (128, 128, 32)    — q residual
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset


# Normalization constants
TERRAIN_SCALE = 500.0   # elevation [m]
Z0_SCALE = 1.0          # roughness [m]
WIND_UV_SCALE = 20.0    # horizontal velocity [m/s]
WIND_W_SCALE = 5.0      # vertical velocity [m/s]
T_SCALE = 30.0          # temperature deviation [K]
Q_SCALE = 0.01          # humidity [kg/kg]
K_SCALE = 5.0           # TKE [m²/s²]

# ERA5 profile normalization (per-variable)
ERA5_SCALES = {
    "u": WIND_UV_SCALE,
    "v": WIND_UV_SCALE,
    "T": 300.0,   # absolute T [K]
    "q": Q_SCALE,
    "k": K_SCALE,
}


class Wind9kDataset(Dataset):
    """Dataset for 9k wind downscaling with geographic site split.

    Returns separate terrain and ERA5 tensors for the ViT architecture.
    """

    def __init__(
        self,
        data_dir: str | Path,
        dataset_yaml: str | Path,
        split: str = "train",
        use_residual: bool = True,
        augment: bool = False,
        era5_mode: str = "1d",  # "1d", "3d" (full 3×3), "grad" (center + gradients)
    ):
        self.era5_mode = era5_mode
        self.data_dir = Path(data_dir)
        with open(dataset_yaml) as f:
            self.meta = yaml.safe_load(f)

        self.cases = [c for c in self.meta["cases"] if c["split"] == split]
        self.use_residual = use_residual
        self.augment = augment

        # Verify geographic exclusivity
        sites_in_split = set()
        all_splits_by_site: dict[str, set] = {}
        for c in self.meta["cases"]:
            site = c["case_id"].rsplit("_case_", 1)[0]
            all_splits_by_site.setdefault(site, set()).add(c["split"])
        leaked = [s for s, sp in all_splits_by_site.items() if len(sp) > 1]
        if leaked:
            raise ValueError(
                f"Geographic leakage detected! {len(leaked)} sites appear in "
                f"multiple splits: {leaked[:5]}..."
            )

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        import zarr

        case = self.cases[idx]
        case_id = case["case_id"]
        store = zarr.open_group(
            str(self.data_dir / case_id / "grid.zarr"), mode="r")

        # ── Terrain input (2, 128, 128) ──
        terrain = np.array(store["input/terrain"][:], dtype=np.float32)
        z0 = np.array(store["input/z0"][:], dtype=np.float32)
        terrain_input = np.stack([
            terrain / TERRAIN_SCALE,
            z0 / Z0_SCALE,
        ], axis=0)  # (2, ny, nx)

        # ── ERA5 input ──
        # Prefer 3D grid (3, 3, 32) if available, else fallback to 1D (5, 32)
        # Load ERA5 3D (5, 3, 3, 32) on log-law reconstructed AGL levels
        if self.era5_mode in ("3d", "grad") and "input/era5_3d" in store:
            era5_3d_grp = store["input/era5_3d"]
            era5_channels = []
            for var in ["u", "v", "T", "q", "k"]:
                if var in era5_3d_grp:
                    arr = np.array(era5_3d_grp[var][:], dtype=np.float32)
                    era5_channels.append(arr / ERA5_SCALES[var])  # (3, 3, 32)
                else:
                    era5_channels.append(np.zeros((3, 3, 32), dtype=np.float32))
            era5_3d = np.stack(era5_channels, axis=0)  # (5, 3, 3, 32)

            if self.era5_mode == "3d":
                era5_input = era5_3d  # (5, 3, 3, 32)
            else:  # grad
                center = era5_3d[:, 1, 1, :]  # (5, 32)
                grad_x = (era5_3d[:, 1, 2, :] - era5_3d[:, 1, 0, :]) / 2  # (5, 32)
                grad_y = (era5_3d[:, 2, 1, :] - era5_3d[:, 0, 1, :]) / 2  # (5, 32)
                era5_input = np.stack([center, grad_x, grad_y], axis=0)  # (3, 5, 32)
        else:
            # 1D mode (baseline) or fallback
            era5_grp = store["input/era5"]
            profiles = []
            for var in ["u", "v", "T", "q", "k"]:
                if var in era5_grp:
                    prof = np.array(era5_grp[var][:], dtype=np.float32)
                    profiles.append(prof / ERA5_SCALES[var])
                else:
                    profiles.append(np.zeros(32, dtype=np.float32))
            era5_input = np.stack(profiles, axis=0)  # (5, 32)

        # ── Target (5, ny, nx, nz) ──
        if self.use_residual and "residual" in store:
            U = np.array(store["residual/U"][:], dtype=np.float32)  # (ny,nx,nz,3)
            T = np.array(store["residual/T"][:], dtype=np.float32)  # (ny,nx,nz)
            q = np.array(store["residual/q"][:], dtype=np.float32)  # (ny,nx,nz)
        else:
            U = np.array(store["target/U"][:], dtype=np.float32)
            T = np.array(store["target/T"][:], dtype=np.float32)
            q = np.array(store["target/q"][:], dtype=np.float32)

        target = np.stack([
            U[:, :, :, 0] / WIND_UV_SCALE,  # u
            U[:, :, :, 1] / WIND_UV_SCALE,  # v
            U[:, :, :, 2] / WIND_W_SCALE,   # w
            T / T_SCALE,
            q / Q_SCALE,
        ], axis=0).copy()  # (5, ny, nx, nz)

        # Augmentation disabled for 3D/grad modes (spatial rotation complex)
        # Kept only as no-op to preserve signature

        return (
            torch.from_numpy(terrain_input),  # (2, 128, 128)
            torch.from_numpy(era5_input),      # (5, 32) 1D, (5,3,3,32) 3D, or (3,5,32) grad
            torch.from_numpy(target),          # (5, 128, 128, 32)
        )
