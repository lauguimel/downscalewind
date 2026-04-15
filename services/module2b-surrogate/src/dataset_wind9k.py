"""
dataset_wind9k.py — PyTorch dataset for wind+T+q downscaling surrogate.

Supports two grid schemas:

  (a) 9k/legacy  (used until complex_terrain_v1):
      input/terrain, input/z0, input/era5/{u,v,T,q,k}
      target/U, target/T, target/q

  (b) complex_terrain_v1 (2026-04-14):
      input/terrain, input/z0,
      input/era5/{u,v,T,q,k},
      input/era5_3d/{u,v,T,q,k}                (optional 3×3),
      input/era5_surface/{t2m,d2m,u10,v10}    (optional surface vars),
      target/U, target/T, target/q,
      target/{k,epsilon,nut}                   (optional extra targets)

Geographic split:
  - Strict watertight per site (site never in two splits).
  - Read from `dataset.yaml` OR `manifests/splits.yaml`.

Grid schema (default 128×128×32):
    input/terrain    (128, 128)    — elevation [m]
    input/z0         (128, 128)    — roughness [m]
    input/era5/{u,v,T,q,k} (32,)  — ERA5 inflow profiles
    input/era5_3d/{u,v,T,q,k}     (3, 3, 32)  — 3×3 ERA5 grid
    input/era5_surface/{t2m,d2m,u10,v10} (3, 3) or () — surface ERA5
    target/U         (128, 128, 32, 3) — CFD velocity
    target/T         (128, 128, 32)    — CFD temperature
    target/q         (128, 128, 32)    — CFD humidity
    target/k, target/epsilon, target/nut (optional)
    residual/U, residual/T, residual/q, residual/k (CFD - ERA5)
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
EPS_SCALE = 0.1         # dissipation [m²/s³]
NUT_SCALE = 1.0         # turb viscosity [m²/s]

# ERA5 profile normalization (per-variable, for input channels)
ERA5_SCALES = {
    "u": WIND_UV_SCALE,
    "v": WIND_UV_SCALE,
    "T": 300.0,   # absolute T [K]
    "q": Q_SCALE,
    "k": K_SCALE,
}

# Surface ERA5 normalization
ERA5_SURFACE_SCALES = {
    "t2m": 300.0,   # K
    "d2m": 300.0,   # K
    "u10": WIND_UV_SCALE,
    "v10": WIND_UV_SCALE,
}

EXTRA_TARGET_SCALES = {
    "k": K_SCALE,
    "epsilon": EPS_SCALE,
    "nut": NUT_SCALE,
}


class Wind9kDataset(Dataset):
    """Unified dataset for 9k + complex_terrain_v1 wind downscaling.

    Parameters
    ----------
    data_dir : directory containing case dirs `{site_id}_case_tsNNN/grid.zarr`
    split : 'train' / 'val' / 'test'
    dataset_yaml : either `dataset.yaml` (9k schema, cases+split) or
                   `manifests/splits.yaml` (complex_terrain_v1 schema)
    site_manifest : optional `manifests/sites.yaml` for per-site metadata lookup
                    (used to cross-check splits if dataset_yaml holds only splits)
    use_residual : return CFD-ERA5 residual as target (recommended for training stability)
    era5_mode : '1d' (5, 32), '3d' (5, 3, 3, 32), 'grad' (3, 5, 32), '1d+surface' (adds 4 surface channels)
    extra_targets : tuple of additional fields to include in target tensor
                    (subset of {'k', 'epsilon', 'nut'})
    """

    def __init__(
        self,
        data_dir: str | Path,
        dataset_yaml: str | Path,
        split: str = "train",
        *,
        site_manifest: str | Path | None = None,
        use_residual: bool = True,
        augment: bool = False,
        era5_mode: str = "1d",
        extra_targets: tuple[str, ...] = (),
    ) -> None:
        self.era5_mode = era5_mode
        self.data_dir = Path(data_dir)
        self.extra_targets = tuple(extra_targets)
        for t in self.extra_targets:
            if t not in EXTRA_TARGET_SCALES:
                raise ValueError(
                    f"extra_targets contains unknown field '{t}'. Valid: "
                    f"{list(EXTRA_TARGET_SCALES)}")

        with open(dataset_yaml) as f:
            meta = yaml.safe_load(f)

        # Two supported schemas:
        # (a) 9k: meta has "cases": [{case_id, split}, ...]
        # (b) complex_terrain_v1: meta has {"train": [...], "val": [...], "test": [...]}
        self.meta = meta
        if "cases" in meta:
            self.cases = [c for c in meta["cases"] if c["split"] == split]
            self._schema = "9k"
        elif split in meta:
            # site-level split → enumerate case dirs per site
            site_ids = meta[split]
            self.cases = []
            for site_id in site_ids:
                for case_dir in sorted(self.data_dir.glob(f"{site_id}_case_ts*")):
                    if (case_dir / "grid.zarr").exists():
                        self.cases.append({"case_id": case_dir.name, "split": split})
            self._schema = "complex_terrain_v1"
        else:
            raise KeyError(
                f"dataset_yaml must contain either 'cases' (9k schema) or "
                f"'{split}' (complex_terrain_v1 schema); got keys {list(meta)}")

        self.use_residual = use_residual
        self.augment = augment

        # Verify geographic exclusivity (watertight site split)
        all_splits_by_site: dict[str, set] = {}
        if self._schema == "9k":
            for c in meta["cases"]:
                site = c["case_id"].rsplit("_case_", 1)[0]
                all_splits_by_site.setdefault(site, set()).add(c["split"])
        else:
            for s in ["train", "val", "test"]:
                for site_id in meta.get(s, []):
                    all_splits_by_site.setdefault(site_id, set()).add(s)
        leaked = [s for s, sp in all_splits_by_site.items() if len(sp) > 1]
        if leaked:
            raise ValueError(
                f"Geographic leakage detected! {len(leaked)} sites appear in "
                f"multiple splits: {leaked[:5]}...")

        # Optional site metadata
        self._site_meta: dict[str, dict] = {}
        if site_manifest is not None:
            with open(site_manifest) as f:
                sm = yaml.safe_load(f) or {}
            for s in sm.get("sites", []):
                self._site_meta[s["site_id"]] = s

    def __len__(self) -> int:
        return len(self.cases)

    def _load_case(self, idx: int):
        import zarr
        case = self.cases[idx]
        case_id = case["case_id"]
        store = zarr.open_group(str(self.data_dir / case_id / "grid.zarr"), mode="r")
        return case_id, store

    # ── Input assembly ───────────────────────────────────────────────
    def _build_era5_input(self, store) -> np.ndarray:
        """Assemble ERA5 input tensor according to era5_mode."""
        if self.era5_mode in ("3d", "grad") and "input/era5_3d" in store:
            era5_3d_grp = store["input/era5_3d"]
            channels = []
            for var in ["u", "v", "T", "q", "k"]:
                if var in era5_3d_grp:
                    arr = np.array(era5_3d_grp[var][:], dtype=np.float32)
                    channels.append(arr / ERA5_SCALES[var])
                else:
                    channels.append(np.zeros((3, 3, 32), dtype=np.float32))
            era5_3d = np.stack(channels, axis=0)  # (5, 3, 3, 32)
            if self.era5_mode == "3d":
                return era5_3d
            # grad: center + gradients
            center = era5_3d[:, 1, 1, :]
            gx = (era5_3d[:, 1, 2, :] - era5_3d[:, 1, 0, :]) / 2
            gy = (era5_3d[:, 2, 1, :] - era5_3d[:, 0, 1, :]) / 2
            return np.stack([center, gx, gy], axis=0)  # (3, 5, 32)

        # ── 1D mode ──
        era5_grp = store["input/era5"]
        profiles = []
        for var in ["u", "v", "T", "q", "k"]:
            if var in era5_grp:
                prof = np.array(era5_grp[var][:], dtype=np.float32)
                profiles.append(prof / ERA5_SCALES[var])
            else:
                profiles.append(np.zeros(32, dtype=np.float32))
        era5_1d = np.stack(profiles, axis=0)  # (5, 32)

        if self.era5_mode != "1d+surface":
            return era5_1d

        # "1d+surface": append 4 surface channels broadcast over 32 levels
        # Each channel is a scalar (or 3×3 → take center) broadcast to 32
        surf_values = []
        has_surface = "input/era5_surface" in store
        surf_grp = store["input/era5_surface"] if has_surface else None
        for var in ["t2m", "d2m", "u10", "v10"]:
            if surf_grp is not None and var in surf_grp:
                arr = np.array(surf_grp[var][:], dtype=np.float32)
                val = float(arr[1, 1] if arr.ndim == 2 else arr.item())
            else:
                val = np.nan
            surf_values.append(val / ERA5_SURFACE_SCALES[var] if np.isfinite(val) else 0.0)
        surf_broadcast = np.array(surf_values, dtype=np.float32)[:, None] * np.ones(
            32, dtype=np.float32)[None, :]
        return np.concatenate([era5_1d, surf_broadcast], axis=0)  # (9, 32)

    def _build_target(self, store) -> np.ndarray:
        """Assemble target tensor: (C, ny, nx, nz) with C = 5 + len(extra_targets)."""
        if self.use_residual and "residual" in store:
            U = np.array(store["residual/U"][:], dtype=np.float32)
            T = np.array(store["residual/T"][:], dtype=np.float32) if "residual/T" in store else None
            q = np.array(store["residual/q"][:], dtype=np.float32) if "residual/q" in store else None
            extras = {
                t: (np.array(store[f"residual/{t}"][:], dtype=np.float32)
                    if f"residual/{t}" in store
                    else np.array(store[f"target/{t}"][:], dtype=np.float32))
                for t in self.extra_targets
                if (f"residual/{t}" in store or f"target/{t}" in store)
            }
        else:
            U = np.array(store["target/U"][:], dtype=np.float32)
            T = np.array(store["target/T"][:], dtype=np.float32) if "target/T" in store else None
            q = np.array(store["target/q"][:], dtype=np.float32) if "target/q" in store else None
            extras = {
                t: np.array(store[f"target/{t}"][:], dtype=np.float32)
                for t in self.extra_targets
                if f"target/{t}" in store
            }

        chans = [
            U[:, :, :, 0] / WIND_UV_SCALE,
            U[:, :, :, 1] / WIND_UV_SCALE,
            U[:, :, :, 2] / WIND_W_SCALE,
        ]
        # Fill missing T/q with zeros of correct shape (edge cases)
        shape3 = U.shape[:3]
        chans.append((T / T_SCALE) if T is not None else np.zeros(shape3, dtype=np.float32))
        chans.append((q / Q_SCALE) if q is not None else np.zeros(shape3, dtype=np.float32))
        for t in self.extra_targets:
            if t in extras:
                chans.append(extras[t] / EXTRA_TARGET_SCALES[t])
            else:
                chans.append(np.zeros(shape3, dtype=np.float32))
        return np.stack(chans, axis=0).copy()

    def __getitem__(self, idx: int):
        case_id, store = self._load_case(idx)

        # ── Terrain input (2, 128, 128) ──
        terrain = np.array(store["input/terrain"][:], dtype=np.float32)
        z0 = np.array(store["input/z0"][:], dtype=np.float32)
        terrain_input = np.stack([
            terrain / TERRAIN_SCALE,
            z0 / Z0_SCALE,
        ], axis=0)

        # ── ERA5 input (variable shape depending on era5_mode) ──
        era5_input = self._build_era5_input(store)

        # ── Target ──
        target = self._build_target(store)

        return (
            torch.from_numpy(terrain_input),
            torch.from_numpy(era5_input),
            torch.from_numpy(target),
        )
