"""
dataset_v2.py — PyTorch Dataset for the campaign-v2 native-grid surrogate.

Reads grid.zarr stores produced by services/module2a-cfd/export_to_grid_zarr_v2.py
on a 180×180×40 mesh-native grid.

Convention (matches export_to_grid_zarr_v2.py):
    coords/x, y                       (180,)            voxel column centres [m]
    coords/z                          (180, 180, 40)    real cell-centre altitude
    input/terrain                     (180, 180)        ground altitude
    input/                                              attrs: lat, lon, z0_eff
    input/era5_pressure_levels        (N_p,)            hPa
    input/era5_3d/{u,v,T,q}           (3, 3, N_p)
    input/era5_surface/{t2m,d2m,u10,v10}  (3, 3)
    target/U                          (180, 180, 40, 3) m/s
    target/T                          (180, 180, 40)    K
    target/q                          (180, 180, 40)    kg/kg

Outputs:
    inputs  (C_in, NI, NJ, NK)  — terrain/z0/lat/AGL/era5/surface broadcast onto the 3D grid
    target  (5,    NI, NJ, NK)  — (u/scale, v/scale, w/scale, (T-T0)/scale, q/scale)

Splits:
    Reads `dataset_v2_splits.yaml` (site → split) — watertight by site (seed=42).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
import zarr
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Native grid (must match export_to_grid_zarr_v2.NI/NJ/NK)
NI, NJ, NK = 180, 180, 40

# Default normalisation. Replace with data-driven stats once
# compute_norm_stats_v2.py has run on the train split.
DEFAULT_NORM = {
    "U_uv_scale": 15.0,    # m/s
    "U_w_scale":  3.0,     # m/s
    "T_offset":   290.0,   # K (centring)
    "T_scale":    10.0,    # K
    "q_scale":    0.01,    # kg/kg
    "terrain_scale": 1500.0,  # m
    "z_scale":      2500.0,   # m (top of inner block)
    "z0_scale":     0.5,      # m (typical effective z0)
    "lat_scale":    90.0,     # deg (range [-90,90])
    "era5_u_scale": 15.0,
    "era5_v_scale": 15.0,
    "era5_T_offset": 270.0,
    "era5_T_scale":  20.0,
    "era5_q_scale":  0.01,
    "t2m_offset":    290.0,
    "t2m_scale":     10.0,
    "d2m_offset":    285.0,
    "d2m_scale":     10.0,
    "u10_scale":     10.0,
    "v10_scale":     10.0,
    "pressure_offset": 700.0,  # hPa centring
    "pressure_scale":  300.0,
}


def _broadcast(value: np.ndarray | float, shape: tuple[int, ...]) -> np.ndarray:
    """Broadcast a scalar / 1D profile / 2D field to `shape` (NI, NJ, NK)."""
    if np.ndim(value) == 0:
        return np.full(shape, float(value), dtype=np.float32)
    if value.ndim == 1:
        # Pressure-level profile (N_p,) → tile in (i, j) and resample to NK along k
        # We just broadcast to (NI, NJ, N_p) then take a fixed mapping below.
        return np.broadcast_to(value.astype(np.float32), shape).copy()
    if value.ndim == 2:
        return np.broadcast_to(value[:, :, None].astype(np.float32), shape).copy()
    return np.asarray(value, dtype=np.float32)


class WindV2Dataset(Dataset):
    """Native-grid dataset for the v2 surrogate.

    Parameters
    ----------
    data_dir : directory containing `<site_id>_<case_name>/grid.zarr` cases
    splits_yaml : path to `dataset_v2_splits.yaml` ({train: [sites...], val:..., test:...})
    split : 'train' | 'val' | 'test'
    norm : dict overriding `DEFAULT_NORM`
    include_z : whether to add the real-altitude channel `coords/z` to inputs
    """

    def __init__(
        self,
        data_dir: str | Path,
        splits_yaml: str | Path,
        split: str = "train",
        *,
        norm: dict | None = None,
        include_z: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.norm = {**DEFAULT_NORM, **(norm or {})}
        self.include_z = include_z

        with open(splits_yaml) as f:
            splits = yaml.safe_load(f)
        if split not in splits:
            raise KeyError(f"split '{split}' not in {list(splits)}")
        site_ids: list[str] = splits[split]

        cases: list[Path] = []
        missing_sites = 0
        for sid in site_ids:
            site_cases = sorted(self.data_dir.glob(f"{sid}_case_ts*"))
            site_cases = [c for c in site_cases if (c / "grid.zarr").exists()]
            if not site_cases:
                missing_sites += 1
                continue
            cases.extend(site_cases)
        self.cases = cases

        logger.info(
            "WindV2Dataset[%s]: %d cases from %d sites (missing %d sites)",
            split, len(self.cases), len(site_ids) - missing_sites, missing_sites,
        )

    def __len__(self) -> int:
        return len(self.cases)

    # ── Helpers ────────────────────────────────────────────────────────────
    def _build_input_tensor(self, store: Any) -> np.ndarray:
        """Concat all inputs into (C_in, NI, NJ, NK) float32 tensor."""
        n = self.norm
        chans: list[np.ndarray] = []

        # 1. terrain (NI, NJ) → broadcast over k
        terrain = np.asarray(store["input/terrain"][:], dtype=np.float32)
        chans.append(np.broadcast_to(terrain[:, :, None], (NI, NJ, NK)).copy()
                     / n["terrain_scale"])

        # 2. z (real altitude) (NI, NJ, NK) — explicit grid deformation
        if self.include_z:
            z = np.asarray(store["coords/z"][:], dtype=np.float32)
            chans.append(z / n["z_scale"])

        # 3. AGL (z - terrain), more useful than absolute z for log-law
        z = np.asarray(store["coords/z"][:], dtype=np.float32)
        agl = z - terrain[:, :, None]
        chans.append(agl / n["z_scale"])

        # 4. z0 (scalar) broadcast
        z0 = float(store["input"].attrs.get("z0_eff", 0.0))
        chans.append(np.full((NI, NJ, NK), z0 / n["z0_scale"], dtype=np.float32))

        # 5. lat (Coriolis) scalar broadcast
        lat = float(store["input"].attrs.get("lat", 0.0))
        chans.append(np.full((NI, NJ, NK), lat / n["lat_scale"], dtype=np.float32))

        # 6. ERA5 1D profile from centre of 3×3 (interpolate from N_p hPa to NK z-bins)
        plev = np.asarray(store["input/era5_pressure_levels"][:], dtype=np.float32)
        # Map ERA5 pressure profile to the NK levels using the column mean z.
        # We use the column-mean AGL as a proxy for k → height mapping.
        for var, scale, offset in [
            ("u", n["era5_u_scale"], 0.0),
            ("v", n["era5_v_scale"], 0.0),
            ("T", n["era5_T_scale"], n["era5_T_offset"]),
            ("q", n["era5_q_scale"], 0.0),
        ]:
            era5_var = np.asarray(store[f"input/era5_3d/{var}"][:], dtype=np.float32)
            # Take centre of 3×3 grid: shape (N_p,)
            prof_1d = era5_var[1, 1, :]
            # Broadcast over (NI, NJ) and interpolate from N_p to NK:
            # We linearly interpolate along k by mapping pressure to fractional index.
            # For simplicity, project on NK via uniform indexing — model can learn the mapping.
            k_idx = np.linspace(0, len(prof_1d) - 1, NK)
            prof_nk = np.interp(k_idx, np.arange(len(prof_1d)), prof_1d)  # (NK,)
            field = np.broadcast_to(prof_nk[None, None, :], (NI, NJ, NK)).copy()
            chans.append((field - offset) / scale)

        # 7. ERA5 pressure level profile itself (NK,) broadcast
        plev_k = np.interp(np.linspace(0, len(plev) - 1, NK),
                           np.arange(len(plev)), plev)
        chans.append(np.broadcast_to(
            ((plev_k - n["pressure_offset"]) / n["pressure_scale"])[None, None, :],
            (NI, NJ, NK)).copy().astype(np.float32))

        # 8. Surface ERA5 (3×3 → centre scalar) broadcast
        for var, scale, offset in [
            ("t2m", n["t2m_scale"], n["t2m_offset"]),
            ("d2m", n["d2m_scale"], n["d2m_offset"]),
            ("u10", n["u10_scale"], 0.0),
            ("v10", n["v10_scale"], 0.0),
        ]:
            arr = np.asarray(store[f"input/era5_surface/{var}"][:], dtype=np.float32)
            val = float(arr[1, 1])
            chans.append(np.full((NI, NJ, NK), (val - offset) / scale, dtype=np.float32))

        return np.stack(chans, axis=0)  # (C_in, NI, NJ, NK)

    def _build_target_tensor(self, store: Any) -> np.ndarray:
        """Build (5, NI, NJ, NK) target = (u, v, w, T, q) normalised."""
        n = self.norm
        U = np.asarray(store["target/U"][:], dtype=np.float32)  # (NI, NJ, NK, 3)
        T = np.asarray(store["target/T"][:], dtype=np.float32)
        q = np.asarray(store["target/q"][:], dtype=np.float32)
        return np.stack([
            U[..., 0] / n["U_uv_scale"],
            U[..., 1] / n["U_uv_scale"],
            U[..., 2] / n["U_w_scale"],
            (T - n["T_offset"]) / n["T_scale"],
            q / n["q_scale"],
        ], axis=0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        case_dir = self.cases[idx]
        store = zarr.open_group(str(case_dir / "grid.zarr"), mode="r")
        inp = self._build_input_tensor(store)
        tgt = self._build_target_tensor(store)
        return torch.from_numpy(inp), torch.from_numpy(tgt), case_dir.name


if __name__ == "__main__":
    # Quick sanity check
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/scratch/maitreje/dsw/training_v2")
    ap.add_argument("--splits-yaml",
                    default="/scratch/maitreje/dsw/complex_terrain_v1/manifests/dataset_v2_splits.yaml")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    ds = WindV2Dataset(args.data_dir, args.splits_yaml, args.split)
    print(f"Dataset: {len(ds)} cases")
    inp, tgt, case_id = ds[0]
    print(f"  input shape: {inp.shape}, range [{inp.min():.3f}, {inp.max():.3f}]")
    print(f"  target shape: {tgt.shape}, range [{tgt.min():.3f}, {tgt.max():.3f}]")
    print(f"  case_id: {case_id}")
