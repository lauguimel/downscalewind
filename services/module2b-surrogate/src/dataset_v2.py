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

# FWI-oriented AGL target grid: denser below 5 m, then 5 m spacing to 100 m.
DEFAULT_AGL_0_100_24 = (
    0.0, 2.0, 3.0, 4.0, 5.0,
    10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0,
    55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0,
)

# Default normalisation. Replace with data-driven stats once
# compute_norm_stats_v2.py has run on the train split.
DEFAULT_NORM = {
    "U_x_offset": 0.0,
    "U_y_offset": 0.0,
    "U_z_offset": 0.0,
    "U_uv_scale": 15.0,    # m/s
    "U_w_scale":  3.0,     # m/s
    "T_offset":   290.0,   # K (centring)
    "T_scale":    10.0,    # K
    "q_offset":   0.0,
    "q_scale":    0.01,    # kg/kg
    "terrain_scale": 1500.0,  # m
    "z_scale":      2500.0,   # m (top of inner block)
    "agl_scale":    2500.0,   # m
    "z0_scale":     0.5,      # m (typical effective z0)
    "lat_scale":    90.0,     # deg (range [-90,90])
    "era5_u_offset": 0.0,
    "era5_u_scale": 15.0,
    "era5_v_offset": 0.0,
    "era5_v_scale": 15.0,
    "era5_T_offset": 270.0,
    "era5_T_scale":  20.0,
    "era5_q_offset": 0.0,
    "era5_q_scale":  0.01,
    "t2m_offset":    290.0,
    "t2m_scale":     10.0,
    "d2m_offset":    285.0,
    "d2m_scale":     10.0,
    "u10_offset":     0.0,
    "u10_scale":     10.0,
    "v10_offset":     0.0,
    "v10_scale":     10.0,
    "pressure_offset": 700.0,  # hPa centring
    "pressure_scale":  300.0,
}


def dewpoint_k_to_specific_humidity(dewpoint_k: np.ndarray, p_hpa: float = 1013.25) -> np.ndarray:
    """Specific humidity from dewpoint temperature and pressure.

    ERA5 surface input stores d2m but the surrogate target uses q. For a
    near-ground residual baseline, d2m gives vapour pressure directly.
    """
    td_c = np.asarray(dewpoint_k, dtype=np.float32) - 273.15
    e_hpa = 6.112 * np.exp(17.67 * td_c / (td_c + 243.5))
    return (0.622 * e_hpa / np.maximum(p_hpa - 0.378 * e_hpa, 1e-6)).astype(np.float32)


def build_era5_baseline_tensor(
    store: Any,
    norm: dict[str, float],
    nz: int,
    *,
    mode: str = "pressure_index",
) -> np.ndarray:
    """Build the normalised ERA5 residual baseline on the target grid.

    `pressure_index` is the legacy mode used by existing residual checkpoints:
    it interpolates pressure-level arrays by array index. This is retained for
    reproducibility only.

    `surface` is the near-ground mode for AGL/FWI models: it uses ERA5 surface
    u10/v10/t2m/d2m and broadcasts them over the near-surface target levels.
    """
    n = norm
    if mode == "pressure_index":
        def profile(var: str) -> np.ndarray:
            arr = np.asarray(store[f"input/era5_3d/{var}"][:], dtype=np.float32)
            prof_1d = arr[1, 1, :]
            k_idx = np.linspace(0, len(prof_1d) - 1, nz, dtype=np.float32)
            return np.interp(k_idx, np.arange(len(prof_1d), dtype=np.float32),
                             prof_1d).astype(np.float32)

        u = np.broadcast_to(profile("u")[None, None, :], (NI, NJ, nz)).copy()
        v = np.broadcast_to(profile("v")[None, None, :], (NI, NJ, nz)).copy()
        T = np.broadcast_to(profile("T")[None, None, :], (NI, NJ, nz)).copy()
        q = np.broadcast_to(profile("q")[None, None, :], (NI, NJ, nz)).copy()
    elif mode == "surface":
        surf = store["input/era5_surface"]
        u0 = float(np.asarray(surf["u10"][:], dtype=np.float32)[1, 1])
        v0 = float(np.asarray(surf["v10"][:], dtype=np.float32)[1, 1])
        T0 = float(np.asarray(surf["t2m"][:], dtype=np.float32)[1, 1])
        d2m0 = float(np.asarray(surf["d2m"][:], dtype=np.float32)[1, 1])
        q0 = float(dewpoint_k_to_specific_humidity(np.array([d2m0], dtype=np.float32))[0])
        u = np.full((NI, NJ, nz), u0, dtype=np.float32)
        v = np.full((NI, NJ, nz), v0, dtype=np.float32)
        T = np.full((NI, NJ, nz), T0, dtype=np.float32)
        q = np.full((NI, NJ, nz), q0, dtype=np.float32)
    else:
        raise ValueError(f"Unknown residual baseline mode: {mode}")

    w = np.zeros((NI, NJ, nz), dtype=np.float32)
    return np.stack([
        (u - n["U_x_offset"]) / n["U_uv_scale"],
        (v - n["U_y_offset"]) / n["U_uv_scale"],
        (w - n["U_z_offset"]) / n["U_w_scale"],
        (T - n["T_offset"]) / n["T_scale"],
        (q - n["q_offset"]) / n["q_scale"],
    ], axis=0).astype(np.float32)


def parse_agl_levels(levels: str | list[float] | tuple[float, ...] | np.ndarray | None) -> np.ndarray | None:
    """Parse an optional fixed-AGL target grid.

    Accepted strings:
      - "agl_0_100_24" / "fwi_0_100_24"
      - comma-separated levels, e.g. "0,2,5,10,20,50,100"
    """
    if levels is None:
        return None
    if isinstance(levels, str):
        key = levels.strip().lower()
        if key in {"", "native", "none"}:
            return None
        if key in {"agl_0_100_24", "fwi_0_100_24"}:
            arr = np.asarray(DEFAULT_AGL_0_100_24, dtype=np.float32)
        else:
            arr = np.asarray([float(x) for x in key.split(",") if x.strip()], dtype=np.float32)
    else:
        arr = np.asarray(levels, dtype=np.float32)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"Invalid AGL levels: {levels!r}")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"AGL levels contain non-finite values: {levels!r}")
    arr = np.asarray(sorted(float(x) for x in arr), dtype=np.float32)
    if np.any(np.diff(arr) <= 0):
        raise ValueError(f"AGL levels must be strictly increasing: {arr}")
    return arr


def resample_volume_to_agl_levels(
    volume: np.ndarray,
    native_agl: np.ndarray,
    target_agl_levels: np.ndarray,
) -> np.ndarray:
    """Linearly resample (C, NI, NJ, NK) fields onto fixed AGL levels."""
    if volume.shape[-3:] != native_agl.shape:
        raise ValueError(
            f"volume/native_agl shape mismatch: {volume.shape[-3:]} vs {native_agl.shape}"
        )
    levels = np.asarray(target_agl_levels, dtype=np.float32)
    out = np.empty((*volume.shape[:-1], levels.size), dtype=np.float32)
    native_agl = np.asarray(native_agl, dtype=np.float32)
    volume = np.asarray(volume, dtype=np.float32)

    for out_k, h in enumerate(levels):
        k1 = np.sum(native_agl < h, axis=-1)
        k1 = np.clip(k1, 1, native_agl.shape[-1] - 1).astype(np.int64)
        k0 = k1 - 1
        idx0 = k0[None, :, :, None]
        idx1 = k1[None, :, :, None]
        z0 = np.take_along_axis(native_agl, k0[:, :, None], axis=-1)[:, :, 0]
        z1 = np.take_along_axis(native_agl, k1[:, :, None], axis=-1)[:, :, 0]
        v0 = np.take_along_axis(volume, idx0, axis=-1)[..., 0]
        v1 = np.take_along_axis(volume, idx1, axis=-1)[..., 0]
        frac = np.clip((h - z0) / np.maximum(z1 - z0, 1e-6), 0.0, 1.0)
        out[..., out_k] = v0 + (v1 - v0) * frac[None, :, :]
    return out


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
        include_slopes: bool = False,
        use_residual: bool = False,
        residual_baseline_mode: str = "pressure_index",
        return_weight: bool = False,
        agl_weight_alpha: float = 0.0,
        agl_weight_height: float = 300.0,
        target_agl_levels: str | list[float] | tuple[float, ...] | np.ndarray | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.norm = {**DEFAULT_NORM, **(norm or {})}
        self.include_z = include_z
        self.include_slopes = include_slopes
        self.use_residual = use_residual
        self.residual_baseline_mode = residual_baseline_mode
        self.return_weight = return_weight
        self.agl_weight_alpha = agl_weight_alpha
        self.agl_weight_height = agl_weight_height
        self.target_agl_levels = parse_agl_levels(target_agl_levels)

        with open(splits_yaml) as f:
            splits = yaml.safe_load(f)
        if split not in splits:
            raise KeyError(f"split '{split}' not in {list(splits)}")
        site_ids: list[str] = splits[split]

        cases: list[Path] = []
        missing_sites = 0
        n_skipped = 0
        for sid in site_ids:
            site_cases = sorted(self.data_dir.glob(f"{sid}_case_ts*"))
            site_cases = [c for c in site_cases if (c / "grid.zarr").exists()]
            if not site_cases:
                missing_sites += 1
                continue
            for case_dir in site_cases:
                try:
                    g = zarr.open_group(str(case_dir / "grid.zarr"), mode="r")
                    tk = set(g["target"])
                    if {"U", "T", "q"}.issubset(tk):
                        cases.append(case_dir)
                    else:
                        n_skipped += 1
                except Exception:
                    n_skipped += 1
        self.cases = cases

        logger.info(
            "WindV2Dataset[%s]: %d cases from %d sites (missing %d sites, "
            "%d cases skipped for incomplete targets)",
            split, len(self.cases), len(site_ids) - missing_sites, missing_sites, n_skipped,
        )

    def __len__(self) -> int:
        return len(self.cases)

    # ── Helpers ────────────────────────────────────────────────────────────
    def _build_input_tensor(self, store: Any) -> np.ndarray:
        """Concat all inputs into (C_in, NI, NJ, NK) float32 tensor."""
        n = self.norm
        chans: list[np.ndarray] = []
        levels = self.target_agl_levels
        nz = NK if levels is None else int(levels.size)

        # 1. terrain (NI, NJ) → broadcast over k
        terrain = np.asarray(store["input/terrain"][:], dtype=np.float32)
        chans.append(np.broadcast_to(terrain[:, :, None], (NI, NJ, nz)).copy()
                     / n["terrain_scale"])

        if self.include_slopes:
            slope_y, slope_x = np.gradient(terrain, 33.333, 33.333)
            chans.append(np.broadcast_to(slope_x[:, :, None], (NI, NJ, nz)).copy()
                         .astype(np.float32))
            chans.append(np.broadcast_to(slope_y[:, :, None], (NI, NJ, nz)).copy()
                         .astype(np.float32))

        # 2. z (real altitude) (NI, NJ, NK) — explicit grid deformation
        native_z = np.asarray(store["coords/z"][:], dtype=np.float32)
        if levels is None:
            z = native_z
            agl = z - terrain[:, :, None]
        else:
            agl = np.broadcast_to(levels[None, None, :], (NI, NJ, nz)).copy()
            z = terrain[:, :, None] + agl
        if self.include_z:
            chans.append(z / n["z_scale"])

        # 3. AGL (z - terrain), more useful than absolute z for log-law
        chans.append(agl / n["agl_scale"])

        # 4. z0 (scalar) broadcast
        z0 = float(store["input"].attrs.get("z0_eff", 0.0))
        chans.append(np.full((NI, NJ, nz), z0 / n["z0_scale"], dtype=np.float32))

        # 5. lat (Coriolis) scalar broadcast
        lat = float(store["input"].attrs.get("lat", 0.0))
        chans.append(np.full((NI, NJ, nz), lat / n["lat_scale"], dtype=np.float32))

        # 6. ERA5 1D profile from centre of 3×3 (interpolate from N_p hPa to NK z-bins)
        plev = np.asarray(store["input/era5_pressure_levels"][:], dtype=np.float32)
        # Map ERA5 pressure profile to the NK levels using the column mean z.
        # We use the column-mean AGL as a proxy for k → height mapping.
        for var, scale, offset in [
            ("u", n["era5_u_scale"], n["era5_u_offset"]),
            ("v", n["era5_v_scale"], n["era5_v_offset"]),
            ("T", n["era5_T_scale"], n["era5_T_offset"]),
            ("q", n["era5_q_scale"], n["era5_q_offset"]),
        ]:
            era5_var = np.asarray(store[f"input/era5_3d/{var}"][:], dtype=np.float32)
            # Take centre of 3×3 grid: shape (N_p,)
            prof_1d = era5_var[1, 1, :]
            # Broadcast over (NI, NJ) and interpolate from N_p to the output z grid:
            # We linearly interpolate along k by mapping pressure to fractional index.
            # For simplicity, project on NK via uniform indexing — model can learn the mapping.
            k_idx = np.linspace(0, len(prof_1d) - 1, nz, dtype=np.float32)
            prof_nk = np.interp(k_idx, np.arange(len(prof_1d), dtype=np.float32),
                                prof_1d).astype(np.float32)
            field = np.broadcast_to(prof_nk[None, None, :], (NI, NJ, nz)).copy()
            chans.append(((field - offset) / scale).astype(np.float32))

        # 7. ERA5 pressure level profile itself broadcast on the output z grid
        plev_k = np.interp(np.linspace(0, len(plev) - 1, nz, dtype=np.float32),
                           np.arange(len(plev), dtype=np.float32), plev).astype(np.float32)
        chans.append(np.broadcast_to(
            ((plev_k - n["pressure_offset"]) / n["pressure_scale"])[None, None, :],
            (NI, NJ, nz)).copy().astype(np.float32))

        # 8. Surface ERA5 (3×3 → centre scalar) broadcast
        for var, scale, offset in [
            ("t2m", n["t2m_scale"], n["t2m_offset"]),
            ("d2m", n["d2m_scale"], n["d2m_offset"]),
            ("u10", n["u10_scale"], n["u10_offset"]),
            ("v10", n["v10_scale"], n["v10_offset"]),
        ]:
            arr = np.asarray(store[f"input/era5_surface/{var}"][:], dtype=np.float32)
            val = float(arr[1, 1])
            chans.append(np.full((NI, NJ, nz), (val - offset) / scale, dtype=np.float32))

        return np.stack(chans, axis=0).astype(np.float32)  # (C_in, NI, NJ, NK)

    def _build_target_tensor(self, store: Any) -> np.ndarray:
        """Build (5, NI, NJ, NK) target = (u, v, w, T, q) normalised."""
        n = self.norm
        U = np.asarray(store["target/U"][:], dtype=np.float32)  # (NI, NJ, NK, 3)
        T = np.asarray(store["target/T"][:], dtype=np.float32)
        q = np.asarray(store["target/q"][:], dtype=np.float32)
        target = np.stack([
            (U[..., 0] - n["U_x_offset"]) / n["U_uv_scale"],
            (U[..., 1] - n["U_y_offset"]) / n["U_uv_scale"],
            (U[..., 2] - n["U_z_offset"]) / n["U_w_scale"],
            (T - n["T_offset"]) / n["T_scale"],
            (q - n["q_offset"]) / n["q_scale"],
        ], axis=0)
        if self.target_agl_levels is None:
            return target
        terrain = np.asarray(store["input/terrain"][:], dtype=np.float32)
        z = np.asarray(store["coords/z"][:], dtype=np.float32)
        agl = z - terrain[:, :, None]
        return resample_volume_to_agl_levels(target, agl, self.target_agl_levels)

    def _build_era5_baseline_tensor(self, store: Any) -> np.ndarray:
        nz = NK if self.target_agl_levels is None else int(self.target_agl_levels.size)
        return build_era5_baseline_tensor(
            store,
            self.norm,
            nz,
            mode=self.residual_baseline_mode,
        )

    def _build_loss_weight(self, store: Any) -> np.ndarray:
        if self.target_agl_levels is None:
            terrain = np.asarray(store["input/terrain"][:], dtype=np.float32)
            z = np.asarray(store["coords/z"][:], dtype=np.float32)
            agl = np.maximum(z - terrain[:, :, None], 0.0)
        else:
            agl = np.broadcast_to(
                self.target_agl_levels[None, None, :],
                (NI, NJ, self.target_agl_levels.size),
            ).copy()
        alpha = float(self.agl_weight_alpha)
        height = max(float(self.agl_weight_height), 1.0)
        weight = 1.0 + alpha * np.exp(-agl / height)
        return weight[None, ...].astype(np.float32)

    def __getitem__(self, idx: int):
        case_dir = self.cases[idx]
        store = zarr.open_group(str(case_dir / "grid.zarr"), mode="r")
        inp = self._build_input_tensor(store)
        tgt = self._build_target_tensor(store)
        if self.use_residual:
            tgt = tgt - self._build_era5_baseline_tensor(store)
        if self.return_weight:
            weight = self._build_loss_weight(store)
            return (torch.from_numpy(inp), torch.from_numpy(tgt),
                    torch.from_numpy(weight), case_dir.name)
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
