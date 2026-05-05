"""Generate a demo .bin case file for the Plume 3D viewer.

Two modes:

1. Synthetic (default) — no model/data needed, great for frontend dev:
   python generate_demo_case.py --name synthetic --synthetic ridge

2. Real case export — loads a trained grid.zarr and writes the .bin file:
   python generate_demo_case.py --name perdigao --from-grid /path/to/case/grid.zarr

Binary layout (little-endian, exact bytes, all row-major):

    magic        4s     b"PLM2"
    nx           u32
    ny           u32
    nz           u32
    dx_m         f32    horizontal spacing (meters)
    dy_m         f32
    lat_center   f32    degrees
    lon_center   f32    degrees
    pad          8B     reserved (zero)
    z_levels     f32[nz]                 AGL heights (meters)
    terrain      f32[ny*nx]              elevation (meters, absolute)
    u            f32[ny*nx*nz]           wind u (m/s, eastward)
    v            f32[ny*nx*nz]           wind v (m/s, northward)
    w            f32[ny*nx*nz]           wind w (m/s, upward)

Header: 40 bytes fixed, then z_levels (nz*4), then fields.
Total size for 128x128x32: 40 + 128 + 65536 + 3*2097152 ≈ 6.37 MB raw
(≈ 1-2 MB after gzip thanks to smooth fields).
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


NX, NY, NZ = 128, 128, 32
DX = DY = 31.25  # meters — 4 km / 128
MAGIC = b"PLM2"


def log_z_levels(nz: int = NZ, z_min: float = 5.0, z_max: float = 5000.0) -> np.ndarray:
    return np.geomspace(z_min, z_max, nz).astype(np.float32)


def synthetic_ridge() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ridge terrain (Perdigão-like double ridge) with SW flow + lee recirculation.

    Not physically accurate — purely for viewer development.
    Returns (terrain[ny,nx], u[ny,nx,nz], v[ny,nx,nz], w[ny,nx,nz]).
    """
    x = np.linspace(-2000, 2000, NX, dtype=np.float32)
    y = np.linspace(-2000, 2000, NY, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing="xy")

    # Two parallel ridges oriented NW-SE
    theta = np.deg2rad(45.0)
    u_dir = np.array([np.cos(theta), np.sin(theta)])
    p_perp = -X * u_dir[1] + Y * u_dir[0]  # distance along perpendicular
    ridge1 = 300.0 * np.exp(-((p_perp - 400.0) ** 2) / (2 * 200.0 ** 2))
    ridge2 = 250.0 * np.exp(-((p_perp + 400.0) ** 2) / (2 * 200.0 ** 2))
    terrain = (200.0 + ridge1 + ridge2).astype(np.float32)

    z = log_z_levels()
    # Background log-law wind from SW (225°), speed 10 m/s at 60m
    z0 = 0.1
    u_star = 10.0 * 0.4 / np.log(60.0 / z0)
    speed_bg = (u_star / 0.4) * np.log(np.maximum(z, z0 + 1e-3) / z0)  # (nz,)

    u_hub_dir = np.deg2rad(225.0)  # wind coming FROM SW → blowing toward NE
    u_hub = -np.sin(u_hub_dir)     # eastward component
    v_hub = -np.cos(u_hub_dir)     # northward component

    u = (speed_bg[None, None, :] * u_hub).astype(np.float32)
    v = (speed_bg[None, None, :] * v_hub).astype(np.float32)
    u = np.broadcast_to(u, (NY, NX, NZ)).copy()
    v = np.broadcast_to(v, (NY, NX, NZ)).copy()

    # Terrain-induced speedup over the ridge crest (decreases with height)
    speedup = 1.0 + 0.5 * (terrain[:, :, None] - 200.0) / 300.0 * np.exp(-z[None, None, :] / 400.0)
    u *= speedup
    v *= speedup

    # Vertical velocity: lift upwind, sink downwind (proportional to along-flow slope)
    slope_x = np.gradient(terrain, DX, axis=1)
    slope_y = np.gradient(terrain, DY, axis=0)
    w_surface = u[:, :, 0] * slope_x + v[:, :, 0] * slope_y  # advective
    decay = np.exp(-z / 800.0)
    w = (w_surface[:, :, None] * decay[None, None, :]).astype(np.float32)

    return terrain, u, v, w


def write_bin(path: Path, terrain: np.ndarray, u: np.ndarray, v: np.ndarray, w: np.ndarray,
              z_levels: np.ndarray, lat_center: float, lon_center: float) -> None:
    assert terrain.shape == (NY, NX)
    assert u.shape == v.shape == w.shape == (NY, NX, NZ)
    assert z_levels.shape == (NZ,)

    header = MAGIC + struct.pack(
        "<IIIffffxxxxxxxx",
        NX, NY, NZ, float(DX), float(DY),
        float(lat_center), float(lon_center),
    )
    assert len(header) == 40, f"header is {len(header)} bytes, expected 40"

    with open(path, "wb") as f:
        f.write(header)
        f.write(z_levels.astype(np.float32).tobytes())
        f.write(terrain.astype(np.float32).tobytes())
        f.write(u.astype(np.float32).tobytes())
        f.write(v.astype(np.float32).tobytes())
        f.write(w.astype(np.float32).tobytes())

    size_mb = path.stat().st_size / 1e6
    print(f"[ok] wrote {path} ({size_mb:.2f} MB)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="Output file name (without .bin)")
    p.add_argument("--out-dir", type=Path, default=Path("app/demo_data"))
    p.add_argument("--synthetic", choices=["ridge"], default="ridge")
    p.add_argument("--from-grid", type=Path, help="Load from an existing grid.zarr (real case)")
    p.add_argument("--lat", type=float, default=39.7125, help="Domain center latitude (Perdigão default)")
    p.add_argument("--lon", type=float, default=-7.7386, help="Domain center longitude")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_grid:
        import zarr
        store = zarr.open_group(str(args.from_grid), mode="r")
        terrain = np.array(store["input/terrain"][:], dtype=np.float32)
        U = np.array(store["target/U"][:], dtype=np.float32)  # (ny,nx,nz,3)
        u, v, w = U[..., 0], U[..., 1], U[..., 2]
        # Reconstruct z_levels from stored coords if available
        if "coords/z_levels_agl" in store:
            z = np.array(store["coords/z_levels_agl"][:], dtype=np.float32)
        else:
            z = log_z_levels()
    else:
        terrain, u, v, w = synthetic_ridge()
        z = log_z_levels()

    out_path = args.out_dir / f"{args.name}.bin"
    write_bin(out_path, terrain, u, v, w, z, args.lat, args.lon)


if __name__ == "__main__":
    main()
