#!/usr/bin/env python3
"""
generate_uluru_demo.py — Uluru wind downscaling demo for website 3D viewer.

Downloads COP-DEM GLO-30 terrain (30 m), resamples at 3 resolutions,
and computes physically plausible wind fields via terrain perturbation model
(mass-conservation speed-up + directional wake + deflection).

Usage:
    conda run -n downscalewind python notebooks/generate_uluru_demo.py

Output (data/website/uluru/):
    terrain_100m.csv, terrain_1km.csv, terrain_10km.csv
    wind_cross_100m.csv  wind_along_100m.csv   (15 m/s from S / from W)
    wind_cross_1km.csv   wind_along_1km.csv
    wind_cross_10km.csv  wind_along_10km.csv
"""

from __future__ import annotations

import math
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

# ── Constants ────────────────────────────────────────────────────────────────

ULURU_LAT = -25.3444
ULURU_LON = 131.0369

# Domain: 20x20 km total
DOMAIN_HALF_KM = 10

# Earth geometry at Uluru latitude
DEG_TO_M_LAT = 111_320  # m per degree latitude (constant)
DEG_TO_M_LON = DEG_TO_M_LAT * math.cos(math.radians(abs(ULURU_LAT)))  # ~100 630 m

# COP-DEM GLO-30 (public, no auth)
COPDEM_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"

# Resolutions
RESOLUTIONS = [
    ("100m", 100),
    ("1km", 1000),
    ("10km", 10000),
]

# Wind conditions: (name, speed_m/s, direction_from_deg)
# direction_from: meteorological convention (0=N, 90=E, 180=S, 270=W)
# Uluru long axis ≈ WNW-ESE
#   cross = from south → hits the long face → dramatic speed-up + wake north
#   along = from west  → flows along length → wake east + flanking speed-up
WIND_CONDITIONS = [
    ("cross", 15.0, 180),  # from south
    ("along", 15.0, 270),  # from west
]

OUTDIR = Path(__file__).resolve().parents[1] / "data" / "website" / "uluru"


# ── Terrain download ─────────────────────────────────────────────────────────

def _copdem_url(lat_sw: int, lon_sw: int) -> str:
    lh = "N" if lat_sw >= 0 else "S"
    lnh = "E" if lon_sw >= 0 else "W"
    name = f"Copernicus_DSM_COG_10_{lh}{abs(lat_sw):02d}_00_{lnh}{abs(lon_sw):03d}_00_DEM"
    return f"{COPDEM_BASE}/{name}/{name}.tif"


def download_terrain(dest: Path) -> None:
    """Download and merge COP-DEM tiles covering the 20x20 km domain."""
    import rasterio
    import rasterio.merge
    from rasterio.transform import from_bounds

    margin_deg = DOMAIN_HALF_KM / 111.32 * 1.3  # generous margin
    north = ULURU_LAT + margin_deg
    south = ULURU_LAT - margin_deg
    east = ULURU_LON + margin_deg / math.cos(math.radians(abs(ULURU_LAT)))
    west = ULURU_LON - margin_deg / math.cos(math.radians(abs(ULURU_LAT)))

    tiles = [
        (lat, lon)
        for lat in range(math.floor(south), math.floor(north) + 1)
        for lon in range(math.floor(west), math.floor(east) + 1)
    ]
    print(f"  Tiles: {tiles}")

    with tempfile.TemporaryDirectory(prefix="copdem_uluru_") as tmpdir:
        paths = []
        for lat, lon in tiles:
            url = _copdem_url(lat, lon)
            fp = Path(tmpdir) / f"t_{lat}_{lon}.tif"
            print(f"  GET {url} ...")
            try:
                urllib.request.urlretrieve(url, fp)
                paths.append(fp)
            except Exception as e:
                print(f"  WARN: {e}")

        if not paths:
            raise RuntimeError("No COP-DEM tiles downloaded")

        datasets = [rasterio.open(str(p)) for p in paths]
        merged, transform = rasterio.merge.merge(datasets)
        profile = datasets[0].profile.copy()
        for ds in datasets:
            ds.close()

        # Clip to domain
        pw, ph = transform.a, transform.e
        left, top = transform.c, transform.f
        c0 = max(0, int((west - left) / pw))
        r0 = max(0, int((top - north) / (-ph)))
        c1 = min(merged.shape[2], int((east - left) / pw) + 1)
        r1 = min(merged.shape[1], int((top - south) / (-ph)) + 1)
        clipped = merged[:, r0:r1, c0:c1]
        new_left = left + c0 * pw
        new_top = top - r0 * (-ph)
        new_tf = from_bounds(
            new_left, new_top + ph * clipped.shape[1],
            new_left + pw * clipped.shape[2], new_top,
            clipped.shape[2], clipped.shape[1],
        )

        profile.update(
            driver="GTiff", height=clipped.shape[1], width=clipped.shape[2],
            count=1, dtype="float32", crs="EPSG:4326", transform=new_tf,
            compress="lzw",
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(str(dest), "w", **profile) as dst:
            dst.write(clipped[0].astype(np.float32), 1)

        print(f"  DEM: {dest} ({clipped.shape[1]}x{clipped.shape[2]} px)")


# ── Terrain resampling ────────────────────────────────────────────────────────

def resample_terrain(
    tif: Path, dx_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample DEM to target resolution. Returns (lon_1d, lat_1d, z_2d)."""
    import rasterio

    with rasterio.open(str(tif)) as src:
        data = src.read(1)
        tf = src.transform

    ncols, nrows = data.shape[1], data.shape[0]
    lons_nat = np.array([tf.c + (j + 0.5) * tf.a for j in range(ncols)])
    lats_nat = np.array([tf.f + (i + 0.5) * tf.e for i in range(nrows)])

    half_lat = DOMAIN_HALF_KM / 111.32
    half_lon = DOMAIN_HALF_KM / (111.32 * math.cos(math.radians(abs(ULURU_LAT))))
    step_lat = dx_m / DEG_TO_M_LAT
    step_lon = dx_m / DEG_TO_M_LON

    lon_t = np.arange(
        ULURU_LON - half_lon, ULURU_LON + half_lon + step_lon * 0.5, step_lon
    )
    lat_t = np.arange(
        ULURU_LAT + half_lat, ULURU_LAT - half_lat - step_lat * 0.5, -step_lat
    )

    # RegularGridInterpolator needs ascending axes
    interp = RegularGridInterpolator(
        (lats_nat[::-1], lons_nat),
        data[::-1],
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    lon_g, lat_g = np.meshgrid(lon_t, lat_t)
    z = interp((lat_g, lon_g))

    # Fill any NaN with mean
    if np.any(np.isnan(z)):
        z[np.isnan(z)] = np.nanmean(z)

    print(f"  {dx_m}m -> {z.shape[0]}x{z.shape[1]}, z=[{z.min():.0f},{z.max():.0f}]m")
    return lon_t, lat_t, z


# ── Wind model ────────────────────────────────────────────────────────────────

def compute_wind(
    z: np.ndarray,
    dx_m: float,
    speed: float,
    direction_from_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Terrain perturbation wind model.

    Physics:
      1. High-pass filter isolates terrain features (removes background slope)
      2. Mass conservation: flow accelerates where terrain constricts the BL
      3. Deflection: cross-wind pressure gradient pushes flow around obstacles
      4. Directional wake: velocity deficit propagated downstream of steep lee slopes
      5. Vertical velocity: kinematic terrain-slope condition

    Returns u (east), v (north), w (up) in m/s.
    """
    ny, nx = z.shape
    if ny < 3 or nx < 3:
        # Too few points for meaningful perturbation -> uniform
        dir_r = np.radians(direction_from_deg)
        u0 = -speed * np.sin(dir_r) * np.ones_like(z)
        v0 = -speed * np.cos(dir_r) * np.ones_like(z)
        return u0, v0, np.zeros_like(z)

    dir_r = np.radians(direction_from_deg)
    u0 = -speed * np.sin(dir_r)  # eastward freestream
    v0 = -speed * np.cos(dir_r)  # northward freestream

    # ── 1. Terrain features (high-pass) ──────────────────────────────────────
    bg_sigma = max(2, 8000 / dx_m)  # ~8 km smoothing -> background
    z_bg = gaussian_filter(z, sigma=bg_sigma)
    z_feat = z - z_bg  # positive = bump, negative = depression

    # ── 2. Speed-up (mass conservation) ──────────────────────────────────────
    # For an isolated obstacle, only a fraction alpha of the flow goes over
    # (rest goes around). alpha ~ 0.35 for Uluru (bluff body, Fr~4)
    H_eff = 1000.0  # effective BL depth [m]
    alpha = 0.35
    speedup = 1.0 + alpha * z_feat / (H_eff - alpha * z_feat)
    speedup = np.clip(speedup, 0.5, 2.0)
    speedup = gaussian_filter(speedup, sigma=max(1, 200 / dx_m))

    # ── 3. Terrain gradient ──────────────────────────────────────────────────
    dh_de = np.gradient(z, dx_m, axis=1)        # dh/d(east)
    dh_dn = -np.gradient(z, dx_m, axis=0)       # dh/d(north)

    # Unit vectors: wind direction and perpendicular (90 deg left)
    wu_e, wu_n = u0 / speed, v0 / speed          # along wind
    pp_e, pp_n = -wu_n, wu_e                      # perpendicular (left)

    # Slope along wind and cross-wind
    slope_along = dh_de * wu_e + dh_dn * wu_n
    slope_cross = dh_de * pp_e + dh_dn * pp_n

    # Deflection perpendicular to wind (pushes flow away from upslope)
    defl = -0.5 * slope_cross * speed
    defl = gaussian_filter(defl, sigma=max(1, 200 / dx_m))

    # ── 4. Directional wake ──────────────────────────────────────────────────
    # Source: terrain blockage from features
    blockage = np.clip(z_feat / 200.0, 0, 1.0)

    # Propagate downstream via iterative pixel shift
    # Downwind direction (where the wind GOES TO) in grid coords:
    #   east  -> +column
    #   north -> -row  (row 0 = north)
    dw_e = -np.sin(dir_r)  # downwind eastward
    dw_n = -np.cos(dir_r)  # downwind northward

    wake = np.zeros_like(z)
    wake_len_px = max(4, int(4000 / dx_m))  # ~4 km wake

    for d in range(1, wake_len_px + 1):
        di = round((-dw_n) * d)   # row shift (north = -row)
        dj = round(dw_e * d)      # col shift (east = +col)
        if abs(di) >= ny or abs(dj) >= nx:
            break
        shifted = np.roll(np.roll(blockage, di, axis=0), dj, axis=1)
        # Zero out wrapped edges
        if di > 0:
            shifted[:di, :] = 0
        elif di < 0:
            shifted[di:, :] = 0
        if dj > 0:
            shifted[:, :dj] = 0
        elif dj < 0:
            shifted[:, dj:] = 0
        decay = np.exp(-d * dx_m / 2000.0)
        wake += shifted * decay

    # Smooth laterally
    wake = gaussian_filter(wake, sigma=max(1, 500 / dx_m))
    if wake.max() > 0:
        wake = wake / wake.max() * 0.35  # max 35 % deficit

    # ── 5. Combine ───────────────────────────────────────────────────────────
    u = u0 * speedup * (1 - wake) + defl * pp_e
    v = v0 * speedup * (1 - wake) + defl * pp_n

    # ── 6. Vertical velocity ─────────────────────────────────────────────────
    # w ≈ U_horiz . grad(h)  (terrain-following kinematic BC)
    w = u * dh_de + v * dh_dn
    w = gaussian_filter(w, sigma=max(1, 150 / dx_m))

    # ── 7. Boundary taper (perturbations -> 0 at edges) ──────────────────────
    margin = max(2, min(8, ny // 5, nx // 5))
    taper = np.ones_like(z)
    ramp = np.linspace(0, 1, margin)
    taper[:margin, :] *= ramp[:, None]
    taper[-margin:, :] *= ramp[::-1][:, None]
    taper[:, :margin] *= ramp[None, :]
    taper[:, -margin:] *= ramp[::-1][None, :]

    # Apply taper to perturbation only (not freestream)
    u_pert = u - u0
    v_pert = v - v0
    u = u0 + u_pert * taper
    v = v0 + v_pert * taper
    w = w * taper

    return u, v, w


# ── CSV output ────────────────────────────────────────────────────────────────

def write_terrain_csv(path: Path, lon: np.ndarray, lat: np.ndarray, z: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("lon,lat,z\n")
        for i in range(len(lat)):
            for j in range(len(lon)):
                f.write(f"{lon[j]:.6f},{lat[i]:.6f},{z[i, j]:.1f}\n")


def write_wind_csv(
    path: Path, lon: np.ndarray, lat: np.ndarray,
    u: np.ndarray, v: np.ndarray, w: np.ndarray,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("lon,lat,u,v,w\n")
        for i in range(len(lat)):
            for j in range(len(lon)):
                f.write(f"{lon[j]:.6f},{lat[i]:.6f},{u[i,j]:.2f},{v[i,j]:.2f},{w[i,j]:.2f}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    dem_tif = OUTDIR / "copdem_uluru_30m.tif"

    # --- Step 1: Download terrain ---
    print("=" * 60)
    print("Step 1 — Download COP-DEM GLO-30 terrain")
    print("=" * 60)
    if dem_tif.exists():
        print(f"  Already exists: {dem_tif}")
    else:
        download_terrain(dem_tif)

    # --- Step 2: Resample + terrain CSVs ---
    print("\n" + "=" * 60)
    print("Step 2 — Resample terrain at 3 resolutions")
    print("=" * 60)
    grids: dict[str, tuple] = {}
    for label, dx in RESOLUTIONS:
        print(f"\n  [{label}]")
        lon, lat, z = resample_terrain(dem_tif, dx)
        grids[label] = (lon, lat, z, dx)
        csv_path = OUTDIR / f"terrain_{label}.csv"
        write_terrain_csv(csv_path, lon, lat, z)
        n = len(lon) * len(lat)
        print(f"  -> {csv_path.name}  ({n} points, {csv_path.stat().st_size/1024:.1f} KB)")

    # --- Step 3: Wind fields ---
    print("\n" + "=" * 60)
    print("Step 3 — Compute wind fields")
    print("=" * 60)
    for cond_name, wspd, wdir in WIND_CONDITIONS:
        print(f"\n  === {cond_name} ({wspd} m/s from {wdir} deg) ===")
        for label, dx in RESOLUTIONS:
            lon, lat, z, _ = grids[label]
            u, v, w = compute_wind(z, dx, wspd, wdir)

            # Stats
            spd = np.sqrt(u**2 + v**2)
            print(f"  [{label}] speed: mean={spd.mean():.1f}, "
                  f"max={spd.max():.1f}, min={spd.min():.1f} m/s")

            csv_path = OUTDIR / f"wind_{cond_name}_{label}.csv"
            write_wind_csv(csv_path, lon, lat, u, v, w)
            print(f"         -> {csv_path.name} ({csv_path.stat().st_size/1024:.1f} KB)")

    # --- Summary ---
    print("\n" + "=" * 60)
    print(f"Done! Files in {OUTDIR}/")
    print("=" * 60)
    total_kb = 0
    for p in sorted(OUTDIR.glob("*.csv")):
        sz = p.stat().st_size / 1024
        total_kb += sz
        print(f"  {p.name:30s} {sz:8.1f} KB")
    print(f"  {'TOTAL':30s} {total_kb:8.1f} KB")


if __name__ == "__main__":
    main()
