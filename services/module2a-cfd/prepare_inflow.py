"""
prepare_inflow.py — ERA5 → 3-layer ABL inlet profile for buoyantSimpleFoam

Reconstructs a physically consistent vertical wind profile from ERA5 reanalysis
data to use as the inlet boundary condition for the CFD simulation.

Three-layer reconstruction
--------------------------
  0–100 m   : log law + Monin-Obukhov similarity (Businger-Dyer corrections)
              u* estimated from ERA5 10-m wind or lowest pressure level
  100 m–2 km: cubic-spline interpolation on ERA5 pressure levels
  > 2 km    : ERA5 values directly

The resulting profile is written as a JSON file consumed by the Jinja2
templates (U.j2, k.j2, omega.j2) to set `atmBoundaryLayerInletVelocity`
parameters.

Usage (CLI)
-----------
    python prepare_inflow.py \
        --era5   data/raw/era5_perdigao.zarr \
        --z0map  data/raw/z0_perdigao.tif \
        --site   perdigao \
        --case   2017-05-15T12:00 \
        --output data/processed/inflow_2017-05-15T12.json

Output JSON
-----------
{
  "u_hub"      : float,   # m/s at hub height (80 m)
  "u_star"     : float,   # friction velocity [m/s]
  "z0_eff"     : float,   # effective roughness length [m] (geometric mean over upstream patch)
  "L_mo"       : float,   # Monin-Obukhov length [m] (inf = neutral)
  "flowDir_x"  : float,   # unit vector, eastward component
  "flowDir_y"  : float,   # unit vector, northward component
  "T_ref"      : float,   # reference temperature [K] at hub height
  "Ri_b"       : float,   # bulk Richardson number (stability indicator)
  "z_levels"   : [...],   # height array [m] used for profile table
  "u_profile"  : [...],   # speed [m/s] at z_levels
  "T_profile"  : [...],   # temperature [K] at z_levels
}
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
KAPPA      = 0.41     # von Kármán constant
G          = 9.81     # m s-2
RD         = 287.05   # J kg-1 K-1  (dry air)
T0         = 288.15   # K reference temperature (ISA)
P0         = 101325.0 # Pa reference pressure
CP         = 1005.0   # J kg-1 K-1  specific heat at constant pressure
HUB_HEIGHT = 80.0     # m  (hub height for normalisation)
Z_SURF     = 10.0     # m  ERA5 10-m wind reference height

# Layer boundaries
Z_LAYER1_TOP = 100.0   # m  — top of log-law layer
Z_LAYER2_TOP = 2000.0  # m  — top of cubic-spline layer (above = ERA5 direct)


# ---------------------------------------------------------------------------
# Monin-Obukhov stability correction functions (Businger-Dyer)
# ---------------------------------------------------------------------------

def psi_m(zeta: float | np.ndarray) -> float | np.ndarray:
    """Integrated stability correction for momentum (Businger-Dyer).

    Parameters
    ----------
    zeta:
        z / L_mo — positive = stable, negative = unstable.

    Returns
    -------
    Ψ_m(ζ) — dimensionless stability correction.
    """
    zeta = np.asarray(zeta, dtype=float)
    psi = np.zeros_like(zeta)

    # Stable: ζ > 0  (Dyer 1974)
    stable = zeta > 0
    psi[stable] = -5.0 * zeta[stable]

    # Unstable: ζ < 0  (Businger 1971, Paulson 1970)
    unstable = zeta < 0
    x = (1.0 - 16.0 * zeta[unstable]) ** 0.25
    psi[unstable] = (
        2.0 * np.log((1.0 + x) / 2.0)
        + np.log((1.0 + x**2) / 2.0)
        - 2.0 * np.arctan(x)
        + np.pi / 2.0
    )

    return float(psi) if psi.ndim == 0 else psi


def monin_obukhov_length(u_star: float, T_ref: float, H0: float) -> float:
    """Compute Monin-Obukhov length L from surface heat flux.

    L = - u_star³ ρ cp T / (κ g H0)

    Parameters
    ----------
    u_star:
        Friction velocity [m/s].
    T_ref:
        Near-surface temperature [K].
    H0:
        Surface kinematic heat flux [K m/s]  (positive = upward = unstable).
        ERA5 "surface sensible heat flux" is in W/m², divide by ρ·cp ≈ 1220.

    Returns
    -------
    L [m] — positive stable, negative unstable, inf neutral.
    """
    if abs(H0) < 1e-6:
        return np.inf
    rho = P0 / (RD * T_ref)
    L = -(u_star**3 * rho * CP * T_ref) / (KAPPA * G * H0 * rho * CP)
    return float(L)


# ---------------------------------------------------------------------------
# Log-law profile with Monin-Obukhov corrections
# ---------------------------------------------------------------------------

def log_law_speed(
    z: np.ndarray,
    u_star: float,
    z0: float,
    L_mo: float = np.inf,
) -> np.ndarray:
    """Compute wind speed u(z) from the MOST log-law with MO correction.

    u(z) = (u* / κ) · [ln(z/z₀) - Ψ_m(z/L) + Ψ_m(z₀/L)]

    Parameters
    ----------
    z:
        Heights above ground [m].
    u_star:
        Friction velocity [m/s].
    z0:
        Aerodynamic roughness length [m].
    L_mo:
        Monin-Obukhov length [m].  Use np.inf for neutral.

    Returns
    -------
    u [m/s] at each z, clipped to ≥ 0.
    """
    z = np.asarray(z, dtype=float)
    z = np.maximum(z, z0 * 1.01)  # avoid log(0)

    if np.isinf(L_mo):
        psi = 0.0
        psi0 = 0.0
    else:
        psi = psi_m(z / L_mo)
        psi0 = psi_m(z0 / L_mo)

    speed = (u_star / KAPPA) * (np.log(z / z0) - psi + psi0)
    return np.maximum(speed, 0.0)


def estimate_ustar(
    u_ref: float,
    z_ref: float,
    z0: float,
    L_mo: float = np.inf,
) -> float:
    """Back-calculate u* from a reference speed at z_ref.

    Solves u_ref = (u*/κ)·[ln(z_ref/z₀) - Ψ_m(z_ref/L) + Ψ_m(z₀/L)]

    Parameters
    ----------
    u_ref:
        Reference wind speed [m/s].
    z_ref:
        Reference height [m].
    z0:
        Roughness length [m].
    L_mo:
        Monin-Obukhov length [m] (use np.inf for neutral).

    Returns
    -------
    u_star [m/s].
    """
    if np.isinf(L_mo):
        phi = np.log(z_ref / z0)
    else:
        phi = np.log(z_ref / z0) - psi_m(z_ref / L_mo) + psi_m(z0 / L_mo)

    phi = max(phi, 0.1)  # guard against division by zero / negative
    return float(KAPPA * u_ref / phi)


# ---------------------------------------------------------------------------
# ERA5 extraction
# ---------------------------------------------------------------------------

def _bilinear_weights(
    lats: np.ndarray,
    lons: np.ndarray,
    target_lat: float,
    target_lon: float,
) -> tuple[int, int, int, int, float, float]:
    """Return 4 corner indices + bilinear weights.

    Handles both ascending and descending coordinate arrays
    (ERA5 convention: lat[0] = North, i.e. descending).
    """
    # searchsorted requires ascending arrays
    if lats[0] > lats[-1]:  # descending (ERA5 lat convention)
        lats_asc = lats[::-1]
        i0_asc = int(np.searchsorted(lats_asc, target_lat)) - 1
        i0_asc = max(0, min(i0_asc, len(lats) - 2))
        i0 = len(lats) - 2 - i0_asc  # map back to descending index
    else:
        i0 = int(np.searchsorted(lats, target_lat)) - 1
    i0 = max(0, min(i0, len(lats) - 2))

    if lons[0] > lons[-1]:  # descending (unlikely for ERA5 but defensive)
        lons_asc = lons[::-1]
        j0_asc = int(np.searchsorted(lons_asc, target_lon)) - 1
        j0_asc = max(0, min(j0_asc, len(lons) - 2))
        j0 = len(lons) - 2 - j0_asc
    else:
        j0 = int(np.searchsorted(lons, target_lon)) - 1
    j0 = max(0, min(j0, len(lons) - 2))

    i1, j1 = i0 + 1, j0 + 1

    wlat = float((target_lat - lats[i0]) / (lats[i1] - lats[i0]))
    wlon = float((target_lon - lons[j0]) / (lons[j1] - lons[j0]))
    # Clamp weights to [0, 1] for safety
    wlat = max(0.0, min(1.0, wlat))
    wlon = max(0.0, min(1.0, wlon))
    return i0, i1, j0, j1, wlat, wlon


def _apply_bilinear(
    field: np.ndarray,
    i0: int, i1: int, j0: int, j1: int,
    wlat: float, wlon: float,
) -> np.ndarray:
    """Bilinear interpolation in lat/lon (last two axes)."""
    return (
        (1 - wlat) * (1 - wlon) * field[..., i0, j0]
        + (1 - wlat) * wlon       * field[..., i0, j1]
        + wlat       * (1 - wlon) * field[..., i1, j0]
        + wlat       * wlon       * field[..., i1, j1]
    )


def extract_era5_profile(
    era5_data: dict,
    timestamp: np.datetime64,
    lat: float,
    lon: float,
) -> dict:
    """Extract ERA5 vertical profile at a given timestamp and location.

    Parameters
    ----------
    era5_data:
        Dict as returned by shared.data_io with keys:
        times, pressure_levels, lats, lons, u, v, t, z (geopotential).
    timestamp:
        Target timestamp (nearest ERA5 time step is used).
    lat, lon:
        Target location [°N, °E].

    Returns
    -------
    dict with keys: z_m, u_ms, v_ms, T_K, pressure_hPa
        Vertical arrays ordered from surface to top.
    """
    times = era5_data["times"]  # array of np.datetime64

    # Nearest time index
    dt = np.abs(times - timestamp)
    t_idx = int(np.argmin(dt))
    if dt[t_idx] > np.timedelta64(4, "h"):
        warnings.warn(
            f"ERA5 nearest time is {dt[t_idx]/np.timedelta64(1,'h'):.1f} h "
            f"away from requested {timestamp}",
            stacklevel=2,
        )

    lats = era5_data["lats"]
    lons = era5_data["lons"]
    i0, i1, j0, j1, wlat, wlon = _bilinear_weights(lats, lons, lat, lon)

    def interp(field4d: np.ndarray) -> np.ndarray:
        """shape (time, level, lat, lon) → (level,) at t_idx, lat, lon"""
        return _apply_bilinear(field4d[t_idx], i0, i1, j0, j1, wlat, wlon)

    u_lev = interp(era5_data["u"])   # [level]
    v_lev = interp(era5_data["v"])
    T_lev = interp(era5_data["t"])
    z_lev = interp(era5_data["z"]) / G  # geopotential → geometric height [m]

    pressure_hPa = np.asarray(era5_data["pressure_levels"], dtype=float)

    # Sort from surface (high pressure) to top (low pressure)
    order = np.argsort(-pressure_hPa)
    return {
        "z_m":          z_lev[order],
        "u_ms":         u_lev[order],
        "v_ms":         v_lev[order],
        "T_K":          T_lev[order],
        "pressure_hPa": pressure_hPa[order],
    }


# ---------------------------------------------------------------------------
# Effective roughness length from raster
# ---------------------------------------------------------------------------

def read_z0_effective(
    z0_tif: Path | str,
    site_lat: float,
    site_lon: float,
    upstream_dir_deg: float,
    patch_radius_m: float = 5000.0,
) -> float:
    """Read effective roughness length from GeoTIFF raster.

    Computes the geometric mean of z₀ values in a circular patch upstream of
    the site (radius = patch_radius_m metres, centred on the point
    patch_radius_m in the upwind direction from the site).

    Falls back to 0.05 m if the raster cannot be read.

    Parameters
    ----------
    z0_tif:
        Path to z₀ raster (GeoTIFF, values in metres).
    site_lat, site_lon:
        Site coordinates [°].
    upstream_dir_deg:
        Wind *coming from* this azimuth [°].  The upstream patch is placed
        in this direction from the site.
    patch_radius_m:
        Radius of the upstream circular patch [m].

    Returns
    -------
    z0_eff [m] — geometric mean (or 0.05 m on error).
    """
    try:
        import rasterio
        from rasterio.transform import rowcol
    except ImportError:
        logger.warning("rasterio not installed — using z0_eff = 0.05 m")
        return 0.05

    z0_tif = Path(z0_tif)
    if not z0_tif.exists():
        logger.warning("z0 raster not found: %s — using z0_eff = 0.05 m", z0_tif)
        return 0.05

    try:
        # Upstream patch centre (1° lat ≈ 111 km)
        DEG_PER_M_LAT = 1.0 / 111_000.0
        DEG_PER_M_LON = 1.0 / (111_000.0 * np.cos(np.radians(site_lat)))

        dir_rad = np.radians(upstream_dir_deg)  # wind coming FROM this dir
        upstream_lat = site_lat + patch_radius_m * np.cos(dir_rad) * DEG_PER_M_LAT
        upstream_lon = site_lon + patch_radius_m * np.sin(dir_rad) * DEG_PER_M_LON

        with rasterio.open(z0_tif) as src:
            # Sample a small window around the upstream point
            row, col = rowcol(src.transform, upstream_lon, upstream_lat)
            half = max(1, int(patch_radius_m / (2 * abs(src.res[0]))))
            window = rasterio.windows.Window(
                col - half, row - half, 2 * half, 2 * half
            )
            z0_patch = src.read(1, window=window).flatten()

        z0_patch = z0_patch[(z0_patch > 0) & np.isfinite(z0_patch)]
        if len(z0_patch) == 0:
            return 0.05

        z0_eff = float(np.exp(np.mean(np.log(z0_patch))))
        logger.debug("z0_eff = %.4f m (patch mean over %d pixels)", z0_eff, len(z0_patch))
        return z0_eff

    except Exception as exc:
        logger.warning("Could not read z0 raster: %s — using 0.05 m", exc)
        return 0.05


# ---------------------------------------------------------------------------
# Main reconstruction function
# ---------------------------------------------------------------------------

def reconstruct_inlet_profile(
    era5_profile: dict,
    z0_eff: float,
    L_mo: float = np.inf,
    z_output: np.ndarray | None = None,
) -> dict:
    """Build a 3-layer ABL vertical profile from an ERA5 column.

    Layer 1 (0–100 m)    : log law + MO correction
    Layer 2 (100 m–2 km) : cubic spline through ERA5 levels
    Layer 3 (> 2 km)     : ERA5 directly

    Parameters
    ----------
    era5_profile:
        Dict from extract_era5_profile().
    z0_eff:
        Effective roughness length [m].
    L_mo:
        Monin-Obukhov length [m] (default np.inf = neutral).
    z_output:
        Height grid [m] for the output profile.  Defaults to a log-spaced
        grid from z0_eff to the top of the ERA5 column.

    Returns
    -------
    dict with keys: z_m, u_ms, v_ms, T_K, speed_ms, dir_deg
    """
    z_era5 = era5_profile["z_m"]
    u_era5 = era5_profile["u_ms"]
    v_era5 = era5_profile["v_ms"]
    T_era5 = era5_profile["T_K"]

    # Speed and direction from ERA5 surface reference
    u_ref_10m = float(u_era5[0])  # ~lowest ERA5 level (~1000 hPa, ~100–200 m AGL at Perdigão)
    v_ref_10m = float(v_era5[0])
    spd_ref   = float(np.hypot(u_ref_10m, v_ref_10m))

    # Flow direction: unit vector (we keep the ERA5 lowest-level direction)
    if spd_ref < 0.5:
        flow_dir_x, flow_dir_y = 1.0, 0.0  # fallback: westerly
        logger.warning("ERA5 surface wind speed < 0.5 m/s — using default W flow direction")
    else:
        flow_dir_x = u_ref_10m / spd_ref
        flow_dir_y = v_ref_10m / spd_ref

    # Reference temperature (lowest ERA5 level)
    T_ref = float(T_era5[0])

    # u* from ERA5 lowest level
    z_ref_era5 = float(z_era5[0])
    z_ref_era5 = max(z_ref_era5, Z_SURF)
    u_star = estimate_ustar(spd_ref, z_ref_era5, z0_eff, L_mo)
    u_star = max(u_star, 0.05)  # physical lower bound

    logger.debug(
        "u_star=%.3f m/s, z0_eff=%.4f m, L_mo=%s m, T_ref=%.1f K",
        u_star, z0_eff,
        f"{L_mo:.0f}" if not np.isinf(L_mo) else "inf",
        T_ref,
    )

    # Output height grid
    if z_output is None:
        z_log   = np.logspace(np.log10(max(z0_eff * 5, 1.0)), np.log10(Z_LAYER1_TOP), 30)
        z_mid   = np.linspace(Z_LAYER1_TOP, Z_LAYER2_TOP, 50)[1:]
        z_upper = z_era5[z_era5 > Z_LAYER2_TOP]
        z_output = np.concatenate([z_log, z_mid, z_upper])
    z_output = np.sort(np.unique(z_output))

    # -----------
    # Layer 1 : log law (0 → Z_LAYER1_TOP)
    # -----------
    mask1 = z_output <= Z_LAYER1_TOP
    z1    = z_output[mask1]
    spd1  = log_law_speed(z1, u_star, z0_eff, L_mo)

    # -----------
    # Layer 2 : cubic spline (Z_LAYER1_TOP → Z_LAYER2_TOP)
    # -----------
    # Include the top of layer 1 as a boundary condition
    era5_in_layer2 = (z_era5 >= Z_LAYER1_TOP) & (z_era5 <= Z_LAYER2_TOP + 500)
    z_knots   = np.concatenate([[Z_LAYER1_TOP], z_era5[era5_in_layer2]])
    spd_era5  = np.hypot(u_era5, v_era5)
    spd_knots = np.concatenate([
        [float(log_law_speed(np.array([Z_LAYER1_TOP]), u_star, z0_eff, L_mo))],
        spd_era5[era5_in_layer2],
    ])

    z_knots, unique_idx = np.unique(z_knots, return_index=True)
    spd_knots = spd_knots[unique_idx]

    if len(z_knots) >= 2:
        cs_spd = CubicSpline(z_knots, spd_knots, extrapolate=True)
    else:
        cs_spd = lambda z: spd_knots[0] * np.ones_like(z)  # noqa

    mask2 = (z_output > Z_LAYER1_TOP) & (z_output <= Z_LAYER2_TOP)
    z2    = z_output[mask2]
    spd2  = cs_spd(z2)
    spd2  = np.maximum(spd2, 0.0)

    # -----------
    # Layer 3 : ERA5 direct (above Z_LAYER2_TOP)
    # -----------
    mask3 = z_output > Z_LAYER2_TOP
    z3    = z_output[mask3]
    spd_era5_full = np.hypot(u_era5, v_era5)
    if len(z_era5) >= 2:
        cs_upper = CubicSpline(z_era5, spd_era5_full, extrapolate=True)
        spd3 = np.maximum(cs_upper(z3), 0.0)
    else:
        spd3 = np.full_like(z3, spd_era5_full[-1])

    # Temperature profile (cubic spline through ERA5 levels)
    if len(z_era5) >= 2:
        cs_T = CubicSpline(z_era5, T_era5, extrapolate=True)
        T_out = cs_T(z_output)
    else:
        T_out = np.full_like(z_output, T_ref)

    # Combine all layers
    speed_out = np.concatenate([spd1, spd2, spd3])
    u_out     = speed_out * flow_dir_x
    v_out     = speed_out * flow_dir_y

    # Values at hub height
    hub_mask = np.searchsorted(z_output, HUB_HEIGHT)
    if hub_mask >= len(z_output):
        hub_mask = len(z_output) - 1
    u_hub = float(speed_out[hub_mask])
    T_hub = float(T_out[hub_mask])

    return {
        "z_m":        z_output.tolist(),
        "u_ms":       u_out.tolist(),
        "v_ms":       v_out.tolist(),
        "speed_ms":   speed_out.tolist(),
        "T_K":        T_out.tolist(),
        "u_hub":      u_hub,
        "u_star":     float(u_star),
        "z0_eff":     float(z0_eff),
        "L_mo":       float(L_mo) if not np.isinf(L_mo) else None,
        "T_ref":      T_hub,
        "flowDir_x":  float(flow_dir_x),
        "flowDir_y":  float(flow_dir_y),
    }


# ---------------------------------------------------------------------------
# Bulk Richardson number (for stability diagnostics)
# ---------------------------------------------------------------------------

def bulk_richardson_number(
    era5_profile: dict,
    z_lower: float = 100.0,
    z_upper: float = 500.0,
) -> float:
    """Estimate bulk Ri_b between two heights.

    Ri_b = (g/T_m) · (ΔT/Δz) / (Δu/Δz)²

    Returns np.inf if wind shear is negligible.
    """
    z = era5_profile["z_m"]
    u = np.hypot(era5_profile["u_ms"], era5_profile["v_ms"])
    T = era5_profile["T_K"]

    idx_lo = int(np.argmin(np.abs(np.array(z) - z_lower)))
    idx_hi = int(np.argmin(np.abs(np.array(z) - z_upper)))

    dz     = z[idx_hi] - z[idx_lo]
    du     = u[idx_hi] - u[idx_lo]
    dT     = T[idx_hi] - T[idx_lo]
    T_mean = 0.5 * (T[idx_lo] + T[idx_hi])

    if abs(du) < 0.5:
        return np.inf if dT >= 0 else -np.inf

    Ri_b = (G / T_mean) * (dT / dz) / (du / dz) ** 2
    return float(Ri_b)


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

def prepare_inflow(
    era5_zarr: Path | str,
    timestamp: str,
    site_lat: float,
    site_lon: float,
    z0_tif: Path | str | None = None,
    output_json: Path | str | None = None,
) -> dict:
    """Full pipeline: ERA5 zarr → inlet profile JSON.

    Parameters
    ----------
    era5_zarr:
        Path to ERA5 zarr store (shared.data_io format).
    timestamp:
        ISO-8601 timestamp string (e.g. "2017-05-15T12:00").
    site_lat, site_lon:
        Site coordinates [°N, °E].
    z0_tif:
        Optional path to z₀ raster (GeoTIFF).  Falls back to 0.05 m.
    output_json:
        Optional path to write the output JSON.

    Returns
    -------
    dict — inlet profile (same as reconstruct_inlet_profile output + Ri_b).
    """
    import zarr  # lazy import

    logger.info("Loading ERA5 from %s", era5_zarr)
    store = zarr.open_group(str(era5_zarr), mode="r")

    # DownscaleWind zarr schema: coords/time (int64 ns), coords/level, coords/lat, coords/lon
    # pressure/{u,v,t,z,q} — shape [time, level, lat, lon]
    times_int64 = np.array(store["coords/time"][:])
    times_ns = times_int64.astype("datetime64[ns]") if times_int64.dtype.kind in ("i","u") \
               else times_int64.astype("datetime64[ns]")
    era5_data = {
        "times":           times_ns.astype("datetime64[s]"),
        "pressure_levels": store["coords/level"][:],
        "lats":            store["coords/lat"][:],
        "lons":            store["coords/lon"][:],
        "u":               store["pressure/u"][:],
        "v":               store["pressure/v"][:],
        "t":               store["pressure/t"][:],
        "z":               store["pressure/z"][:],
    }

    ts = np.datetime64(timestamp, "s")

    logger.info("Extracting ERA5 profile at %s, lat=%.3f lon=%.3f", timestamp, site_lat, site_lon)
    profile = extract_era5_profile(era5_data, ts, site_lat, site_lon)

    # Effective roughness length
    if z0_tif is not None:
        # Infer upstream direction from ERA5 lowest-level wind
        u_lo = float(profile["u_ms"][0])
        v_lo = float(profile["v_ms"][0])
        wind_from_deg = float((270.0 - np.degrees(np.arctan2(v_lo, u_lo))) % 360)
        z0_eff = read_z0_effective(z0_tif, site_lat, site_lon, wind_from_deg)
    else:
        logger.info("No z0 raster provided — using z0_eff = 0.05 m")
        z0_eff = 0.05

    # Stability (neutral for now; can be extended with ERA5 surface heat flux)
    Ri_b = bulk_richardson_number(profile)
    L_mo = np.inf  # neutral default; extend with ERA5 H0 if available
    if abs(Ri_b) > 0.25:
        logger.warning("Ri_b = %.2f > 0.25 — flow is not near-neutral; L_mo set to inf", Ri_b)

    # Reconstruct profile
    result = reconstruct_inlet_profile(profile, z0_eff=z0_eff, L_mo=L_mo)
    result["Ri_b"] = Ri_b
    result["z_levels"] = result.pop("z_m")
    result["u_profile"] = result.pop("speed_ms")
    result["T_profile"] = result.pop("T_K")

    # Fields expected by Jinja2 templates (aliased / derived)
    result["z0"]       = float(z0_eff)                         # alias of z0_eff
    result["kappa"]    = 0.4                                   # von Kármán constant
    result["d"]        = 0.0                                   # displacement height (flat)
    result["z_ref"]    = float(result.get("z_ref", 80.0))      # reference height for u_hub

    # Wind direction FROM North [degrees] derived from flow vector
    fx = float(result.get("flowDir_x", 0.0))
    fy = float(result.get("flowDir_y", 0.0))
    # flowDir = direction wind blows TOWARD; met convention = direction wind comes FROM
    # wind_from = atan2(-fx, -fy) in degrees, measured clockwise from North
    import math
    wind_dir_deg = (math.degrees(math.atan2(-fx, -fy)) + 360) % 360
    result["wind_dir"] = round(wind_dir_deg, 1)

    if output_json is not None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Inflow profile written to %s", output_json)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Reconstruct ERA5 → ABL inlet profile for buoyantSimpleFoam"
    )
    parser.add_argument("--era5",    required=True, help="ERA5 zarr store")
    parser.add_argument("--z0map",   default=None,  help="z0 raster (GeoTIFF)")
    parser.add_argument("--site",    default="perdigao",
                        help="Site name (used to look up lat/lon in perdigao.yaml)")
    parser.add_argument("--lat",     type=float, default=None, help="Override site lat")
    parser.add_argument("--lon",     type=float, default=None, help="Override site lon")
    parser.add_argument("--case",    required=True,
                        help="Timestamp ISO-8601 (e.g. 2017-05-15T12:00)")
    parser.add_argument("--output",  default=None, help="Output JSON path")
    args = parser.parse_args()

    # Resolve site coords
    if args.lat is not None and args.lon is not None:
        lat, lon = args.lat, args.lon
    else:
        import yaml
        cfg_path = (
            Path(__file__).parents[2]
            / "configs" / "sites" / f"{args.site}.yaml"
        )
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        lat = cfg["site"]["coordinates"]["latitude"]
        lon = cfg["site"]["coordinates"]["longitude"]

    result = prepare_inflow(
        era5_zarr=args.era5,
        timestamp=args.case,
        site_lat=lat,
        site_lon=lon,
        z0_tif=args.z0map,
        output_json=args.output,
    )

    print(f"u_hub    = {result['u_hub']:.2f} m/s")
    print(f"u_star   = {result['u_star']:.3f} m/s")
    print(f"z0_eff   = {result['z0_eff']:.4f} m")
    print(f"Ri_b     = {result['Ri_b']:.3f}")
    print(f"T_ref    = {result['T_ref']:.1f} K")
    print(f"flowDir  = ({result['flowDir_x']:.3f}, {result['flowDir_y']:.3f})")
