"""Fetch ERA5/IFS profiles via Open-Meteo and interpolate onto 32 AGL levels.

Open-Meteo provides hourly pressure-level winds/temperature/geopotential/humidity
with no auth required. We request the nearest grid point to (lat, lon) and
interpolate the pressure levels onto our log-spaced AGL grid (5 m → 5000 m).

Endpoint (hourly, pressure levels):
    https://api.open-meteo.com/v1/forecast
       ?latitude=..&longitude=..
       &hourly=wind_speed_1000hPa,wind_direction_1000hPa,temperature_1000hPa,
              geopotential_height_1000hPa,relative_humidity_1000hPa,
              (... for 925, 850, 700, 500, 300, 250, 200, 150, 100 ...)
       &start_hour=..&end_hour=..

For historical timestamps, Open-Meteo exposes /v1/era5 instead (same schema).
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import numpy as np

from ..config import settings

PRESSURE_LEVELS = [1000, 925, 850, 700, 500, 300, 250, 200, 150, 100]
VARS = [
    "wind_speed",
    "wind_direction",
    "temperature",
    "geopotential_height",
    "relative_humidity",
]


def _build_hourly_query() -> str:
    parts: list[str] = []
    for v in VARS:
        for lvl in PRESSURE_LEVELS:
            parts.append(f"{v}_{lvl}hPa")
    return ",".join(parts)


def _rh_to_specific_humidity(rh_pct: np.ndarray, T_K: np.ndarray, p_hPa: np.ndarray) -> np.ndarray:
    """Rough Tetens-based conversion RH (%) + T (K) + p (hPa) → q (kg/kg)."""
    T_C = T_K - 273.15
    es_hPa = 6.112 * np.exp(17.67 * T_C / (T_C + 243.5))   # saturation vapor pressure
    e_hPa = (rh_pct / 100.0) * es_hPa
    eps = 0.622
    return (eps * e_hPa) / (p_hPa - (1 - eps) * e_hPa)


def log_z_levels(nz: int | None = None) -> np.ndarray:
    nz = nz or settings.grid_nz
    return np.geomspace(5.0, 5000.0, nz).astype(np.float32)


def _interp_to_agl(values_at_levels: np.ndarray, agl_at_levels: np.ndarray,
                   z_target: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Monotonic linear interpolation of a 1D profile from (agl_at_levels, values) → z_target."""
    order = np.argsort(agl_at_levels)
    a = agl_at_levels[order]
    v = values_at_levels[order]
    return np.interp(z_target, a, v, left=v[0], right=v[-1]).astype(np.float32)


def fetch_era5_profile(lat: float, lon: float, when: datetime) -> dict[str, np.ndarray]:
    """Return a dict of (nz,) arrays: u, v, T, q, k, at the configured AGL levels.

    Uses Open-Meteo ERA5 endpoint for historical timestamps (default) and
    falls back to the forecast endpoint if /era5 fails.
    """
    nz = settings.grid_nz
    z_target = log_z_levels(nz)

    # Normalize datetime to UTC hour
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    t_utc = when.astimezone(timezone.utc)
    date_str = t_utc.strftime("%Y-%m-%d")

    hourly = _build_hourly_query()
    base_candidates = [
        ("https://archive-api.open-meteo.com/v1/era5",
         {"latitude": lat, "longitude": lon, "hourly": hourly,
          "start_date": date_str, "end_date": date_str, "timezone": "UTC"}),
        (f"{settings.openmeteo_base_url}/forecast",
         {"latitude": lat, "longitude": lon, "hourly": hourly,
          "start_date": date_str, "end_date": date_str, "timezone": "UTC"}),
    ]

    last_err: Exception | None = None
    data = None
    for url, params in base_candidates:
        try:
            r = httpx.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            last_err = e
            continue
    if data is None:
        raise RuntimeError(f"Open-Meteo fetch failed: {last_err}")

    hourly_data = data.get("hourly", {})
    times = hourly_data.get("time", [])
    if not times:
        raise RuntimeError("Open-Meteo returned empty time axis")

    # Select the hour closest to `when`
    target_hour = t_utc.strftime("%Y-%m-%dT%H:00")
    try:
        idx = times.index(target_hour)
    except ValueError:
        # Fall back to closest
        from datetime import datetime as _dt
        parsed = [_dt.fromisoformat(t) for t in times]
        idx = int(np.argmin([abs((p - t_utc.replace(tzinfo=None)).total_seconds()) for p in parsed]))

    # Assemble per-level arrays
    def _get(var: str) -> np.ndarray:
        vals = []
        for lvl in PRESSURE_LEVELS:
            key = f"{var}_{lvl}hPa"
            series = hourly_data.get(key, [])
            if idx < len(series) and series[idx] is not None:
                vals.append(float(series[idx]))
            else:
                vals.append(np.nan)
        return np.array(vals, dtype=np.float32)

    speed = _get("wind_speed")           # m/s (Open-Meteo default is km/h → see note below)
    direction = _get("wind_direction")   # degrees, meteorological (from)
    T = _get("temperature") + 273.15     # °C → K
    gh = _get("geopotential_height")     # m (geometric proxy)
    rh = _get("relative_humidity")       # %

    # NOTE: Open-Meteo returns wind_speed in km/h by default unless we ask for m/s.
    # We append wind_speed_unit=ms to the URL above implicitly? No — default is km/h.
    # Convert safely: the API also exposes wind_speed_unit via `wind_speed_unit` param,
    # but to be robust we assume km/h here and convert.
    speed = speed / 3.6  # km/h → m/s

    # Reconstruct u/v (meteorological → mathematical convention)
    # Direction is the direction the wind is coming FROM (degrees, N=0, clockwise).
    dir_rad = np.deg2rad(direction)
    u = -speed * np.sin(dir_rad)  # eastward
    v = -speed * np.cos(dir_rad)  # northward

    # Convert RH → q
    p_arr = np.array(PRESSURE_LEVELS, dtype=np.float32)
    q = _rh_to_specific_humidity(rh, T, p_arr)

    # Station elevation gives us AGL from geopotential height
    elev = float(data.get("elevation", 0.0))
    agl_at_levels = gh - elev  # rough: geometric altitude - station elevation

    # Drop NaNs (some high levels may be missing for short forecasts)
    valid = ~(np.isnan(u) | np.isnan(v) | np.isnan(T) | np.isnan(q) | np.isnan(agl_at_levels))
    if valid.sum() < 2:
        raise RuntimeError("too few valid pressure levels from Open-Meteo")
    agl_v = agl_at_levels[valid]
    u_v = u[valid]
    v_v = v[valid]
    T_v = T[valid]
    q_v = q[valid]

    u_z = _interp_to_agl(u_v, agl_v, z_target)
    v_z = _interp_to_agl(v_v, agl_v, z_target)
    T_z = _interp_to_agl(T_v, agl_v, z_target)
    q_z = _interp_to_agl(q_v, agl_v, z_target)

    # TKE: crude log-law estimate k = (u*/Cmu^0.25)^2 with u* from surface speed.
    # Use the lowest-AGL wind speed we have (typically 1000 hPa).
    speed_surface = float(np.hypot(u_z[0], v_z[0]))
    z0 = 0.1
    kappa = 0.4
    u_star = speed_surface * kappa / max(np.log(max(z_target[0], z0 + 1e-3) / z0), 1e-3)
    Cmu = 0.09
    k_surface = (u_star ** 2) / np.sqrt(Cmu)
    k_z = (k_surface * np.exp(-z_target / 500.0)).astype(np.float32)

    return {"u": u_z, "v": v_z, "T": T_z, "q": q_z, "k": k_z, "z_agl": z_target}
