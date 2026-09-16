"""
shared.fwi — Canadian Forest Fire Weather Index System (Van Wagner 1987).

Computes the 6 FWI components from noon weather observations, plus the
sub-daily (hourly) variant of the fast components after Van Wagner (1977).
Designed for use with downscaled atmospheric fields (T, q, wind) where
precipitation comes from the driving model (not downscaled).

References:
    Van Wagner, C.E. (1987). Development and Structure of the Canadian
    Forest Fire Weather Index System. Forestry Technical Report 35.
    Canadian Forestry Service, Ottawa.
    Van Wagner, C.E. (1977). A method of computing fine fuel moisture
    content throughout the diurnal cycle. Information Report PS-X-69.
"""

from __future__ import annotations

import numpy as np


# ── Unit conversions ─────────────────────────────────────────────────────────


def specific_humidity_to_rh(
    q: np.ndarray, t_kelvin: np.ndarray, p_hpa: np.ndarray
) -> np.ndarray:
    """Convert specific humidity (kg/kg) to relative humidity (%).

    Uses the Tetens formula for saturation vapour pressure:
        e_sat = 6.1078 * 10^(7.5 * T_c / (237.3 + T_c))  [hPa]
    """
    t_c = t_kelvin - 273.15
    e_sat = 6.1078 * np.power(10.0, 7.5 * t_c / (237.3 + t_c))
    # Actual vapour pressure from specific humidity: e = q * p / (0.622 + 0.378 * q)
    e = q * p_hpa / (0.622 + 0.378 * q)
    rh = 100.0 * e / e_sat
    return np.clip(rh, 0.0, 100.0)


# ── FFMC (Fine Fuel Moisture Code) ──────────────────────────────────────────


def ffmc(
    t_c: np.ndarray,
    rh: np.ndarray,
    ws_kmh: np.ndarray,
    rain_mm: np.ndarray,
    ffmc_prev: np.ndarray | float = 85.0,
) -> np.ndarray:
    """Fine Fuel Moisture Code (equations 1-13, Van Wagner 1987)."""
    t_c = np.asarray(t_c, dtype=np.float64)
    rh = np.asarray(rh, dtype=np.float64)
    ws_kmh = np.asarray(ws_kmh, dtype=np.float64)
    rain_mm = np.asarray(rain_mm, dtype=np.float64)
    ffmc_prev = np.asarray(ffmc_prev, dtype=np.float64)

    # Previous moisture content (eq 1)
    mo = 147.2 * (101.0 - ffmc_prev) / (59.5 + ffmc_prev)

    # Rain phase (eq 2-4)
    rf = np.where(rain_mm > 0.5, rain_mm - 0.5, 0.0)
    rf_safe = np.maximum(rf, 1e-10)  # avoid div-by-zero in np.where branch eval
    mo_safe = np.minimum(mo, 250.9)  # avoid exp overflow when mo ~ 251
    base_rain = 42.5 * rf_safe * np.exp(-100.0 / (251.0 - mo_safe)) * (1.0 - np.exp(-6.93 / rf_safe))
    mr = np.where(
        mo <= 150.0,
        mo + base_rain,
        mo + base_rain + 0.0015 * np.square(mo - 150.0) * np.sqrt(rf_safe),
    )
    mr = np.minimum(mr, 250.0)
    mo = np.where(rain_mm > 0.5, mr, mo)

    # Equilibrium moisture content (eq 5-6)
    ed = (
        0.942 * np.power(rh, 0.679)
        + 11.0 * np.exp((rh - 100.0) / 10.0)
        + 0.18 * (21.1 - t_c) * (1.0 - np.exp(-0.115 * rh))
    )
    ew = (
        0.618 * np.power(rh, 0.753)
        + 10.0 * np.exp((rh - 100.0) / 10.0)
        + 0.18 * (21.1 - t_c) * (1.0 - np.exp(-0.115 * rh))
    )

    # Log drying/wetting rate (eq 7-10)
    k0_d = 0.424 * (1.0 - np.power(rh / 100.0, 1.7)) + 0.0694 * np.sqrt(ws_kmh) * (
        1.0 - np.power(rh / 100.0, 8.0)
    )
    kd = k0_d * 0.581 * np.exp(0.0365 * t_c)

    k0_w = 0.424 * (1.0 - np.power((100.0 - rh) / 100.0, 1.7)) + 0.0694 * np.sqrt(
        ws_kmh
    ) * (1.0 - np.power((100.0 - rh) / 100.0, 8.0))
    kw = k0_w * 0.581 * np.exp(0.0365 * t_c)

    # Final moisture (eq 11-12)
    m = np.where(mo > ed, ed + (mo - ed) * np.power(10.0, -kd), mo)
    m = np.where(mo < ew, ew - (ew - mo) * np.power(10.0, -kw), m)

    # Back to FFMC scale (eq 13)
    return 59.5 * (250.0 - m) / (147.2 + m)


# ── DMC (Duff Moisture Code) ────────────────────────────────────────────────

# Effective day-length factors by month (1-indexed via month-1)
_LE = np.array([6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0])


def dmc(
    t_c: np.ndarray,
    rh: np.ndarray,
    rain_mm: np.ndarray,
    dmc_prev: np.ndarray | float,
    month: np.ndarray | int,
) -> np.ndarray:
    """Duff Moisture Code (equations 14-21, Van Wagner 1987)."""
    t_c = np.asarray(t_c, dtype=np.float64)
    rh = np.asarray(rh, dtype=np.float64)
    rain_mm = np.asarray(rain_mm, dtype=np.float64)
    dmc_prev = np.asarray(dmc_prev, dtype=np.float64)
    month = np.asarray(month, dtype=np.int32)

    # Effective temperature (clamp at -1.1)
    t_eff = np.maximum(t_c, -1.1)

    # Rain phase (eq 14-17)
    re = 0.92 * rain_mm - 1.27
    mo = 20.0 + np.exp(5.6348 - dmc_prev / 43.43)  # eq 15
    b = np.where(
        dmc_prev <= 33.0,
        100.0 / (0.5 + 0.3 * dmc_prev),
        np.where(
            dmc_prev <= 65.0,
            14.0 - 1.3 * np.log(dmc_prev),
            6.2 * np.log(dmc_prev) - 17.2,
        ),
    )
    mr = mo + 1000.0 * re / (48.77 + b * re)  # eq 16
    pr = 244.72 - 43.43 * np.log(mr - 20.0)  # eq 17
    pr = np.maximum(pr, 0.0)
    dmc_after_rain = np.where(rain_mm > 1.5, pr, dmc_prev)

    # Drying phase (eq 18-21)
    le = _LE[month - 1]
    log_k = 1.894 * (t_eff + 1.1) * (100.0 - rh) * le * 1e-6  # eq 20
    return np.maximum(dmc_after_rain + log_k, 0.0)


# ── DC (Drought Code) ───────────────────────────────────────────────────────

# Day-length adjustment factors by month
_LF = np.array([-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6])


def dc(
    t_c: np.ndarray,
    rain_mm: np.ndarray,
    dc_prev: np.ndarray | float,
    month: np.ndarray | int,
) -> np.ndarray:
    """Drought Code (equations 22-28, Van Wagner 1987)."""
    t_c = np.asarray(t_c, dtype=np.float64)
    rain_mm = np.asarray(rain_mm, dtype=np.float64)
    dc_prev = np.asarray(dc_prev, dtype=np.float64)
    month = np.asarray(month, dtype=np.int32)

    t_eff = np.maximum(t_c, -2.8)

    # Rain phase (eq 22-24)
    rd = 0.83 * rain_mm - 1.27  # eq 22
    qo = 800.0 * np.exp(-dc_prev / 400.0)  # eq 23
    qr = qo + 3.937 * rd  # eq 24
    dr = 400.0 * np.log(800.0 / qr)  # eq 25
    dr = np.maximum(dr, 0.0)
    dc_after_rain = np.where(rain_mm > 2.8, dr, dc_prev)

    # Drying phase (eq 26-28)
    lf = _LF[month - 1]
    v = 0.36 * (t_eff + 2.8) + lf  # eq 27
    v = np.maximum(v, 0.0)
    return np.maximum(dc_after_rain + 0.5 * v, 0.0)


# ── ISI (Initial Spread Index) ──────────────────────────────────────────────


def isi(ffmc_val: np.ndarray, ws_kmh: np.ndarray) -> np.ndarray:
    """Initial Spread Index (equations 29-30, Van Wagner 1987)."""
    ffmc_val = np.asarray(ffmc_val, dtype=np.float64)
    ws_kmh = np.asarray(ws_kmh, dtype=np.float64)

    # Moisture content from FFMC (eq 1 inverted)
    m = 147.2 * (101.0 - ffmc_val) / (59.5 + ffmc_val)

    # Wind function (eq 29)
    fw = np.exp(0.05039 * ws_kmh)

    # Moisture function (eq 30)
    ff = 91.9 * np.exp(-0.1386 * m) * (1.0 + np.power(m, 5.31) / 4.93e7)

    return 0.208 * fw * ff


# ── BUI (Buildup Index) ─────────────────────────────────────────────────────


def bui(dmc_val: np.ndarray, dc_val: np.ndarray) -> np.ndarray:
    """Buildup Index (equations 31-32, Van Wagner 1987)."""
    dmc_val = np.asarray(dmc_val, dtype=np.float64)
    dc_val = np.asarray(dc_val, dtype=np.float64)

    # Primary formula (eq 31)
    u = np.where(
        dmc_val <= 0.4 * dc_val,
        0.8 * dmc_val * dc_val / (dmc_val + 0.4 * dc_val),
        dmc_val
        - (1.0 - 0.8 * dc_val / (dmc_val + 0.4 * dc_val))
        * (0.92 + np.power(0.0114 * dmc_val, 1.7)),
    )
    return np.maximum(u, 0.0)


# ── FWI (Fire Weather Index) ────────────────────────────────────────────────


def fwi(isi_val: np.ndarray, bui_val: np.ndarray) -> np.ndarray:
    """Fire Weather Index (equations 33-34, Van Wagner 1987)."""
    isi_val = np.asarray(isi_val, dtype=np.float64)
    bui_val = np.asarray(bui_val, dtype=np.float64)

    # BUI effect on fire intensity (eq 33)
    fd = np.where(
        bui_val <= 80.0,
        0.626 * np.power(bui_val, 0.809) + 2.0,
        1000.0 / (25.0 + 108.64 * np.exp(-0.023 * bui_val)),
    )

    # Intermediate fire intensity (eq 33)
    b = 0.1 * isi_val * fd

    # Final FWI (eq 34). np.where evaluates both branches, so the log branch is
    # clamped: for b <= 1 it would take a negative base to a fractional power and
    # raise an invalid-value warning on every call (noisy once the index is
    # computed hourly on a full map). The returned values are unchanged.
    b_safe = np.maximum(b, 1.0 + 1e-12)
    s = np.where(
        b > 1.0,
        np.exp(2.72 * np.power(0.434 * np.log(b_safe), 0.647)),
        b,
    )
    return s


# ── Daily time series ────────────────────────────────────────────────────────


def compute_fwi_series(
    t_c: np.ndarray,
    rh: np.ndarray,
    ws_kmh: np.ndarray,
    rain_mm: np.ndarray,
    months: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute full FWI system for a daily time series.

    All inputs are 1D arrays (one value per day at noon).
    months: 1D array of month numbers (1-12).
    Returns dict with keys: ffmc, dmc, dc, isi, bui, fwi (all 1D float64).
    """
    n = len(t_c)
    out = {k: np.zeros(n, dtype=np.float64) for k in ("ffmc", "dmc", "dc", "isi", "bui", "fwi")}

    ffmc_prev = 85.0
    dmc_prev = 6.0
    dc_prev = 15.0

    for i in range(n):
        ffmc_prev = float(ffmc(t_c[i], rh[i], ws_kmh[i], rain_mm[i], ffmc_prev))
        dmc_prev = float(dmc(t_c[i], rh[i], rain_mm[i], dmc_prev, months[i]))
        dc_prev = float(dc(t_c[i], rain_mm[i], dc_prev, months[i]))

        isi_val = float(isi(ffmc_prev, ws_kmh[i]))
        bui_val = float(bui(dmc_prev, dc_prev))
        fwi_val = float(fwi(isi_val, bui_val))

        out["ffmc"][i] = ffmc_prev
        out["dmc"][i] = dmc_prev
        out["dc"][i] = dc_prev
        out["isi"][i] = isi_val
        out["bui"][i] = bui_val
        out["fwi"][i] = fwi_val

    return out


# ── Spatial field (single timestep) ─────────────────────────────────────────


def compute_fwi_field(
    t_kelvin: np.ndarray,
    q_kgkg: np.ndarray,
    p_hpa: np.ndarray,
    u_ms: np.ndarray,
    v_ms: np.ndarray,
    rain_mm: np.ndarray,
    month: int,
    *,
    ffmc_prev: np.ndarray | float = 85.0,
    dmc_prev: np.ndarray | float = 6.0,
    dc_prev: np.ndarray | float = 15.0,
) -> dict[str, np.ndarray]:
    """Compute instantaneous FWI components on a spatial grid.

    Inputs can be any shape (2D lat/lon, 3D, etc.) but must broadcast.
    Converts units internally: T K→°C, q→RH, wind m/s→km/h.

    Returns dict with keys: rh, ws_kmh, ffmc, dmc, dc, isi, bui, fwi.
    """
    t_c = t_kelvin - 273.15
    rh_pct = specific_humidity_to_rh(q_kgkg, t_kelvin, p_hpa)
    ws_kmh = np.sqrt(u_ms**2 + v_ms**2) * 3.6

    ffmc_val = ffmc(t_c, rh_pct, ws_kmh, rain_mm, ffmc_prev)
    dmc_val = dmc(t_c, rh_pct, rain_mm, dmc_prev, month)
    dc_val = dc(t_c, rain_mm, dc_prev, month)
    isi_val = isi(ffmc_val, ws_kmh)
    bui_val = bui(dmc_val, dc_val)
    fwi_val = fwi(isi_val, bui_val)

    return {
        "rh": rh_pct,
        "ws_kmh": ws_kmh,
        "ffmc": ffmc_val,
        "dmc": dmc_val,
        "dc": dc_val,
        "isi": isi_val,
        "bui": bui_val,
        "fwi": fwi_val,
    }


# ── Hybrid FWI: spatial ISI × temporal BUI ─────────────────────────────────


def compute_fwi_hybrid(
    t_kelvin: np.ndarray,
    q_kgkg: np.ndarray,
    p_hpa: np.ndarray,
    u_ms: np.ndarray,
    v_ms: np.ndarray,
    month: int,
    *,
    bui_era5: float,
    ffmc_prev: np.ndarray | float = 85.0,
) -> dict[str, np.ndarray]:
    """Hybrid FWI: spatial ISI from CFD wind + temporal BUI from ERA5 series.

    ISI depends on wind (spatially heterogeneous in terrain) and FFMC
    (fast-response, ~1 day memory). BUI depends on DMC/DC which integrate
    weeks-months of rainfall — uniform at CFD domain scale, so ERA5 suffices.

    This decoupling avoids the single-step BUI problem (wrong initial values)
    while capturing terrain-induced wind acceleration in ISI.

    Args:
        t_kelvin, q_kgkg, p_hpa: atmospheric state on spatial grid
        u_ms, v_ms: wind components from CFD (any shape)
        month: month number (1-12)
        bui_era5: BUI value from ERA5 daily time series (scalar, pre-computed)
        ffmc_prev: previous FFMC (scalar or spatial, default 85.0)

    Returns dict with keys: rh, ws_kmh, ffmc, isi, bui, fwi.
    """
    t_c = t_kelvin - 273.15
    rh_pct = specific_humidity_to_rh(q_kgkg, t_kelvin, p_hpa)
    ws_kmh = np.sqrt(u_ms**2 + v_ms**2) * 3.6

    # FFMC: fast response, compute spatially from CFD fields (no rain = dry day)
    ffmc_val = ffmc(t_c, rh_pct, ws_kmh, np.zeros_like(t_c), ffmc_prev)

    # ISI: spatial, from FFMC + CFD wind
    isi_val = isi(ffmc_val, ws_kmh)

    # BUI: from ERA5 time series (temporal, uniform over domain)
    bui_val = np.full_like(isi_val, bui_era5)

    # FWI: combine spatial ISI with temporal BUI
    fwi_val = fwi(isi_val, bui_val)

    return {
        "rh": rh_pct,
        "ws_kmh": ws_kmh,
        "ffmc": ffmc_val,
        "isi": isi_val,
        "bui": bui_val,
        "fwi": fwi_val,
    }


# ── Hourly FFMC / ISI / FWI (Van Wagner 1977) ───────────────────────────────
#
# The daily FFMC above assumes one observation per day at local noon. Van Wagner
# (1977) derived the same moisture model with a sub-daily drying constant, which
# lets the fine-fuel moisture follow the diurnal cycle. Only the fast components
# change: FFMC -> ISI -> FWI are recomputed every step, while BUI (DMC + DC,
# weeks-to-months memory) stays a daily quantity supplied by the caller.
#
# Constants below follow the reference implementation `hffmc` of the R package
# cffdrs (Natural Resources Canada), checked against its source. The single
# difference with the daily code is the log drying/wetting rate: 0.0579 per hour
# instead of 0.581 per day. The rain routine also differs: no 0.5 mm
# interception loss, since an hourly total is not a daily total.
#
# Reference:
#     Van Wagner, C.E. (1977). A method of computing fine fuel moisture content
#     throughout the diurnal cycle. Information Report PS-X-69, Petawawa Forest
#     Experiment Station, Canadian Forestry Service.

# FFMC <-> moisture scale. The daily code above uses Van Wagner's rounded 147.2,
# which makes eq. 1 and eq. 6 slightly inconsistent: 147.2 * 101 = 14867.2 while
# 59.5 * 250 = 14875, so a state that neither dries nor wets still gains ~0.05
# FFMC per conversion. Harmless once a day, a spurious ~1 FFMC/day drift over 24
# hourly steps. 14875/101 = 147.27723 (the constant used by cffdrs) makes the two
# equations exact inverses; the sub-daily code uses it for that reason.
_M_SCALE_SUBDAILY = 14875.0 / 101.0


def hffmc(
    t_c: np.ndarray,
    rh: np.ndarray,
    ws_kmh: np.ndarray,
    rain_mm: np.ndarray,
    ffmc_prev: np.ndarray | float = 85.0,
    time_step_h: float = 1.0,
) -> np.ndarray:
    """Hourly Fine Fuel Moisture Code (Van Wagner 1977).

    One sub-daily step. Every argument broadcasts, so the same call serves a
    single station, a vector of stations, or a 2D map of the downscaled domain
    (in which case `ffmc_prev` carries the per-pixel state).

    Args:
        t_c: air temperature (degC) at the step.
        rh: relative humidity (%) at the step.
        ws_kmh: wind speed (km/h) at the step.
        rain_mm: rainfall accumulated over the step (mm).
        ffmc_prev: FFMC at the end of the previous step.
        time_step_h: duration of the step in hours (1.0 = hourly).

    Returns:
        FFMC at the end of the step, same broadcast shape as the inputs.
    """
    t_c = np.asarray(t_c, dtype=np.float64)
    rh = np.asarray(rh, dtype=np.float64)
    ws_kmh = np.asarray(ws_kmh, dtype=np.float64)
    rain_mm = np.asarray(rain_mm, dtype=np.float64)
    ffmc_prev = np.asarray(ffmc_prev, dtype=np.float64)

    # Moisture content of the previous step (eq 1)
    mo = _M_SCALE_SUBDAILY * (101.0 - ffmc_prev) / (59.5 + ffmc_prev)

    # Rain phase: the whole hourly total wets the fuel, no interception loss
    rf_safe = np.maximum(rain_mm, 1e-10)  # keep the unused branch finite
    mo_safe = np.minimum(mo, 250.9)  # avoid exp overflow when mo -> 251
    base_rain = (
        42.5 * rf_safe * np.exp(-100.0 / (251.0 - mo_safe)) * (1.0 - np.exp(-6.93 / rf_safe))
    )
    mr = np.where(
        mo <= 150.0,
        mo + base_rain,
        mo + base_rain + 0.0015 * np.square(mo - 150.0) * np.sqrt(rf_safe),
    )
    mr = np.minimum(mr, 250.0)
    mo = np.where(rain_mm > 0.0, mr, mo)

    # Equilibrium moisture contents for drying and wetting
    ed = (
        0.942 * np.power(rh, 0.679)
        + 11.0 * np.exp((rh - 100.0) / 10.0)
        + 0.18 * (21.1 - t_c) * (1.0 - np.exp(-0.115 * rh))
    )
    ew = (
        0.618 * np.power(rh, 0.753)
        + 10.0 * np.exp((rh - 100.0) / 10.0)
        + 0.18 * (21.1 - t_c) * (1.0 - np.exp(-0.115 * rh))
    )

    # Log drying / wetting rates, hourly constant 0.0579
    k0_d = 0.424 * (1.0 - np.power(rh / 100.0, 1.7)) + 0.0694 * np.sqrt(ws_kmh) * (
        1.0 - np.power(rh / 100.0, 8.0)
    )
    kd = k0_d * 0.0579 * np.exp(0.0365 * t_c)

    k0_w = 0.424 * (1.0 - np.power((100.0 - rh) / 100.0, 1.7)) + 0.0694 * np.sqrt(
        ws_kmh
    ) * (1.0 - np.power((100.0 - rh) / 100.0, 8.0))
    kw = k0_w * 0.0579 * np.exp(0.0365 * t_c)

    md = ed + (mo - ed) * np.power(10.0, -kd * time_step_h)
    mw = ew - (ew - mo) * np.power(10.0, -kw * time_step_h)

    # Between the two equilibria the fuel neither dries nor wets
    m = np.where(mo > ed, md, mw)
    m = np.where((mo <= ed) & (mo >= ew), mo, m)
    m = np.maximum(m, 0.0)

    return np.maximum(59.5 * (250.0 - m) / (_M_SCALE_SUBDAILY + m), 0.0)


def compute_hfwi_series(
    t_c: np.ndarray,
    rh: np.ndarray,
    ws_kmh: np.ndarray,
    rain_mm: np.ndarray,
    bui_val: np.ndarray,
    *,
    ffmc_init: np.ndarray | float = 85.0,
    time_step_h: float = 1.0,
) -> dict[str, np.ndarray]:
    """Sub-daily FWI series: hourly FFMC and ISI, daily BUI held from the driver.

    The leading axis is time; any trailing axes are carried through, so the same
    function runs a station time series (shape [n_steps]) or a map time series
    (shape [n_steps, ny, nx]) with the moisture state kept per pixel.

    `bui_val` is the Buildup Index of the day each step belongs to — it comes
    from the daily DMC/DC chain of the driving model, which is not downscaled:
    DMC and DC integrate weeks of rainfall and are smooth at domain scale.

    Args:
        t_c, rh, ws_kmh, rain_mm: forcing, shape [n_steps, ...].
        bui_val: daily BUI broadcast onto the step axis, shape [n_steps, ...].
        ffmc_init: FFMC before the first step.
        time_step_h: duration of one step in hours.

    Returns:
        dict with keys ffmc, isi, fwi, dsr; each of shape [n_steps, ...].
    """
    t_c = np.asarray(t_c, dtype=np.float64)
    rh = np.asarray(rh, dtype=np.float64)
    ws_kmh = np.asarray(ws_kmh, dtype=np.float64)
    rain_mm = np.asarray(rain_mm, dtype=np.float64)
    bui_arr = np.broadcast_to(np.asarray(bui_val, dtype=np.float64), t_c.shape)

    ffmc_out = np.empty_like(t_c)
    state = np.broadcast_to(np.asarray(ffmc_init, dtype=np.float64), t_c.shape[1:]).copy()

    for i in range(t_c.shape[0]):
        state = hffmc(t_c[i], rh[i], ws_kmh[i], rain_mm[i], state, time_step_h)
        ffmc_out[i] = state

    isi_out = isi(ffmc_out, ws_kmh)
    fwi_out = fwi(isi_out, bui_arr)
    dsr_out = 0.0272 * np.power(fwi_out, 1.77)  # Daily Severity Rating

    return {"ffmc": ffmc_out, "isi": isi_out, "fwi": fwi_out, "dsr": dsr_out}
