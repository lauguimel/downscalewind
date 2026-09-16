"""Tests for the sub-daily FWI components (shared.fwi.hffmc, compute_hfwi_series).

No published reference table is bundled here, so the suite checks the properties
the Van Wagner (1977) moisture model must satisfy: exact composability of the
time step, convergence to the equilibrium moisture content, the no-change band
between the two equilibria, the slower hourly drying rate compared with the
daily code, and numerical safety at the edges of the input ranges.
"""

from __future__ import annotations

import numpy as np
import pytest

from shared.fwi import compute_hfwi_series, ffmc, hffmc, isi

_M_SCALE = 14875.0 / 101.0  # sub-daily scale: makes eq. 1 and eq. 6 exact inverses


def _ffmc_from_moisture(m: float) -> float:
    """Inverse of eq. 1: FFMC scale from moisture content."""
    return 59.5 * (250.0 - m) / (_M_SCALE + m)


def _equilibria(t_c: float, rh: float) -> tuple[float, float]:
    ed = (
        0.942 * rh**0.679
        + 11.0 * np.exp((rh - 100.0) / 10.0)
        + 0.18 * (21.1 - t_c) * (1.0 - np.exp(-0.115 * rh))
    )
    ew = (
        0.618 * rh**0.753
        + 10.0 * np.exp((rh - 100.0) / 10.0)
        + 0.18 * (21.1 - t_c) * (1.0 - np.exp(-0.115 * rh))
    )
    return ed, ew


@pytest.mark.parametrize("rh", [25.0, 95.0])  # drying branch, then wetting branch
def test_time_step_composes_exactly(rh: float) -> None:
    """Two 1 h steps must equal one 2 h step when the forcing is constant.

    The moisture update is exponential in k * dt, so this holds to machine
    precision; any error in how time_step_h enters the exponent breaks it.
    """
    t_c, ws, rain = 22.0, 10.0, 0.0
    one = hffmc(t_c, rh, ws, rain, 85.0, time_step_h=1.0)
    two = hffmc(t_c, rh, ws, rain, one, time_step_h=1.0)
    direct = hffmc(t_c, rh, ws, rain, 85.0, time_step_h=2.0)
    assert two == pytest.approx(direct, rel=1e-12)


def test_converges_to_drying_equilibrium() -> None:
    """Held in dry air, the fuel moisture must relax onto Ed."""
    t_c, rh, ws = 25.0, 20.0, 8.0
    ed, _ = _equilibria(t_c, rh)

    state = 85.0
    for _ in range(200):
        state = hffmc(t_c, rh, ws, 0.0, state, 1.0)

    m_final = _M_SCALE * (101.0 - state) / (59.5 + state)
    assert m_final == pytest.approx(ed, abs=1e-6)


def test_converges_to_wetting_equilibrium() -> None:
    """Held in humid air, a dry fuel must relax up onto Ew."""
    t_c, rh, ws = 12.0, 96.0, 5.0
    _, ew = _equilibria(t_c, rh)

    state = 95.0  # very dry start
    for _ in range(200):
        state = hffmc(t_c, rh, ws, 0.0, state, 1.0)

    m_final = _M_SCALE * (101.0 - state) / (59.5 + state)
    assert m_final == pytest.approx(ew, abs=1e-6)


def test_no_change_between_equilibria() -> None:
    """Between Ew and Ed the fuel neither dries nor wets: FFMC is unchanged."""
    t_c, rh, ws = 18.0, 55.0, 12.0
    ed, ew = _equilibria(t_c, rh)
    assert ew < ed  # sanity: the band exists

    f_in = _ffmc_from_moisture(0.5 * (ed + ew))
    f_out = hffmc(t_c, rh, ws, 0.0, f_in, 1.0)
    assert f_out == pytest.approx(f_in, rel=1e-12)


def test_hourly_dries_slower_than_daily() -> None:
    """One hour of drying must move FFMC far less than one daily step.

    The only structural difference is the log drying rate (0.0579 per hour vs
    0.581 per day), so a swapped constant would show up here.
    """
    t_c, rh, ws = 28.0, 20.0, 15.0
    start = 70.0
    hourly = float(hffmc(t_c, rh, ws, 0.0, start, 1.0))
    daily = float(ffmc(t_c, rh, ws, 0.0, start))
    assert start < hourly < daily


def test_rain_wets_the_fuel() -> None:
    """Hourly rain must lower FFMC, and more rain must lower it further."""
    t_c, rh, ws = 15.0, 70.0, 10.0
    dry = float(hffmc(t_c, rh, ws, 0.0, 90.0, 1.0))
    light = float(hffmc(t_c, rh, ws, 0.4, 90.0, 1.0))
    heavy = float(hffmc(t_c, rh, ws, 6.0, 90.0, 1.0))
    assert heavy < light < dry


def test_no_interception_threshold_unlike_daily() -> None:
    """0.4 mm in one hour must wet the fuel, where the daily code ignores it.

    The daily FFMC subtracts a 0.5 mm interception loss; applying that rule to an
    hourly total would silently discard drizzle.
    """
    t_c, rh, ws = 15.0, 70.0, 10.0
    assert float(hffmc(t_c, rh, ws, 0.4, 90.0, 1.0)) < float(hffmc(t_c, rh, ws, 0.0, 90.0, 1.0))
    assert float(ffmc(t_c, rh, ws, 0.4, 90.0)) == pytest.approx(float(ffmc(t_c, rh, ws, 0.0, 90.0)))


def test_finite_over_input_extremes() -> None:
    """No NaN or inf at the edges of physically plausible forcing."""
    t_c = np.array([-25.0, 0.0, 45.0, 20.0, 20.0, 20.0])
    rh = np.array([1.0, 50.0, 100.0, 0.0, 100.0, 60.0])
    ws = np.array([0.0, 5.0, 120.0, 30.0, 0.0, 10.0])
    rain = np.array([0.0, 0.0, 50.0, 0.001, 20.0, 0.0])
    prev = np.array([0.0, 30.0, 101.0, 85.0, 99.0, 60.0])

    out = hffmc(t_c, rh, ws, rain, prev, 1.0)
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0) and np.all(out <= 101.0)


def test_series_shapes_and_map_state() -> None:
    """The series runs on [n_steps, ny, nx] and keeps the state per pixel."""
    n_steps, ny, nx = 24, 3, 4
    rng = np.random.default_rng(0)
    t_c = 20.0 + 5.0 * np.sin(np.linspace(0, 2 * np.pi, n_steps))[:, None, None] * np.ones(
        (1, ny, nx)
    )
    rh = 50.0 + 20.0 * rng.random((n_steps, ny, nx))
    ws = np.full((n_steps, ny, nx), 10.0)
    ws[:, :, -1] = 30.0  # one windier column
    rain = np.zeros((n_steps, ny, nx))
    bui = np.full((n_steps, ny, nx), 40.0)

    out = compute_hfwi_series(t_c, rh, ws, rain, bui, ffmc_init=85.0)

    for key in ("ffmc", "isi", "fwi", "dsr"):
        assert out[key].shape == (n_steps, ny, nx)
        assert np.all(np.isfinite(out[key]))

    # Wind enters ISI directly: the windier column must be the most dangerous.
    assert np.all(out["isi"][:, :, -1] > out["isi"][:, :, 0])
    assert np.all(out["fwi"][:, :, -1] > out["fwi"][:, :, 0])


def test_series_matches_step_by_step_loop() -> None:
    """compute_hfwi_series must be exactly the sequential application of hffmc."""
    rng = np.random.default_rng(1)
    n = 48
    t_c = 10.0 + 15.0 * rng.random(n)
    rh = 20.0 + 70.0 * rng.random(n)
    ws = 20.0 * rng.random(n)
    rain = np.where(rng.random(n) < 0.2, 2.0 * rng.random(n), 0.0)
    bui = np.full(n, 55.0)

    out = compute_hfwi_series(t_c, rh, ws, rain, bui, ffmc_init=80.0)

    state = 80.0
    for i in range(n):
        state = float(hffmc(t_c[i], rh[i], ws[i], rain[i], state, 1.0))
        assert out["ffmc"][i] == pytest.approx(state, rel=1e-12)
        assert out["isi"][i] == pytest.approx(float(isi(state, ws[i])), rel=1e-12)


def test_diurnal_cycle_produces_a_range_the_daily_index_cannot() -> None:
    """A realistic summer day must give an intraday FWI range, not a flat value.

    This is the whole point of the sub-daily chain: the daily index reports one
    number per day, computed at noon, and cannot express the afternoon peak.
    """
    hours = np.arange(24)
    t_c = 18.0 + 12.0 * np.sin(np.pi * np.clip(hours - 6, 0, 14) / 14.0)
    rh = 85.0 - 50.0 * np.sin(np.pi * np.clip(hours - 6, 0, 14) / 14.0)
    ws = 8.0 + 12.0 * np.sin(np.pi * np.clip(hours - 8, 0, 12) / 12.0)
    rain = np.zeros(24)
    bui = np.full(24, 60.0)

    out = compute_hfwi_series(t_c, rh, ws, rain, bui, ffmc_init=85.0)

    peak_hour = int(np.argmax(out["fwi"]))
    assert 12 <= peak_hour <= 20, f"afternoon peak expected, got hour {peak_hour}"
    assert out["fwi"].max() > 1.5 * out["fwi"].min()
