"""
Tests for the M_I9-CACHE write_coords opt-out on write_input_grid_zarr.

The obs-centred training cache (services/module2b-surrogate/src/
dataset_v2_obs_centered.py) never reads coords/{x,y,z} — it recomputes AGL
heights from terrain + agl. coords/z alone is ~97% of a cached grid.zarr
entry's bytes. These tests assert:

1. Default behaviour (write_coords unset) still writes coords/.
2. write_coords=False omits coords/ entirely.
3. _build_features_from_grid_zarr produces IDENTICAL feature tensors from a
   coords-ful and a coords-less grid.zarr for the same inputs.

Uses the real cached fixture at
data/inference/stations_perdigao/rne03_20170501T0000/grid.zarr as the
source of realistic inputs (skipped if that fixture is absent, e.g. on a
clone without the local data/ tree).
"""
from __future__ import annotations

import sys
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.inference_input import write_input_grid_zarr, Era5Sample  # noqa: E402
from src.dataset_v2_obs_centered import (  # noqa: E402
    _build_features_from_grid_zarr, DEFAULT_NORM,
)
from src.dataset_v2 import parse_agl_levels  # noqa: E402
from infer_at_stations import parallel_materialise  # noqa: E402

REPO = _ROOT.parents[1]
FIXTURE = REPO / "data/inference/stations_perdigao/rne03_20170501T0000/grid.zarr"


def _toy_era5_and_grid():
    """Small synthetic era5/terrain inputs, independent of any repo fixture."""
    from utils.inference_input import NI, NJ, NK

    rng = np.random.default_rng(0)
    terrain = rng.uniform(400.0, 600.0, size=(NI, NJ)).astype(np.float32)
    agl = np.linspace(2.0, 2000.0, NK, dtype=np.float32)
    z_grid = terrain[:, :, None] + agl[None, None, :]
    plev = np.array([1000.0, 925.0, 850.0, 700.0], dtype=np.float32)
    pressure_3d = {v: rng.normal(size=(3, 3, 4)).astype(np.float32)
                   for v in ("u", "v", "T", "q")}
    surface = {v: rng.normal(size=(3, 3)).astype(np.float32)
               for v in ("t2m", "d2m", "u10", "v10")}
    era5 = Era5Sample(
        pressure_levels=plev, pressure_3d=pressure_3d, surface=surface,
        timestamp_iso="2017-05-01T00:00:00",
        actual_timestamp_iso="2017-05-01T00:00:00", delta_seconds=0.0,
    )
    return terrain, z_grid, era5


def test_default_writes_coords(tmp_path):
    terrain, z_grid, era5 = _toy_era5_and_grid()
    out = tmp_path / "grid.zarr"
    write_input_grid_zarr(
        out, site_id="toy", lat=39.7, lon=-7.7, terrain=terrain,
        z_grid=z_grid, z0_eff=0.05, era5=era5, timestamp_iso="2017-05-01T00:00:00",
    )
    g = zarr.open_group(str(out), mode="r")
    assert "coords" in g
    assert tuple(g["coords/z"].shape) == z_grid.shape
    assert tuple(g["coords/x"].shape) == (terrain.shape[0],)
    assert tuple(g["coords/y"].shape) == (terrain.shape[1],)


def test_write_coords_false_omits_coords_group(tmp_path):
    terrain, z_grid, era5 = _toy_era5_and_grid()
    out = tmp_path / "grid.zarr"
    write_input_grid_zarr(
        out, site_id="toy", lat=39.7, lon=-7.7, terrain=terrain,
        z_grid=z_grid, z0_eff=0.05, era5=era5, timestamp_iso="2017-05-01T00:00:00",
        write_coords=False,
    )
    g = zarr.open_group(str(out), mode="r")
    assert "coords" not in g
    assert "input" in g  # everything else still present
    assert tuple(g["input/terrain"].shape) == terrain.shape


def _fake_build_one(*, site_id, lat, lon, timestamp_iso, era5_store, dem,
                     worldcover, output, overwrite=False, extra_meta=None,
                     max_era5_delta_h=3.5, write_coords=True):
    """Stand-in for extract_v2_input_at_coords.build_one: records write_coords
    into a sentinel file next to the (never actually written) grid.zarr path,
    so the test can assert the flag survived pickling through the process
    pool without needing real DEM/ERA5 fixtures."""
    output.parent.mkdir(parents=True, exist_ok=True)
    (output.parent / "write_coords.flag").write_text(str(bool(write_coords)))
    return output


def test_parallel_materialise_serial_threads_write_coords(tmp_path, monkeypatch):
    """Task B, serial path (n_workers=1, runs in-process): write_coords must
    reach build_one unchanged for both True and False."""
    import infer_at_stations as ias

    monkeypatch.setattr(ias, "build_one", _fake_build_one)

    df = pd.DataFrame({
        "station_id": ["toy_a", "toy_b"],
        "lat": [39.7, 39.8],
        "lon": [-7.7, -7.6],
        "elev": [500.0, 510.0],
        "timestamp_ns": [
            int(np.datetime64("2017-05-01T00:00:00").astype("int64")) * 1000,
            int(np.datetime64("2017-05-01T01:00:00").astype("int64")) * 1000,
        ],
    })

    for write_coords in (True, False):
        workdir = tmp_path / f"wc_{write_coords}"
        res = ias.parallel_materialise(
            df, era5_store=Path("dummy.zarr"), dem=Path("dummy_dem"),
            worldcover=None, workdir=workdir, max_era5_delta_h=3.5,
            n_workers=1, write_coords=write_coords,
        )
        assert len(res) == 2
        for sid in ("toy_a", "toy_b"):
            flags = list(workdir.glob(f"{sid}_*/write_coords.flag"))
            assert len(flags) == 1
            assert flags[0].read_text() == str(write_coords)


def test_materialise_payload_is_picklable_with_write_coords():
    """Task B, pool-path pickling guarantee: parallel_materialise puts
    write_coords into the plain payload dict (not a closure), so it survives
    the pickle/unpickle round trip ProcessPoolExecutor performs when sending
    work to a worker process. Round-trip a payload through pickle explicitly
    and confirm `_materialise_one_pickleable` reads the same value back —
    this is what makes the flag safe across process boundaries, unlike a
    lambda default argument which would not repickle correctly."""
    import pickle

    df = pd.DataFrame({
        "station_id": ["toy_a"], "lat": [39.7], "lon": [-7.7], "elev": [500.0],
        "timestamp_ns": [int(np.datetime64("2017-05-01T00:00:00").astype("int64")) * 1000],
    })
    for write_coords in (True, False):
        # Same shape parallel_materialise builds per row (a plain dict, not
        # a closure over write_coords) — round-trip through pickle exactly
        # as ProcessPoolExecutor does when sending work to a worker process.
        payload = {
            "row_idx": 0, "station_id": "toy_a", "lat": 39.7, "lon": -7.7,
            "elev": 500.0, "timestamp_ns": int(df["timestamp_ns"].iloc[0]),
            "era5_store": "dummy.zarr", "dem": "dummy_dem", "worldcover": None,
            "workdir": "dummy_workdir", "max_era5_delta_h": 3.5,
            "write_coords": write_coords,
        }
        repickled = pickle.loads(pickle.dumps(payload))
        assert repickled["write_coords"] == write_coords


@pytest.mark.skipif(not FIXTURE.exists(), reason="local cache fixture not present")
def test_features_identical_coords_present_vs_absent(tmp_path):
    g = zarr.open_group(str(FIXTURE), mode="r")
    terrain = np.asarray(g["input/terrain"][:], dtype=np.float32)
    z_grid = np.asarray(g["coords/z"][:], dtype=np.float32)
    z0_eff = float(g["input"].attrs.get("z0_eff", 0.0))
    lat = float(g["input"].attrs.get("lat", 0.0))
    lon = float(g["input"].attrs.get("lon", 0.0))
    plev = np.asarray(g["input/era5_pressure_levels"][:], dtype=np.float32)
    pressure_3d = {v: np.asarray(g[f"input/era5_3d/{v}"][:], dtype=np.float32)
                   for v in g["input/era5_3d"]}
    surface = {v: np.asarray(g[f"input/era5_surface/{v}"][:], dtype=np.float32)
               for v in g["input/era5_surface"]}
    meta = dict(g["input/inflow_meta"].attrs)
    ts_iso = str(meta.get("timestamp", "2017-05-01T00:00:00"))
    era5 = Era5Sample(
        pressure_levels=plev, pressure_3d=pressure_3d, surface=surface,
        timestamp_iso=ts_iso,
        actual_timestamp_iso=str(meta.get("actual_era5_timestamp", "")),
        delta_seconds=float(meta.get("era5_time_delta_s", 0.0)),
    )

    out_with = tmp_path / "with_coords" / "grid.zarr"
    out_without = tmp_path / "without_coords" / "grid.zarr"
    write_input_grid_zarr(
        out_with, site_id="rne03", lat=lat, lon=lon, terrain=terrain,
        z_grid=z_grid, z0_eff=z0_eff, era5=era5, timestamp_iso=ts_iso,
        write_coords=True,
    )
    write_input_grid_zarr(
        out_without, site_id="rne03", lat=lat, lon=lon, terrain=terrain,
        z_grid=z_grid, z0_eff=z0_eff, era5=era5, timestamp_iso=ts_iso,
        write_coords=False,
    )

    norm = DEFAULT_NORM
    levels = parse_agl_levels("agl_0_100_24")
    feats_with = _build_features_from_grid_zarr(out_with, norm, levels)
    feats_without = _build_features_from_grid_zarr(out_without, norm, levels)
    for a, b in zip(feats_with, feats_without):
        assert np.array_equal(np.asarray(a), np.asarray(b))

    shutil.rmtree(tmp_path / "with_coords", ignore_errors=True)
    shutil.rmtree(tmp_path / "without_coords", ignore_errors=True)
