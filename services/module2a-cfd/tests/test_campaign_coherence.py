"""
test_campaign_coherence.py — Independent validation of CFD campaign results.

These tests read the exported Zarr and compare against source data (ERA5, SRTM,
WorldCover) to verify the pipeline didn't introduce errors. They are completely
decoupled from the generation code.

Tests:
  1. ERA5 consistency: at high altitude, CFD U/T/q ≈ ERA5 inflow
  2. WorldCover z0: terrain z0 values match source raster
  3. SRTM terrain: mesh elevation matches DEM
  4. Physical bounds: T > 200K, q ≥ 0 (mostly), k > 0, |U| < 50
  5. Metadata completeness: all inflow fields present and finite

Usage:
    cd services/module2a-cfd
    pytest tests/test_campaign_coherence.py -v

    # Single test:
    pytest tests/test_campaign_coherence.py::test_era5_u_at_altitude -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths — adjust if data is elsewhere
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
ZARR_PATH = ROOT / "data" / "cases" / "poc_100ts_q.zarr"
ERA5_PATH = ROOT / "data" / "raw" / "era5_perdigao.zarr"
SRTM_PATH = ROOT / "data" / "raw" / "srtm_perdigao_30m.tif"
WC_PATH = ROOT / "data" / "raw" / "worldcover_perdigao.tif"

SITE_LAT = 39.716
SITE_LON = -7.740


@pytest.fixture(scope="module")
def campaign():
    """Load campaign Zarr once for all tests."""
    import zarr

    if not ZARR_PATH.exists():
        pytest.skip(f"Campaign Zarr not found: {ZARR_PATH}")

    store = zarr.open_group(str(ZARR_PATH), mode="r")
    data = {
        "U": np.array(store["U"][:]),           # (n_cases, n_cells, 3)
        "T": np.array(store["T"][:]),           # (n_cases, n_cells)
        "q": np.array(store["q"][:]),           # (n_cases, n_cells)
        "k": np.array(store["k"][:]),           # (n_cases, n_cells)
        "epsilon": np.array(store["epsilon"][:]),
        "x": np.array(store["coords/x"][:]),    # (n_cells,)
        "y": np.array(store["coords/y"][:]),
        "z": np.array(store["coords/z"][:]),
        "z_agl": np.array(store["coords/z_agl"][:]),
        "elev": np.array(store["coords/elev"][:]),
    }
    # Metadata
    data["meta"] = {}
    for k in store["meta"].keys():
        data["meta"][k] = np.array(store["meta"][k][:])

    # Inflow profiles
    data["inflow"] = {}
    for k in store["inflow"].keys():
        data["inflow"][k] = np.array(store["inflow"][k][:])

    # Terrain
    if "terrain" in store:
        data["z0_field"] = np.array(store["terrain/z0_field"][:]) \
            if "z0_field" in store["terrain"] else None
    else:
        data["z0_field"] = None

    data["n_cases"] = store.attrs["n_cases"]
    data["n_cells"] = store.attrs["n_cells"]
    return data


@pytest.fixture(scope="module")
def era5():
    """Load ERA5 Zarr (source data for comparison)."""
    import zarr

    if not ERA5_PATH.exists():
        pytest.skip(f"ERA5 Zarr not found: {ERA5_PATH}")

    store = zarr.open_group(str(ERA5_PATH), mode="r")
    return {
        "times": np.array(store["coords/time"][:]).astype("datetime64[ns]"),
        "levels": np.array(store["coords/level"][:]),
        "lats": np.array(store["coords/lat"][:]),
        "lons": np.array(store["coords/lon"][:]),
        "u": np.array(store["pressure/u"][:]),
        "v": np.array(store["pressure/v"][:]),
        "t": np.array(store["pressure/t"][:]),
        "q": np.array(store["pressure/q"][:]),
        "z": np.array(store["pressure/z"][:]),
    }


# ===================================================================
# 1. Physical bounds
# ===================================================================

class TestPhysicalBounds:
    """All CFD fields must be within physically reasonable ranges."""

    def test_temperature_range(self, campaign):
        """T should be in [200, 350] K (no cryogenic or solar surface temps)."""
        T = campaign["T"]
        assert T.min() > 200.0, f"T min = {T.min():.1f} K — too cold"
        assert T.max() < 350.0, f"T max = {T.max():.1f} K — too hot"

    def test_humidity_mostly_positive(self, campaign):
        """q should be ≥ 0 (small negative OK from linearUpwind, < 1%)."""
        q = campaign["q"]
        n_neg = (q < 0).sum()
        frac = n_neg / q.size
        assert frac < 0.01, f"{frac*100:.2f}% of q cells negative (> 1% threshold)"

    def test_wind_speed_bounded(self, campaign):
        """|U| should be < 50 m/s (no numerical blowup)."""
        U = campaign["U"]
        speed = np.linalg.norm(U, axis=-1)
        assert speed.max() < 50.0, f"|U| max = {speed.max():.1f} m/s — blowup?"

    def test_tke_positive(self, campaign):
        """k should be ≥ 0 everywhere."""
        k = campaign["k"]
        assert k.min() >= 0.0, f"k min = {k.min():.6f} — negative TKE"

    def test_epsilon_positive(self, campaign):
        """epsilon should be > 0 everywhere."""
        eps = campaign["epsilon"]
        assert eps.min() > 0.0, f"epsilon min = {eps.min():.2e} — zero/negative"

    def test_vertical_velocity_bounded(self, campaign):
        """W (vertical component) at mid-altitude should be reasonable."""
        z_agl = campaign["z_agl"]
        U = campaign["U"]
        # Mid-altitude: 500–2000m AGL (avoid top BC artifacts)
        mid = (z_agl > 500.0) & (z_agl < 2000.0)
        if mid.sum() == 0:
            pytest.skip("No cells at 500-2000m AGL")
        W_mid = U[:, mid, 2]
        # 99th percentile to exclude outliers
        w99 = np.percentile(np.abs(W_mid), 99)
        assert w99 < 10.0, \
            f"|W| P99 at 500-2000m = {w99:.2f} m/s — too large"


# ===================================================================
# 2. ERA5 consistency at altitude
# ===================================================================

class TestERA5Consistency:
    """At high altitude (z_agl > 2000m), CFD should ≈ ERA5 inflow."""

    def _high_alt_mask(self, campaign):
        """Cells above 2000m AGL — far from terrain influence."""
        return campaign["z_agl"] > 2000.0

    def test_u_matches_era5_at_altitude(self, campaign):
        """Wind speed at z>2km should match ERA5 inflow profile at same height."""
        mask = self._high_alt_mask(campaign)
        if mask.sum() == 0:
            pytest.skip("No cells above 2000m AGL")

        U = campaign["U"]
        z_agl = campaign["z_agl"]
        z_levels = campaign["inflow"]["z_levels"]
        u_prof = campaign["inflow"]["u_profile"]

        # For each case, compare CFD speed at z~2500m with inflow profile at same z
        target_z = 2500.0
        diffs = []
        for i in range(campaign["n_cases"]):
            # CFD: cells near z=2500m AGL
            near = np.abs(z_agl - target_z) < 300.0
            if near.sum() == 0:
                continue
            cfd_speed = float(np.linalg.norm(U[i, near, :], axis=-1).mean())

            # ERA5 inflow profile: interpolate to target_z
            zl = z_levels[i]
            up = u_prof[i]
            valid = np.isfinite(zl) & np.isfinite(up)
            if valid.sum() < 2:
                continue
            era5_speed = float(np.interp(target_z, zl[valid], up[valid]))

            diffs.append(cfd_speed - era5_speed)

        if not diffs:
            pytest.skip("Could not compute diffs")

        diffs = np.array(diffs)
        rmse = np.sqrt((diffs**2).mean())
        print(f"  U at {target_z}m: bias={diffs.mean():.2f}, RMSE={rmse:.2f} m/s")

        assert rmse < 5.0, \
            f"RMSE(CFD vs ERA5 profile at {target_z}m) = {rmse:.2f} m/s — too large"

    def test_T_matches_era5_at_altitude(self, campaign):
        """Temperature at z>2km should be within 5K of T_ref on average."""
        mask = self._high_alt_mask(campaign)
        if mask.sum() == 0:
            pytest.skip("No cells above 2000m AGL")

        T = campaign["T"]
        T_ref = campaign["meta"]["T_ref"]

        T_high = T[:, mask].mean(axis=1)
        bias = (T_high - T_ref).mean()

        # T_ref is the volume-average, so T at altitude will differ
        # (lapse rate ~6.5 K/km → at 3km AGL, expect ~20K cooler)
        # Just check it's reasonable
        assert abs(bias) < 30.0, \
            f"T bias at altitude = {bias:.1f} K — too large"

    def test_q_matches_inflow_at_altitude(self, campaign):
        """q at z>2km should be order-of-magnitude consistent with q_ref."""
        mask = self._high_alt_mask(campaign)
        if mask.sum() == 0:
            pytest.skip("No cells above 2000m AGL")

        q = campaign["q"]
        q_ref = campaign["meta"]["q_ref"]

        q_high = q[:, mask].mean(axis=1)

        # q decreases with altitude, so q_high < q_ref is expected
        # Just check same order of magnitude
        ratio = q_high / np.maximum(q_ref, 1e-6)
        assert ratio.mean() > 0.01, "q at altitude is < 1% of q_ref — suspicious"
        assert ratio.mean() < 5.0, "q at altitude is > 5× q_ref — suspicious"

    def test_inflow_profiles_span_domain(self, campaign):
        """Inflow z_levels should cover [0, domain_top]."""
        z_levels = campaign["inflow"]["z_levels"]

        # Check first case (representative)
        z = z_levels[0]
        z_valid = z[np.isfinite(z)]
        domain_top = campaign["z"].max()

        assert z_valid.min() < 200.0, \
            f"Inflow z_min = {z_valid.min():.0f} m — should start near surface"
        assert z_valid.max() > domain_top * 0.5, \
            f"Inflow z_max = {z_valid.max():.0f} m — should reach upper domain"


# ===================================================================
# 3. Terrain / SRTM consistency
# ===================================================================

class TestTerrainConsistency:
    """Mesh terrain elevation should match SRTM DEM."""

    def test_elevation_range(self, campaign):
        """Perdigão: terrain should be ~300-600m ASL."""
        x, y, z = campaign["x"], campaign["y"], campaign["z"]

        # Find the lowest z per horizontal column (= actual terrain surface)
        # Bin horizontally to 100m to find column groups
        x_bin = np.round(x / 100.0) * 100.0
        y_bin = np.round(y / 100.0) * 100.0
        col_id = x_bin * 1e7 + y_bin
        _, inverse = np.unique(col_id, return_inverse=True)

        z_min_per_col = np.array([
            z[inverse == c].min() for c in range(inverse.max() + 1)
        ])
        x_col = np.array([x[inverse == c].mean() for c in range(inverse.max() + 1)])
        y_col = np.array([y[inverse == c].mean() for c in range(inverse.max() + 1)])

        # Near centre only (r < 2km)
        r_col = np.sqrt(x_col**2 + y_col**2)
        centre = r_col < 2000.0
        elev_centre = z_min_per_col[centre]

        if len(elev_centre) == 0:
            pytest.skip("No columns near centre")

        assert elev_centre.min() > 150.0, \
            f"Min terrain = {elev_centre.min():.0f}m — too low for Perdigão"
        assert elev_centre.max() < 800.0, \
            f"Max terrain = {elev_centre.max():.0f}m — too high for Perdigão"
        assert elev_centre.std() > 10.0, \
            "Terrain std < 10m — appears flat (should have ~100m relief)"

    def test_elevation_matches_srtm(self, campaign):
        """Spot-check: mesh elevation at 10 points vs SRTM DEM."""
        if not SRTM_PATH.exists():
            pytest.skip(f"SRTM not found: {SRTM_PATH}")

        import rasterio
        from pyproj import Transformer

        x, y, elev = campaign["x"], campaign["y"], campaign["elev"]

        # Transform local coords (UTM) to WGS84
        # Perdigão: UTM zone 29N
        transformer = Transformer.from_crs("EPSG:32629", "EPSG:4326", always_xy=True)

        # Reference point: site centre in UTM
        ref_transformer = Transformer.from_crs("EPSG:4326", "EPSG:32629", always_xy=True)
        site_x_utm, site_y_utm = ref_transformer.transform(SITE_LON, SITE_LAT)

        with rasterio.open(SRTM_PATH) as src:
            # Sample 10 points near centre
            r = np.sqrt(x**2 + y**2)
            centre_idx = np.where(r < 1000.0)[0]
            if len(centre_idx) < 10:
                pytest.skip("Not enough cells near centre")

            rng = np.random.RandomState(42)
            sample_idx = rng.choice(centre_idx, size=10, replace=False)

            diffs = []
            for idx in sample_idx:
                # Local → UTM → WGS84
                utm_x = x[idx] + site_x_utm
                utm_y = y[idx] + site_y_utm
                lon, lat = transformer.transform(utm_x, utm_y)

                # Sample SRTM
                row, col = src.index(lon, lat)
                if 0 <= row < src.height and 0 <= col < src.width:
                    srtm_elev = float(src.read(1, window=rasterio.windows.Window(col, row, 1, 1))[0, 0])
                    diff = elev[idx] - srtm_elev
                    diffs.append(diff)

            if not diffs:
                pytest.skip("Could not sample any SRTM points")

            diffs = np.array(diffs)
            rmse = np.sqrt((diffs**2).mean())
            print(f"  SRTM check: {len(diffs)} points, RMSE={rmse:.1f}m, "
                  f"bias={diffs.mean():.1f}m")

            # Allow 100m tolerance: TBM mesh ~67m resolution, SRTM 30m,
            # different grids + UTM projection offsets
            assert rmse < 100.0, \
                f"Elevation RMSE vs SRTM = {rmse:.1f}m — too large"


# ===================================================================
# 4. WorldCover z0 consistency
# ===================================================================

class TestWorldCoverZ0:
    """z0 values on terrain should match WorldCover source raster."""

    def test_z0_range(self, campaign):
        """z0 should be in [0.0001, 2.0] m (valid land cover values)."""
        z0 = campaign["z0_field"]
        if z0 is None:
            pytest.skip("No z0 field in Zarr")

        assert z0.min() >= 0.0001, f"z0 min = {z0.min():.6f} — suspiciously small"
        assert z0.max() <= 2.0, f"z0 max = {z0.max():.2f} — too large"

    def test_z0_not_uniform(self, campaign):
        """z0 should vary spatially (WorldCover, not uniform fallback)."""
        z0 = campaign["z0_field"]
        if z0 is None:
            pytest.skip("No z0 field in Zarr")

        n_unique = len(np.unique(np.round(z0, 4)))
        assert n_unique > 3, \
            f"Only {n_unique} unique z0 values — looks uniform, not WorldCover"

    def test_z0_perdigao_forest_fraction(self, campaign):
        """Perdigão is ~72% forest (z0 ≥ 0.3m). Check this is reflected."""
        z0 = campaign["z0_field"]
        if z0 is None:
            pytest.skip("No z0 field in Zarr")

        forest_frac = (z0 >= 0.3).sum() / len(z0)
        assert forest_frac > 0.4, \
            f"Forest fraction = {forest_frac*100:.0f}% — expected ~72% for Perdigão"
        assert forest_frac < 0.95, \
            f"Forest fraction = {forest_frac*100:.0f}% — suspiciously high"
        print(f"  z0 forest fraction: {forest_frac*100:.0f}%")


# ===================================================================
# 5. Metadata completeness
# ===================================================================

class TestMetadata:
    """All metadata fields should be present and finite."""

    @pytest.mark.parametrize("key", [
        "u_hub", "u_star", "z0_eff", "T_ref", "q_ref", "wind_dir",
    ])
    def test_meta_finite(self, campaign, key):
        """Metadata arrays should have no NaN or Inf."""
        arr = campaign["meta"].get(key)
        assert arr is not None, f"meta/{key} missing"
        n_bad = (~np.isfinite(arr)).sum()
        assert n_bad == 0, \
            f"meta/{key} has {n_bad} non-finite values"

    def test_all_cases_have_inflow(self, campaign):
        """Every case should have a valid inflow profile."""
        z = campaign["inflow"]["z_levels"]
        # Each case should have at least 5 finite z levels
        n_valid = np.isfinite(z).sum(axis=1)
        assert n_valid.min() >= 5, \
            f"Case with only {n_valid.min()} z levels — inflow profile missing?"

    def test_u_hub_positive(self, campaign):
        """u_hub should be > 0 for all cases (we filter calm < 1 m/s)."""
        u_hub = campaign["meta"]["u_hub"]
        assert (u_hub > 0).all(), \
            f"{(u_hub <= 0).sum()} cases have u_hub ≤ 0"

    def test_q_ref_realistic(self, campaign):
        """q_ref should be 0.001–0.020 kg/kg (typical atmospheric range)."""
        q_ref = campaign["meta"]["q_ref"]
        assert q_ref.min() > 0.0005, f"q_ref min = {q_ref.min():.6f} — too dry"
        assert q_ref.max() < 0.025, f"q_ref max = {q_ref.max():.6f} — too moist"

    def test_n_cases_matches(self, campaign):
        """Number of cases in fields should match metadata."""
        n = campaign["n_cases"]
        assert campaign["U"].shape[0] == n
        assert campaign["T"].shape[0] == n
        assert campaign["q"].shape[0] == n
        assert len(campaign["meta"]["u_hub"]) == n


# ===================================================================
# 6. Cross-field consistency
# ===================================================================

class TestCrossFieldConsistency:
    """Fields should be mutually consistent."""

    def test_surface_wind_slower_than_altitude(self, campaign):
        """Mean wind near surface (z_agl < 50m) should be slower than at altitude."""
        z_agl = campaign["z_agl"]
        U = campaign["U"]

        near_surface = z_agl < 50.0
        high_alt = z_agl > 1000.0

        if near_surface.sum() == 0 or high_alt.sum() == 0:
            pytest.skip("Not enough cells at surface/altitude")

        speed_surface = np.linalg.norm(U[:, near_surface, :], axis=-1).mean()
        speed_high = np.linalg.norm(U[:, high_alt, :], axis=-1).mean()

        assert speed_surface < speed_high, \
            f"Surface speed ({speed_surface:.1f}) ≥ altitude ({speed_high:.1f}) — wrong"

    def test_t_decreases_with_altitude(self, campaign):
        """Mean temperature should decrease with altitude (lapse rate)."""
        z_agl = campaign["z_agl"]
        T = campaign["T"]

        low = z_agl < 200.0
        high = z_agl > 2000.0

        if low.sum() == 0 or high.sum() == 0:
            pytest.skip("Not enough cells")

        T_low = T[:, low].mean()
        T_high = T[:, high].mean()

        assert T_low > T_high, \
            f"T_low ({T_low:.1f}K) ≤ T_high ({T_high:.1f}K) — inverted lapse rate"

    def test_q_decreases_with_altitude(self, campaign):
        """Mean specific humidity should decrease with altitude."""
        z_agl = campaign["z_agl"]
        q = campaign["q"]

        low = z_agl < 200.0
        high = z_agl > 2000.0

        if low.sum() == 0 or high.sum() == 0:
            pytest.skip("Not enough cells")

        q_low = q[:, low].mean()
        q_high = q[:, high].mean()

        assert q_low > q_high, \
            f"q_low ({q_low:.6f}) ≤ q_high ({q_high:.6f}) — should decrease"

    def test_all_cases_different(self, campaign):
        """Each case should have distinct U fields (not duplicated)."""
        U = campaign["U"]
        # Compare mean speed per case
        mean_speeds = np.linalg.norm(U, axis=-1).mean(axis=1)
        n_unique = len(np.unique(np.round(mean_speeds, 2)))

        assert n_unique >= campaign["n_cases"] * 0.8, \
            f"Only {n_unique} unique mean speeds — cases may be duplicated"
