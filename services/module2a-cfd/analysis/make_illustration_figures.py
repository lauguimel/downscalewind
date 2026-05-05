"""
make_illustration_figures.py — Publication-quality 3D terrain + wind illustrations

Generates figures inspired by Lin et al. (2026) Nat Commun:
  1. fig_terrain_relief.png       — Pure 3D terrain with dramatic lighting
  2. fig_terrain_streamlines.png  — 3D terrain + wind streamlines (warm palette)
  3. fig_dem_z0_windfield.png     — DEM + z0 + wind field panels
  4. fig_horizontal_slices.png    — u/v/w/k horizontal slices at multiple heights

Usage
-----
    conda run -n downscalewind python make_illustration_figures.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

logger = logging.getLogger(__name__)

# --- Paths ---
ROOT = Path(__file__).resolve().parents[3]
SRTM_PATH = ROOT / "data" / "raw" / "srtm_perdigao_30m.tif"
WORLDCOVER_PATH = ROOT / "data" / "raw" / "worldcover_perdigao.tif"
ZARR_100M = ROOT / "data" / "cfd-database" / "perdigao" / "20170511T12_100m" / "fields.zarr"
ZARR_500M = ROOT / "data" / "cfd-database" / "perdigao" / "20170511T12_500m" / "fields.zarr"
OUT_DIR = ROOT / "data" / "validation" / "figures" / "illustrations"

SITE_LAT, SITE_LON = 39.716, -7.740

# Vertical exaggeration for all 3D renders
VERT_EXAG = 5.0


# --- Windy-style warm colormap ---
def _windy_cmap():
    """Yellow → orange → red → magenta → purple wind speed colormap."""
    colors = [
        "#ffffb2",  # pale yellow (calm)
        "#fed976",  # yellow
        "#feb24c",  # light orange
        "#fd8d3c",  # orange
        "#fc4e2a",  # red-orange
        "#e31a1c",  # red
        "#bd0026",  # dark red
        "#800026",  # very dark red / maroon
    ]
    return mcolors.LinearSegmentedColormap.from_list("windy", colors, N=256)


WINDY_CMAP = _windy_cmap()


# =====================================================================
# Data loaders
# =====================================================================
def load_srtm_crop(center_lat, center_lon, half_extent_deg=0.08):
    """Load SRTM DEM cropped around site center, return x/y in meters."""
    import rasterio
    from rasterio.windows import from_bounds

    with rasterio.open(str(SRTM_PATH)) as src:
        window = from_bounds(
            center_lon - half_extent_deg,
            center_lat - half_extent_deg,
            center_lon + half_extent_deg,
            center_lat + half_extent_deg,
            src.transform,
        )
        dem = src.read(1, window=window).astype(np.float32)
        transform = src.window_transform(window)
        nrows, ncols = dem.shape
        cols, rows = np.meshgrid(np.arange(ncols), np.arange(nrows))
        lon = transform[2] + cols * transform[0]
        lat = transform[5] + rows * transform[4]

    x_m = (lon - center_lon) * 111320 * np.cos(np.radians(center_lat))
    y_m = (lat - center_lat) * 110540
    return x_m, y_m, dem, lon, lat


def load_worldcover_crop(center_lat, center_lon, half_extent_deg=0.08):
    """Load WorldCover → z0 map."""
    import rasterio
    from rasterio.windows import from_bounds

    with rasterio.open(str(WORLDCOVER_PATH)) as src:
        window = from_bounds(
            center_lon - half_extent_deg,
            center_lat - half_extent_deg,
            center_lon + half_extent_deg,
            center_lat + half_extent_deg,
            src.transform,
        )
        lc = src.read(1, window=window)
        transform = src.window_transform(window)
        nrows, ncols = lc.shape
        cols, rows = np.meshgrid(np.arange(ncols), np.arange(nrows))
        lon = transform[2] + cols * transform[0]
        lat = transform[5] + rows * transform[4]

    z0_map = {
        10: 1.0, 20: 0.5, 30: 0.05, 40: 0.1, 50: 1.0,
        60: 0.005, 70: 0.001, 80: 0.0002, 90: 0.3, 95: 1.0, 100: 0.03,
    }
    z0 = np.vectorize(lambda c: z0_map.get(c, 0.05))(lc).astype(np.float32)
    return lon, lat, lc, z0


def load_cfd_fields(zarr_path):
    """Load CFD fields from Zarr store."""
    import zarr
    store = zarr.open(str(zarr_path), mode="r")
    data = {}
    for key in ["x", "y", "z", "U", "k"]:
        data[key] = np.array(store[key][:])
    n = min(len(data["x"]), len(data["U"]), len(data["k"]))
    for key in data:
        data[key] = data[key][:n]
    data["speed"] = np.linalg.norm(data["U"], axis=1)
    return data


def make_horizontal_slice(cfd, mask, xg, yg, target_z, dz=100):
    """Interpolate CFD fields to a horizontal slice at target_z ± dz."""
    from scipy.interpolate import griddata

    x, y, z = cfd["x"][mask], cfd["y"][mask], cfd["z"][mask]
    zmask = np.abs(z - target_z) < dz
    if zmask.sum() < 100:
        return None

    xz, yz = x[zmask], y[zmask]
    Xg, Yg = np.meshgrid(xg, yg)
    slices = {}
    U = cfd["U"][mask]
    slices["u"] = griddata((xz, yz), U[zmask, 0], (Xg, Yg), method="linear")
    slices["v"] = griddata((xz, yz), U[zmask, 1], (Xg, Yg), method="linear")
    slices["w"] = griddata((xz, yz), U[zmask, 2], (Xg, Yg), method="linear")
    slices["k"] = griddata((xz, yz), cfd["k"][mask][zmask], (Xg, Yg), method="linear")
    slices["speed"] = griddata(
        (xz, yz), cfd["speed"][mask][zmask], (Xg, Yg), method="linear"
    )
    return slices


# =====================================================================
# PyVista terrain builder with vertical exaggeration + lighting
# =====================================================================
def _build_terrain_mesh(half_deg=0.06, step=3):
    """Build a PyVista terrain mesh with vertical exaggeration."""
    import pyvista as pv

    x_m, y_m, dem, _, _ = load_srtm_crop(SITE_LAT, SITE_LON, half_deg)
    xs = x_m[::step, ::step]
    ys = y_m[::step, ::step]
    zs = dem[::step, ::step].astype(np.float64) * VERT_EXAG

    grid = pv.StructuredGrid(xs, ys, zs)
    grid["elevation"] = (zs / VERT_EXAG).ravel(order="F")
    return grid


def _setup_terrain_lighting(plotter):
    """Add dramatic multi-source lighting with shadows."""
    import pyvista as pv

    # Remove default lights
    plotter.remove_all_lights()

    # Key light — warm sun from the west, elevated
    key = pv.Light(
        position=(-15000, -5000, 12000 * VERT_EXAG),
        focal_point=(0, 0, 400 * VERT_EXAG),
        intensity=0.85,
        color="#fff5e6",
        shadow_attenuation=0.5,
    )
    key.positional = False
    plotter.add_light(key)

    # Fill light — cooler, from the east, softer
    fill = pv.Light(
        position=(10000, 8000, 5000 * VERT_EXAG),
        focal_point=(0, 0, 300 * VERT_EXAG),
        intensity=0.3,
        color="#d0e0f0",
    )
    fill.positional = False
    plotter.add_light(fill)

    # Rim light — subtle backlight for depth
    rim = pv.Light(
        position=(5000, -12000, 8000 * VERT_EXAG),
        focal_point=(0, 0, 400 * VERT_EXAG),
        intensity=0.2,
        color="#ffffff",
    )
    rim.positional = False
    plotter.add_light(rim)


# =====================================================================
# Figure 1: Pure terrain relief — dramatic lighting, no color
# =====================================================================
def figure_terrain_relief():
    """3D terrain with vertical exag, grey tones, dramatic shadows."""
    import pyvista as pv

    logger.info("Generating: Pure terrain relief...")
    pv.OFF_SCREEN = True

    grid = _build_terrain_mesh(half_deg=0.07, step=3)

    pl = pv.Plotter(off_screen=True, window_size=[2400, 1600])
    pl.set_background("#1a1a2e", top="#16213e")  # dark moody background

    _setup_terrain_lighting(pl)

    # Terrain: neutral grey with PBR for realistic shading
    pl.add_mesh(
        grid,
        color="#b0a898",  # warm grey/stone
        smooth_shading=True,
        pbr=True,
        metallic=0.0,
        roughness=0.65,
        show_scalar_bar=False,
        specular=0.15,
    )

    # Camera: oblique 3/4 view
    pl.camera_position = [
        (-8000, -12000, 5000 * VERT_EXAG),
        (0, 0, 350 * VERT_EXAG),
        (0, 0, 1),
    ]
    pl.camera.zoom(1.15)

    out = OUT_DIR / "fig_terrain_relief.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.screenshot(str(out), transparent_background=False)
    pl.close()
    logger.info("Saved: %s", out)


# =====================================================================
# Figure 1b: Terrain relief with subtle elevation shading
# =====================================================================
def figure_terrain_elevation():
    """3D terrain with muted earth-tone elevation colormap."""
    import pyvista as pv

    logger.info("Generating: Terrain with elevation shading...")
    pv.OFF_SCREEN = True

    grid = _build_terrain_mesh(half_deg=0.07, step=3)

    # Muted earth tones colormap (valley green → ridge warm sand)
    earth_colors = [
        "#2d4a22",  # dark forest green (valleys)
        "#4a6b3a",  # olive green
        "#7a8b5a",  # sage
        "#a8a070",  # khaki
        "#c8b888",  # sand
        "#d8c8a0",  # warm beige
        "#e8dab8",  # light cream (ridges)
    ]
    earth_cmap = mcolors.LinearSegmentedColormap.from_list("earth", earth_colors, N=256)

    pl = pv.Plotter(off_screen=True, window_size=[2400, 1600])
    pl.set_background("#f5f0e8", top="#d8d0c0")  # warm off-white

    _setup_terrain_lighting(pl)

    elev = grid["elevation"]
    pl.add_mesh(
        grid, scalars="elevation",
        cmap=earth_cmap,
        clim=[elev.min(), elev.max()],
        smooth_shading=True,
        pbr=True,
        metallic=0.0,
        roughness=0.7,
        show_scalar_bar=True,
        scalar_bar_args={
            "title": "Elevation [m]", "vertical": True,
            "position_x": 0.88, "position_y": 0.15,
            "width": 0.05, "height": 0.35, "fmt": "%.0f",
            "title_font_size": 16, "label_font_size": 13,
            "color": "#3a3a3a",
        },
    )

    pl.camera_position = [
        (-8000, -12000, 5000 * VERT_EXAG),
        (0, 0, 350 * VERT_EXAG),
        (0, 0, 1),
    ]
    pl.camera.zoom(1.15)

    out = OUT_DIR / "fig_terrain_elevation.png"
    pl.screenshot(str(out), transparent_background=False)
    pl.close()
    logger.info("Saved: %s", out)


# =====================================================================
# Figure 2: Terrain + streamlines (warm Windy-style palette)
# =====================================================================
def figure_terrain_streamlines():
    """3D terrain + wind streamlines with warm colormap."""
    import pyvista as pv

    logger.info("Generating: Terrain + streamlines...")
    pv.OFF_SCREEN = True

    grid = _build_terrain_mesh(half_deg=0.06, step=4)

    # --- CFD data for streamlines ---
    cfd = load_cfd_fields(ZARR_100M)
    half = 6000
    region_mask = (
        (np.abs(cfd["x"]) < half * 1.2) &
        (np.abs(cfd["y"]) < half * 1.2) &
        (cfd["z"] < 1500)
    )
    pts = np.column_stack([
        cfd["x"][region_mask],
        cfd["y"][region_mask],
        cfd["z"][region_mask] * VERT_EXAG,  # scale z for streamlines too
    ])
    cloud = pv.PolyData(pts)
    # Scale w component by VERT_EXAG to match the exaggerated z-axis
    U_scaled = cfd["U"][region_mask].copy()
    U_scaled[:, 2] *= VERT_EXAG
    cloud["U"] = U_scaled
    cloud["speed"] = cfd["speed"][region_mask]

    # Interpolate to structured grid
    nx, ny, nz = 80, 80, 25
    xi = np.linspace(-half, half, nx)
    yi = np.linspace(-half, half, ny)
    zi = np.linspace(350 * VERT_EXAG, 1200 * VERT_EXAG, nz)
    Xg, Yg, Zg = np.meshgrid(xi, yi, zi, indexing="ij")
    vol = pv.StructuredGrid(Xg, Yg, Zg)

    logger.info("Interpolating CFD → structured grid...")
    vol_interp = vol.interpolate(cloud, radius=1500, sharpness=6, null_value=0)
    if "U" in vol_interp.point_data:
        u_arr = vol_interp["U"]
        # Compute speed from original (unscaled) horizontal components
        vol_interp["speed"] = np.sqrt(u_arr[:, 0]**2 + u_arr[:, 1]**2 + (u_arr[:, 2] / VERT_EXAG)**2)
    vol_interp.set_active_vectors("U")

    # Seed streamlines from SSW (wind from 220°)
    all_sl = []
    for z_seed_real in [500, 650, 850, 1050]:
        z_seed = z_seed_real * VERT_EXAG
        n_per = 35
        src_x = np.linspace(-half * 0.7, half * 0.7, n_per)
        src_y = np.full(n_per, -half * 0.8)
        src_z = np.full(n_per, z_seed)
        source = pv.PolyData(np.column_stack([src_x, src_y, src_z]))

        sl = vol_interp.streamlines_from_source(
            source, vectors="U",
            initial_step_length=30,
            terminal_speed=0.05,
            max_steps=20000,
        )
        logger.info("  z=%dm → %d pts", z_seed_real, sl.n_points)
        if sl.n_points > 10:
            all_sl.append(sl)

    # --- Render ---
    pl = pv.Plotter(off_screen=True, window_size=[2800, 1800])
    pl.set_background("#1a1a2e", top="#16213e")

    _setup_terrain_lighting(pl)

    # Terrain: dark grey for contrast with warm streamlines
    pl.add_mesh(
        grid,
        color="#706860",
        smooth_shading=True,
        pbr=True,
        metallic=0.0,
        roughness=0.7,
        show_scalar_bar=False,
    )

    # Streamlines with windy colormap
    for i, sl in enumerate(all_sl):
        if sl.n_points == 0:
            continue
        if "U" in sl.point_data:
            u_sl = sl["U"]
            sl["Wind speed [m/s]"] = np.sqrt(
                u_sl[:, 0]**2 + u_sl[:, 1]**2 + (u_sl[:, 2] / VERT_EXAG)**2
            )

        bar_args = None
        if i == 0:
            bar_args = {
                "title": "Wind speed [m/s]", "vertical": True,
                "position_x": 0.02, "position_y": 0.15,
                "width": 0.05, "height": 0.35, "fmt": "%.1f",
                "title_font_size": 16, "label_font_size": 13,
                "color": "#e0e0e0",
            }

        # Vary opacity by layer (lower = more opaque)
        opacity = max(0.4, 1.0 - i * 0.15)
        pl.add_mesh(
            sl, scalars="Wind speed [m/s]",
            cmap=WINDY_CMAP,
            clim=[2, 10],
            line_width=max(2, 5 - i),
            opacity=opacity,
            show_scalar_bar=(i == 0),
            scalar_bar_args=bar_args,
            render_lines_as_tubes=True,
        )

    # Camera
    pl.camera_position = [
        (-6000, -11000, 5500 * VERT_EXAG),
        (500, 500, 400 * VERT_EXAG),
        (0, 0, 1),
    ]
    pl.camera.zoom(1.2)

    out = OUT_DIR / "fig_terrain_streamlines.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.screenshot(str(out), transparent_background=False)
    pl.close()
    logger.info("Saved: %s", out)


# =====================================================================
# Figure 3: DEM + z0 + wind field panels (matplotlib)
# =====================================================================
def figure_dem_z0_windfield():
    """4-panel figure: hillshade, DEM, z0, wind field."""
    logger.info("Generating: DEM + z0 + wind panels...")

    half_deg = 0.06
    x_m, y_m, dem, lon, lat = load_srtm_crop(SITE_LAT, SITE_LON, half_deg)
    lon_wc, lat_wc, lc, z0 = load_worldcover_crop(SITE_LAT, SITE_LON, half_deg)

    from matplotlib.colors import LightSource
    ls = LightSource(azdeg=315, altdeg=35)

    cfd = load_cfd_fields(ZARR_500M)
    half = 8000
    mask = (
        (np.abs(cfd["x"]) < half) & (np.abs(cfd["y"]) < half) & (cfd["z"] < 1500)
    )
    xg = np.linspace(-half, half, 100)
    yg = np.linspace(-half, half, 100)
    slice_ns = make_horizontal_slice(cfd, mask, xg, yg, target_z=300, dz=200)

    # Earth-tone colormaps
    earth_colors = ["#2d4a22", "#4a6b3a", "#7a8b5a", "#a8a070", "#c8b888", "#d8c8a0", "#e8dab8"]
    earth_cmap = mcolors.LinearSegmentedColormap.from_list("earth", earth_colors, N=256)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))

    # (a) Hillshade terrain
    ax = axes[0]
    rgb = ls.shade(dem, cmap=plt.cm.Greys_r, vert_exag=6, blend_mode="soft", dx=30, dy=30)
    ax.imshow(rgb, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
              origin="upper", aspect="equal")
    ax.set_title("Region", fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude [°]")
    ax.set_ylabel("Latitude [°]")

    # (b) DEM
    ax = axes[1]
    im = ax.imshow(dem, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                   origin="upper", cmap=earth_cmap, aspect="equal")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Elevation [m]")
    ax.set_title("DEM", fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude [°]")
    ax.set_ylabel("")

    # (c) z0
    ax = axes[2]
    z0_log = np.log10(np.clip(z0, 1e-4, 10))
    z0_colors = ["#f7f4f0", "#e8d8b8", "#c8a860", "#a07830", "#704820", "#402810"]
    z0_cmap = mcolors.LinearSegmentedColormap.from_list("z0_earth", z0_colors, N=256)
    im = ax.imshow(z0_log,
                   extent=[lon_wc.min(), lon_wc.max(), lat_wc.min(), lat_wc.max()],
                   origin="upper", cmap=z0_cmap, vmin=-3, vmax=0.5, aspect="equal")
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())
    plt.colorbar(im, ax=ax, shrink=0.8, label="log$_{10}$(z$_0$) [m]")
    ax.set_title("z$_0$", fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude [°]")
    ax.set_ylabel("")

    # (d) Wind speed field
    ax = axes[3]
    if slice_ns is not None:
        Xg_2d, Yg_2d = np.meshgrid(xg, yg)
        im = ax.pcolormesh(
            Xg_2d / 1000, Yg_2d / 1000, slice_ns["speed"],
            cmap=WINDY_CMAP, vmin=2, vmax=9, shading="auto",
        )
        # Sparse streamlines overlay
        u_sl = slice_ns["u"].copy()
        v_sl = slice_ns["v"].copy()
        u_sl[np.isnan(u_sl)] = 0
        v_sl[np.isnan(v_sl)] = 0
        ax.streamplot(
            xg / 1000, yg / 1000, u_sl.T, v_sl.T,
            color="white", linewidth=0.6, density=1.2, arrowsize=0.8,
        )
        plt.colorbar(im, ax=ax, shrink=0.8, label="Wind speed [m/s]")
    ax.set_title("3D Wind Field", fontsize=14, fontweight="bold")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_aspect("equal")

    plt.tight_layout()
    out = OUT_DIR / "fig_dem_z0_windfield.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved: %s", out)


# =====================================================================
# Figure 4: Horizontal slices u/v/w/k (style Fig. 7)
# =====================================================================
def figure_horizontal_slices():
    """Multi-panel: (a) 3D terrain, (b) DEM/z0, (c) u/v/w/k at 4 heights."""
    logger.info("Generating: Horizontal slices...")

    x_m, y_m, dem, lon, lat = load_srtm_crop(SITE_LAT, SITE_LON, half_extent_deg=0.06)
    lon_wc, lat_wc, lc, z0 = load_worldcover_crop(SITE_LAT, SITE_LON, half_extent_deg=0.06)

    cfd = load_cfd_fields(ZARR_100M)
    half = 8000
    mask = (
        (np.abs(cfd["x"]) < half) & (np.abs(cfd["y"]) < half) & (cfd["z"] < 1500)
    )
    xg = np.linspace(-half, half, 150)
    yg = np.linspace(-half, half, 150)

    heights = [300, 400, 600, 1000]
    height_labels = ["h≈50m", "h≈150m", "h≈350m", "h≈750m"]
    variables = ["u", "v", "w", "k"]
    var_labels = ["u [m/s]", "v [m/s]", "w [m/s]", "k [m²/s²]"]

    # Colormaps: warm diverging for u/v/w, sequential warm for k
    uv_colors = ["#0a3878", "#2166ac", "#67a9cf", "#f7f7f7", "#ef8a62", "#b2182b", "#67001f"]
    uv_cmap = mcolors.LinearSegmentedColormap.from_list("warm_div", uv_colors, N=256)
    k_colors = ["#fff5eb", "#fee6ce", "#fdd0a2", "#fdae6b", "#fd8d3c", "#e6550d", "#a63603"]
    k_cmap = mcolors.LinearSegmentedColormap.from_list("warm_seq", k_colors, N=256)

    cmaps = [uv_cmap, uv_cmap, uv_cmap, k_cmap]
    clims = [(-2, 10), (-3, 7), (-2, 2), (0, 0.5)]

    # Earth-tone colormaps
    earth_colors = ["#2d4a22", "#4a6b3a", "#7a8b5a", "#a8a070", "#c8b888", "#d8c8a0", "#e8dab8"]
    earth_cmap = mcolors.LinearSegmentedColormap.from_list("earth", earth_colors, N=256)
    z0_colors = ["#f7f4f0", "#e8d8b8", "#c8a860", "#a07830", "#704820", "#402810"]
    z0_cmap = mcolors.LinearSegmentedColormap.from_list("z0_earth", z0_colors, N=256)

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(4, 5, figure=fig, width_ratios=[1.3, 1, 1, 1, 1],
                  wspace=0.15, hspace=0.25)

    # (a) 3D terrain
    ax3d = fig.add_subplot(gs[0:2, 0], projection="3d")
    step = 6
    xp = x_m[::step, ::step] / 1000
    yp = y_m[::step, ::step] / 1000
    zp = dem[::step, ::step]

    from matplotlib.colors import LightSource
    ls = LightSource(azdeg=315, altdeg=35)
    rgb = ls.shade(zp, cmap=plt.cm.Greys_r, vert_exag=5, blend_mode="soft")
    ax3d.plot_surface(xp, yp, zp, facecolors=rgb, linewidth=0,
                      antialiased=True, rstride=1, cstride=1, shade=False)
    ax3d.set_xlabel("x [km]", fontsize=9, labelpad=2)
    ax3d.set_ylabel("y [km]", fontsize=9, labelpad=2)
    ax3d.set_zlabel("z [m]", fontsize=9, labelpad=2)
    ax3d.view_init(elev=35, azim=-60)
    ax3d.set_title("(a)  Topography", fontsize=12, fontweight="bold", pad=10)
    ax3d.tick_params(labelsize=7)

    # (b) DEM
    ax_dem = fig.add_subplot(gs[2, 0])
    im_dem = ax_dem.imshow(dem, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                           origin="upper", cmap=earth_cmap, aspect="equal")
    plt.colorbar(im_dem, ax=ax_dem, shrink=0.8, label="m")
    ax_dem.set_title("(b)  DEM", fontsize=11, fontweight="bold")
    ax_dem.tick_params(labelsize=7)

    # (b') z0
    ax_z0 = fig.add_subplot(gs[3, 0])
    z0_log = np.log10(np.clip(z0, 1e-4, 10))
    im_z0 = ax_z0.imshow(z0_log, extent=[lon_wc.min(), lon_wc.max(), lat_wc.min(), lat_wc.max()],
                          origin="upper", cmap=z0_cmap, vmin=-3, vmax=0.5, aspect="equal")
    ax_z0.set_xlim(lon.min(), lon.max())
    ax_z0.set_ylim(lat.min(), lat.max())
    plt.colorbar(im_z0, ax=ax_z0, shrink=0.8, label="log₁₀(z₀)")
    ax_z0.set_title("(b')  z$_0$", fontsize=11, fontweight="bold")
    ax_z0.tick_params(labelsize=7)

    # Right grid: 4 heights × 4 variables
    logger.info("Computing horizontal slices...")
    for i, (h, hlabel) in enumerate(zip(heights, height_labels)):
        dz = 80 if h < 500 else 150
        sl = make_horizontal_slice(cfd, mask, xg, yg, target_z=h, dz=dz)
        if sl is None:
            continue

        Xg_2d, Yg_2d = np.meshgrid(xg, yg)
        for j, (var, vlabel, cmap, clim) in enumerate(
            zip(variables, var_labels, cmaps, clims)
        ):
            ax = fig.add_subplot(gs[i, j + 1])
            im = ax.pcolormesh(Xg_2d / 1000, Yg_2d / 1000, sl[var],
                               cmap=cmap, vmin=clim[0], vmax=clim[1],
                               shading="auto", rasterized=True)
            if i == 0:
                ax.set_title(vlabel, fontsize=11, fontweight="bold")
            if i == len(heights) - 1:
                ax.set_xlabel("x [km]", fontsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(f"{hlabel}\ny [km]", fontsize=9)
            else:
                ax.set_yticklabels([])
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)
            if i == len(heights) - 1:
                plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)

    out = OUT_DIR / "fig_horizontal_slices.png"
    fig.savefig(str(out), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved: %s", out)


# =====================================================================
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Output → %s", OUT_DIR)

    figure_terrain_relief()
    figure_terrain_elevation()
    figure_terrain_streamlines()
    figure_dem_z0_windfield()
    figure_horizontal_slices()

    logger.info("Done — all figures in %s", OUT_DIR)


if __name__ == "__main__":
    main()
