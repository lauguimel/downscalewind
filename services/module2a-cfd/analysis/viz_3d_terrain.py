"""
viz_3d_terrain.py — Publication-quality 3D terrain + streamlines visualization

Reads CFD results from fields.zarr and terrain STL, produces:
  - Static 3D rendering with terrain surface (colored by elevation)
  - Streamlines colored by wind speed
  - Pale blue atmosphere bounding box
  - Tower positions marked

Usage
-----
    python viz_3d_terrain.py \
        --case-dir data/cases/perdigao_500m_20170511T12 \
        --zarr     data/cfd-database/perdigao/20170511T12_500m/fields.zarr \
        --output   data/validation/500m_20170511T12/viz_3d.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def create_visualization(
    case_dir: Path,
    zarr_path: Path,
    output_path: Path,
    resolution_label: str = "",
    camera_position: str = "sw",
) -> None:
    """Create publication-quality 3D terrain + streamlines figure."""
    import pyvista as pv
    import zarr

    pv.OFF_SCREEN = True
    pv.global_theme.font.color = "black"
    pv.global_theme.font.size = 14

    # ---------------------------------------------------------------
    # Load terrain STL
    # ---------------------------------------------------------------
    stl_path = case_dir / "constant" / "triSurface" / "terrain.stl"
    if not stl_path.exists():
        raise FileNotFoundError(f"Terrain STL not found: {stl_path}")

    terrain = pv.read(str(stl_path))
    logger.info("Terrain STL: %d triangles", terrain.n_cells)

    # ---------------------------------------------------------------
    # Load CFD fields from Zarr
    # ---------------------------------------------------------------
    store = zarr.open(str(zarr_path), mode="r")
    x = np.array(store["x"][:])
    y = np.array(store["y"][:])
    z = np.array(store["z"][:])
    U = np.array(store["U"][:])  # (N, 3)
    k = np.array(store["k"][:])

    n_cells = len(x)
    logger.info("CFD field: %d cells, U shape: %s", n_cells, U.shape)

    # Truncate to min length if fluidfoam includes boundary values
    n = min(n_cells, len(U))
    x, y, z, U, k = x[:n], y[:n], z[:n], U[:n], k[:n]

    # Create unstructured point cloud → interpolate to structured grid for streamlines
    points = np.column_stack([x, y, z])
    cloud = pv.PolyData(points)
    cloud["U"] = U
    cloud["speed"] = np.linalg.norm(U, axis=1)
    cloud["k"] = k

    # ---------------------------------------------------------------
    # Create structured grid for streamlines (subsampled)
    # ---------------------------------------------------------------
    # Find bounds
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()

    # Focus on the central region where terrain is interesting
    cx, cy = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)
    half_extent = 8000  # 8 km — tighter view on ridges
    z_top = min(z_max, 1500)  # cap at 1500m for viz

    # Structured grid for interpolation
    nx, ny, nz = 80, 80, 25
    xi = np.linspace(cx - half_extent, cx + half_extent, nx)
    yi = np.linspace(cy - half_extent, cy + half_extent, ny)
    zi = np.linspace(z_min + 50, z_top, nz)
    xg, yg, zg = np.meshgrid(xi, yi, zi, indexing="ij")

    grid = pv.StructuredGrid(xg, yg, zg)

    # Interpolate CFD fields onto structured grid
    logger.info("Interpolating fields onto structured grid (%dx%dx%d)...", nx, ny, nz)
    grid_interp = grid.interpolate(cloud, radius=2000, sharpness=4, null_value=0)
    # Compute speed on the interpolated grid
    if "U" in grid_interp.point_data:
        u_interp = grid_interp["U"]
        grid_interp["speed"] = np.linalg.norm(u_interp, axis=1)
    grid_interp.set_active_vectors("U")

    # ---------------------------------------------------------------
    # Streamlines
    # ---------------------------------------------------------------
    # Source line upwind (SW for this case)
    n_lines = 40
    src_x = np.full(n_lines, cx - half_extent * 0.85)
    src_y = np.linspace(cy - half_extent * 0.7, cy + half_extent * 0.7, n_lines)
    src_z_base = 150  # m AGL approximate — close to ground
    src_z = np.full(n_lines, z_min + src_z_base)
    source_pts = pv.PolyData(np.column_stack([src_x, src_y, src_z]))

    logger.info("Computing streamlines...")
    streamlines = grid_interp.streamlines_from_source(
        source_pts,
        vectors="U",
        initial_step_length=100,
        terminal_speed=0.5,
        max_steps=5000,
    )

    # Add a second layer higher up
    src_z2 = np.full(n_lines, z_min + 500)
    source_pts2 = pv.PolyData(np.column_stack([src_x, src_y, src_z2]))
    streamlines2 = grid_interp.streamlines_from_source(
        source_pts2,
        vectors="U",
        max_time=80000,
        initial_step_length=100,
        terminal_speed=0.5,
        max_steps=5000,
    )

    # ---------------------------------------------------------------
    # Build scene
    # ---------------------------------------------------------------
    plotter = pv.Plotter(
        off_screen=True,
        window_size=[2400, 1600],
    )

    # Background: pale blue atmosphere gradient
    plotter.set_background("#e8f0fe", top="#b8d4f0")

    # Terrain surface — colored by elevation
    terrain_z = terrain.points[:, 2]
    terrain["elevation_m"] = terrain_z
    plotter.add_mesh(
        terrain,
        scalars="elevation_m",
        cmap="terrain",
        clim=[terrain_z.min(), terrain_z.max()],
        opacity=1.0,
        show_scalar_bar=True,
        scalar_bar_args={
            "title": "Elevation [m]",
            "vertical": True,
            "position_x": 0.88,
            "position_y": 0.15,
            "width": 0.08,
            "height": 0.35,
            "fmt": "%.0f",
        },
        lighting=True,
        smooth_shading=True,
    )

    # Streamlines — colored by speed
    def _add_streamlines(plotter, sl, line_width, opacity, show_bar):
        if sl.n_points == 0:
            return
        # Compute speed on streamline points
        if "U" in sl.point_data:
            sl["speed"] = np.linalg.norm(sl["U"], axis=1)
        elif "speed" not in sl.point_data:
            # Fallback: uniform color
            plotter.add_mesh(sl, color="#4477aa", line_width=line_width,
                             opacity=opacity)
            return

        bar_args = None
        if show_bar:
            bar_args = {
                "title": "Wind speed [m/s]",
                "vertical": True,
                "position_x": 0.02,
                "position_y": 0.15,
                "width": 0.08,
                "height": 0.35,
                "fmt": "%.1f",
            }
        plotter.add_mesh(
            sl,
            scalars="speed",
            cmap="coolwarm",
            clim=[0, 12],
            line_width=line_width,
            opacity=opacity,
            show_scalar_bar=show_bar,
            scalar_bar_args=bar_args,
            render_lines_as_tubes=True,
        )

    _add_streamlines(plotter, streamlines, line_width=4, opacity=0.9, show_bar=True)
    _add_streamlines(plotter, streamlines2, line_width=3, opacity=0.6, show_bar=False)

    # Bounding box — very pale blue
    domain_box = pv.Box(bounds=[
        cx - half_extent, cx + half_extent,
        cy - half_extent, cy + half_extent,
        z_min, z_top,
    ])
    plotter.add_mesh(
        domain_box,
        style="wireframe",
        color="#a0b8d0",
        line_width=1.0,
        opacity=0.4,
    )

    # Tower markers (key towers relative to site center)
    tower_info = {
        "tse04": {"dx": -380, "dy": -1100, "alt": 473},
        "tse09": {"dx": 440, "dy": -540, "alt": 305},
        "tse13": {"dx": 930, "dy": 180, "alt": 453},
    }
    for tid, info in tower_info.items():
        tower_pos = np.array([[cx + info["dx"], cy + info["dy"], info["alt"] + 120]])
        tower_pt = pv.PolyData(tower_pos)
        plotter.add_mesh(tower_pt, color="red", point_size=12, render_points_as_spheres=True)
        plotter.add_point_labels(
            tower_pt, [tid], font_size=14, point_color="red",
            text_color="black", bold=True, shape=None,
        )

    # Camera position — closer, lower angle for dramatic view
    if camera_position == "sw":
        plotter.camera_position = [
            (cx - half_extent * 1.5, cy - half_extent * 1.0, z_top * 0.8),
            (cx, cy, z_min + 400),
            (0, 0, 1),
        ]
    elif camera_position == "top":
        plotter.camera_position = [
            (cx, cy, z_top * 3),
            (cx, cy, z_min + 200),
            (0, 1, 0),
        ]

    # Title
    title = "Perdigão CFD — simpleFoam k-ε"
    if resolution_label:
        title += f" (Δx = {resolution_label})"
    title += "\n2017-05-11 12:00 UTC, SSW 217°"
    plotter.add_text(title, position="upper_left", font_size=12, color="black")

    # Render
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(output_path))
    plotter.close()
    logger.info("3D visualization saved: %s", output_path)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="3D terrain + streamlines visualization")
    parser.add_argument("--case-dir", required=True, help="OpenFOAM case directory")
    parser.add_argument("--zarr",     required=True, help="fields.zarr from export_cfd.py")
    parser.add_argument("--output",   required=True, help="Output PNG path")
    parser.add_argument("--resolution", default="", help="Resolution label (e.g., '500m')")
    parser.add_argument("--camera",   default="sw", choices=["sw", "top"],
                        help="Camera position")
    args = parser.parse_args()

    create_visualization(
        case_dir=Path(args.case_dir),
        zarr_path=Path(args.zarr),
        output_path=Path(args.output),
        resolution_label=args.resolution,
        camera_position=args.camera,
    )


if __name__ == "__main__":
    main()
