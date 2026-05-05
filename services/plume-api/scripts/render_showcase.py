"""render_showcase.py — Generate a side-by-side showcase animation.

Left panel:  ERA5 25 km — flat terrain, coarse wind arrows, grey mesh
Right panel: Plume 30 m — detailed terrain, wind color + streamlines

Camera path (synchronized):
  Phase 1 (frames 0–59):   wide 250 km overview, grey terrain + grid, slow zoom
  Phase 2 (frames 60–119): zoom into 4 km box, terrain + wind appear
  Phase 3 (frames 120–239): 360° orbit around the box at fixed zoom

Outputs: frames as PNGs → stitched to MP4/GIF via ffmpeg.

Usage:
    python render_showcase.py --out-dir /tmp/showcase_frames
    ffmpeg -framerate 30 -i /tmp/showcase_frames/frame_%04d.png -vf "scale=1920:-2" \
           -c:v libx264 -pix_fmt yuv420p -crf 18 showcase.mp4
    # Or for GIF:
    ffmpeg -framerate 30 -i /tmp/showcase_frames/frame_%04d.png -vf "fps=15,scale=960:-1" \
           -loop 0 showcase.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv

# ── Synthetic data generation ──────────────────────────────────────────────


def log_z_levels(nz=32):
    return np.geomspace(5.0, 5000.0, nz).astype(np.float32)


def make_terrain_hires(nx=128, ny=128, extent_m=4000.0):
    """Double-ridge terrain (Perdigão-like), 30 m resolution."""
    x = np.linspace(-extent_m / 2, extent_m / 2, nx)
    y = np.linspace(-extent_m / 2, extent_m / 2, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    theta = np.deg2rad(45.0)
    p = -X * np.sin(theta) + Y * np.cos(theta)
    r1 = 300.0 * np.exp(-((p - 400) ** 2) / (2 * 200 ** 2))
    r2 = 250.0 * np.exp(-((p + 400) ** 2) / (2 * 200 ** 2))
    elev = 200.0 + r1 + r2
    return X, Y, elev.astype(np.float32)


def make_terrain_era5(nx=8, ny=8, extent_m=200_000.0):
    """ERA5-like coarse terrain: nearly flat at 25 km, faint topography."""
    x = np.linspace(-extent_m / 2, extent_m / 2, nx)
    y = np.linspace(-extent_m / 2, extent_m / 2, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    # Very gentle slope — ERA5 averages out terrain
    elev = 180.0 + 40 * np.sin(X / 80000) * np.cos(Y / 80000)
    return X, Y, elev.astype(np.float32)


def make_wind_field(X, Y, elev, nz_display=1, z_agl=60.0):
    """Log-law SW wind field with terrain speedup. Returns u, v, speed (2D at z_agl)."""
    z0 = 0.1
    u_star = 10.0 * 0.4 / np.log(60.0 / z0)
    speed_bg = (u_star / 0.4) * np.log(max(z_agl, z0 + 1) / z0)
    # Direction: from SW (225°)
    angle = np.deg2rad(225.0)
    u_dir, v_dir = -np.sin(angle), -np.cos(angle)
    speedup = 1.0 + 0.4 * (elev - 200) / 300 * np.exp(-z_agl / 400)
    speed = speed_bg * speedup
    u = speed * u_dir
    v = speed * v_dir
    return u, v, speed


# ── Frame rendering ────────────────────────────────────────────────────────

# Phase timings (frame indices)
N_ZOOM_IN = 60     # zoom from wide to medium
N_ZOOM_DETAIL = 60 # zoom to detail + wind appears
N_ORBIT = 120      # 360° orbit
N_TOTAL = N_ZOOM_IN + N_ZOOM_DETAIL + N_ORBIT


def ease_in_out(t):
    """Smooth interpolation 0→1."""
    return t * t * (3 - 2 * t)


def render_frame(pl, frame_idx, meshes):
    """Position the camera for both subplots based on frame index."""
    era5_mesh, era5_edges, hires_mesh, hires_streamlines = meshes

    if frame_idx < N_ZOOM_IN:
        # Phase 1: wide overview → medium zoom
        t = ease_in_out(frame_idx / N_ZOOM_IN)
        zoom = 250_000 * (1 - t) + 15_000 * t
        elev_angle = 70 - 25 * t    # high → medium pitch
        azimuth = 30.0
        # ERA5 side: show grid, Plume side: also show grid (zoomed)
    elif frame_idx < N_ZOOM_IN + N_ZOOM_DETAIL:
        # Phase 2: zoom to 4 km detail
        t = ease_in_out((frame_idx - N_ZOOM_IN) / N_ZOOM_DETAIL)
        zoom = 15_000 * (1 - t) + 4_500 * t
        elev_angle = 45 - 10 * t
        azimuth = 30.0
    else:
        # Phase 3: 360° orbit
        t = (frame_idx - N_ZOOM_IN - N_ZOOM_DETAIL) / N_ORBIT
        zoom = 4_500
        elev_angle = 35
        azimuth = 30.0 + 360 * t

    # Camera position (orbit around center)
    az_rad = np.deg2rad(azimuth)
    el_rad = np.deg2rad(elev_angle)
    cx = zoom * np.cos(el_rad) * np.sin(az_rad)
    cy = zoom * np.cos(el_rad) * np.cos(az_rad)
    cz = zoom * np.sin(el_rad) + 200  # target is at ~200m elevation
    center = (0, 0, 200)

    for renderer_idx in range(2):
        pl.subplot(0, renderer_idx)
        pl.camera.position = (cx, cy, cz)
        pl.camera.focal_point = center
        pl.camera.up = (0, 0, 1)
        pl.camera.clipping_range = (10, zoom * 5)


def build_scene(pl):
    """Create all meshes and add them to the plotter. Return mesh references."""

    # ── LEFT: ERA5 (coarse, flat, grey with grid) ──
    pl.subplot(0, 0)
    pl.set_background("#0a0e1a")

    # Coarse ERA5 terrain (nearly flat)
    X_e, Y_e, Z_e = make_terrain_era5(nx=10, ny=10, extent_m=200_000)
    era5_grid = pv.StructuredGrid(X_e, Y_e, Z_e)
    era5_mesh = pl.add_mesh(
        era5_grid, color="#3a3a4a", show_edges=True,
        edge_color="#5a5a6a", line_width=2, opacity=0.9,
        lighting=True, smooth_shading=True,
    )

    # ERA5 wind as sparse arrows on the flat grid
    _, _, speed_e = make_wind_field(X_e, Y_e, Z_e, z_agl=60)
    n_pts = X_e.size
    points = np.column_stack([X_e.ravel(), Y_e.ravel(), Z_e.ravel() + 500])
    u_e, v_e, _ = make_wind_field(X_e, Y_e, Z_e, z_agl=60)
    vectors = np.column_stack([u_e.ravel(), v_e.ravel(), np.zeros(n_pts)])
    arrow_cloud = pv.PolyData(points)
    arrow_cloud["vectors"] = vectors * 200  # scale for visibility
    arrow_cloud["speed"] = speed_e.ravel()
    arrows = arrow_cloud.glyph(orient="vectors", scale="vectors", factor=0.08)
    pl.add_mesh(arrows, scalars="speed", cmap="coolwarm", show_scalar_bar=False,
                opacity=0.8)

    # Label
    pl.add_text("ERA5  —  25 km", position="upper_left", font_size=14,
                color="white", shadow=True)

    # ── RIGHT: Plume (hi-res terrain + wind) ──
    pl.subplot(0, 1)
    pl.set_background("#0a0e1a")

    X_h, Y_h, Z_h = make_terrain_hires(nx=128, ny=128, extent_m=4000)
    _, _, speed_h = make_wind_field(X_h, Y_h, Z_h, z_agl=60)
    hires_grid = pv.StructuredGrid(X_h, Y_h, Z_h)
    hires_grid["Wind speed (m/s)"] = speed_h.ravel()

    hires_mesh = pl.add_mesh(
        hires_grid, scalars="Wind speed (m/s)", cmap="inferno",
        clim=[4, 16], show_scalar_bar=True, lighting=True,
        smooth_shading=True, opacity=1.0,
    )

    # Streamlines: seed points on a line perpendicular to the wind
    u_h, v_h, _ = make_wind_field(X_h, Y_h, Z_h, z_agl=60)
    pts_stream = np.column_stack([X_h.ravel(), Y_h.ravel(), Z_h.ravel() + 60])
    vecs_stream = np.column_stack([u_h.ravel(), v_h.ravel(), np.zeros(X_h.size)])
    vol = pv.PolyData(pts_stream)
    vol["vectors"] = vecs_stream
    vol["speed"] = speed_h.ravel()
    # Seed a line for streamlines
    seed_y = np.linspace(-1800, 1800, 30)
    seed_pts = np.column_stack([
        np.full_like(seed_y, -1800),
        seed_y,
        np.full_like(seed_y, 350),  # ~60m above terrain avg
    ])
    seed = pv.PolyData(seed_pts)

    # Build streamlines manually via arrows since pyvista streamlines need
    # an UnstructuredGrid with vector data — use arrows instead for visual effect
    # (streamlines from PolyData are unreliable). Use tube-glyphs along the wind.
    subsample = pts_stream[::64]  # every 64th point
    sub_vecs = vecs_stream[::64] * 80
    sub_speed = speed_h.ravel()[::64]
    stream_cloud = pv.PolyData(subsample)
    stream_cloud["vectors"] = sub_vecs
    stream_cloud["speed"] = sub_speed
    stream_arrows = stream_cloud.glyph(orient="vectors", scale="vectors", factor=0.05)
    hires_streamlines = pl.add_mesh(
        stream_arrows, scalars="speed", cmap="inferno", clim=[4, 16],
        show_scalar_bar=False, opacity=0.7,
    )

    pl.add_text("Plume  —  30 m", position="upper_left", font_size=14,
                color="white", shadow=True)

    return era5_mesh, None, hires_mesh, hires_streamlines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/plume_showcase"))
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--preview", action="store_true",
                        help="Open interactive window instead of saving frames")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    pv.global_theme.anti_aliasing = "msaa"
    pv.global_theme.multi_rendering_splitting_position = 0.5

    if args.preview:
        pl = pv.Plotter(shape=(1, 2), window_size=[args.width, args.height],
                        border=False)
        meshes = build_scene(pl)
        render_frame(pl, N_ZOOM_IN + N_ZOOM_DETAIL + 60, meshes)
        pl.show()
        return

    # Offscreen rendering
    pl = pv.Plotter(shape=(1, 2), window_size=[args.width, args.height],
                    border=False, off_screen=True)
    meshes = build_scene(pl)

    print(f"[info] rendering {N_TOTAL} frames to {args.out_dir}")
    for i in range(N_TOTAL):
        render_frame(pl, i, meshes)
        path = args.out_dir / f"frame_{i:04d}.png"
        pl.screenshot(str(path), transparent_background=False)
        if (i + 1) % 30 == 0 or i == 0:
            print(f"  frame {i + 1}/{N_TOTAL}")

    pl.close()

    # Attempt ffmpeg stitch
    mp4_path = args.out_dir.parent / "plume_showcase.mp4"
    gif_path = args.out_dir.parent / "plume_showcase.gif"
    import subprocess
    try:
        subprocess.run([
            "ffmpeg", "-y", "-framerate", str(args.fps),
            "-i", str(args.out_dir / "frame_%04d.png"),
            "-vf", f"scale={args.width}:-2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            str(mp4_path),
        ], check=True, capture_output=True)
        print(f"[ok] MP4: {mp4_path} ({mp4_path.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        print(f"[warn] ffmpeg MP4 failed: {e}")
        print(f"  frames are in {args.out_dir}/, stitch manually with ffmpeg")

    try:
        subprocess.run([
            "ffmpeg", "-y", "-framerate", str(args.fps),
            "-i", str(args.out_dir / "frame_%04d.png"),
            "-vf", f"fps=15,scale={args.width // 2}:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            "-loop", "0",
            str(gif_path),
        ], check=True, capture_output=True)
        print(f"[ok] GIF: {gif_path} ({gif_path.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        print(f"[warn] ffmpeg GIF failed: {e}")

    print("[done]")


if __name__ == "__main__":
    main()
