import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md("""
    # Perdigão CFD — 3D Interactive Visualization

    Interactive terrain + flow visualization for the Perdigão wind field simulation.
    Select resolution and display options below.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from pathlib import Path
    import zarr
    import yaml

    return Path, go, mo, np, yaml, zarr


@app.cell
def _(mo):
    resolution = mo.ui.dropdown(
        options={"500 m": "500m", "250 m": "250m", "100 m": "100m"},
        value="500 m",
        label="Resolution",
    )
    show_streamlines = mo.ui.checkbox(value=True, label="Show streamlines")
    show_towers = mo.ui.checkbox(value=True, label="Show towers")
    z_exag = mo.ui.slider(1, 5, value=2, step=0.5, label="Vertical exaggeration")
    n_streamlines = mo.ui.slider(10, 60, value=30, step=5, label="Number of streamlines")

    mo.hstack([resolution, show_streamlines, show_towers, z_exag, n_streamlines])
    return n_streamlines, resolution, show_streamlines, show_towers, z_exag


@app.cell
def _(Path, np, resolution, yaml, zarr):
    # Load data for selected resolution
    base = Path("../../data")
    res_label = resolution.value

    case_map = {
        "500m": "perdigao_500m_20170511T12",
        "250m": "perdigao_250m_20170511T12",
        "100m": "perdigao_100m_20170511T12",
    }
    case_name = case_map[res_label]
    case_dir = base / "cases" / case_name
    zarr_path = base / f"cfd-database/perdigao/20170511T12_{res_label}/fields.zarr"

    data = {}

    # Load terrain STL as numpy arrays
    try:
        from stl import mesh as stl_mesh
        terrain_stl = stl_mesh.Mesh.from_file(
            str(case_dir / "constant" / "triSurface" / "terrain.stl")
        )
        data["terrain_vertices"] = terrain_stl.vectors
        data["terrain_ok"] = True
    except Exception as e:
        data["terrain_ok"] = False
        data["terrain_error"] = str(e)

    # Load CFD fields
    if zarr_path.exists():
        store = zarr.open(str(zarr_path), mode="r")
        data["x"] = np.array(store["x"][:])
        data["y"] = np.array(store["y"][:])
        data["z"] = np.array(store["z"][:])
        data["U"] = np.array(store["U"][:])
        data["k"] = np.array(store["k"][:])
        data["speed"] = np.linalg.norm(data["U"], axis=1)
        n = len(data["x"])
        n_u = len(data["U"])
        if n_u > n:
            for key in ["U", "k", "speed"]:
                data[key] = data[key][:n]
        data["fields_ok"] = True
    else:
        data["fields_ok"] = False

    # Load tower positions
    try:
        towers_path = Path("../../configs/sites/perdigao_towers.yaml")
        site_path = Path("../../configs/sites/perdigao.yaml")
        with open(towers_path) as f:
            towers_data = yaml.safe_load(f)["towers"]
        with open(site_path) as f:
            site_cfg = yaml.safe_load(f)
        site_lat = site_cfg["site"]["coordinates"]["latitude"]
        site_lon = site_cfg["site"]["coordinates"]["longitude"]

        DEG_PER_M_LAT = 1.0 / 111_000.0
        DEG_PER_M_LON = 1.0 / (111_000.0 * np.cos(np.radians(site_lat)))
        data["towers"] = []
        for tid, tinfo in towers_data.items():
            tx = (tinfo["lon"] - site_lon) / DEG_PER_M_LON
            ty = (tinfo["lat"] - site_lat) / DEG_PER_M_LAT
            data["towers"].append({
                "id": tid, "x": tx, "y": ty,
                "alt": tinfo["altitude_m"],
                "max_h": max(tinfo.get("heights_m", [10])),
            })
        data["towers_ok"] = True
    except Exception:
        data["towers_ok"] = False

    data
    return data, res_label


@app.cell
def _(
    data,
    go,
    mo,
    n_streamlines,
    np,
    res_label,
    show_streamlines,
    show_towers,
    z_exag,
):
    if not data.get("fields_ok"):
        mo.md(f"**No CFD results found for {res_label}** — run simpleFoam first.")
    else:
        fig = go.Figure()

        zex = z_exag.value
        x = data["x"]
        y = data["y"]
        z = data["z"]
        speed = data["speed"]
        U = data["U"]

        # ------- Terrain surface -------
        if data.get("terrain_ok"):
            verts = data["terrain_vertices"]
            # Extract unique vertices and faces for plotly Mesh3d
            # Flatten triangles
            i_vals, j_vals, k_vals = [], [], []
            all_x, all_y, all_z = [], [], []
            for tri_idx in range(0, len(verts), max(1, len(verts) // 5000)):
                tri = verts[tri_idx]
                base_idx = len(all_x)
                for v in tri:
                    all_x.append(v[0])
                    all_y.append(v[1])
                    all_z.append(v[2] * zex)
                i_vals.append(base_idx)
                j_vals.append(base_idx + 1)
                k_vals.append(base_idx + 2)

            all_z_unscaled = [zz / zex for zz in all_z]
            fig.add_trace(go.Mesh3d(
                x=all_x, y=all_y, z=all_z,
                i=i_vals, j=j_vals, k=k_vals,
                intensity=all_z_unscaled,
                colorscale="Earth",
                colorbar=dict(title="Elevation [m]", x=0.95, len=0.5),
                opacity=0.95,
                lighting=dict(ambient=0.5, diffuse=0.6, specular=0.3),
                name="Terrain",
                showlegend=True,
            ))

        # ------- Streamlines (subset of CFD cells) -------
        if show_streamlines.value and data.get("fields_ok"):
            cx = 0.5 * (x.min() + x.max())
            cy = 0.5 * (y.min() + y.max())
            # Filter cells near ground (z < 800m) and in central region
            half = 8000
            mask = (
                (np.abs(x - cx) < half) &
                (np.abs(y - cy) < half) &
                (z < 1000)
            )
            xi, yi, zi = x[mask], y[mask], z[mask]
            ui, vi, wi = U[mask, 0], U[mask, 1], U[mask, 2]
            si = speed[mask]

            # Simple forward Euler streamline integration
            n_sl = int(n_streamlines.value)
            sl_x0 = np.full(n_sl, cx - half * 0.8)
            sl_y0 = np.linspace(cy - half * 0.6, cy + half * 0.6, n_sl)
            sl_z0 = np.full(n_sl, 200)

            from scipy.interpolate import NearestNDInterpolator
            pts_3d = np.column_stack([xi, yi, zi])
            interp_u = NearestNDInterpolator(pts_3d, ui)
            interp_v = NearestNDInterpolator(pts_3d, vi)
            interp_w = NearestNDInterpolator(pts_3d, wi)
            interp_s = NearestNDInterpolator(pts_3d, si)

            dt = 50  # step size [m / (m/s) ≈ seconds]
            n_steps = 200

            for sl_idx in range(n_sl):
                px, py, pz = [sl_x0[sl_idx]], [sl_y0[sl_idx]], [sl_z0[sl_idx]]
                colors = []
                for _ in range(n_steps):
                    pt = np.array([[px[-1], py[-1], pz[-1]]])
                    uu = float(interp_u(pt))
                    vv = float(interp_v(pt))
                    ww = float(interp_w(pt))
                    ss = float(interp_s(pt))
                    if ss < 0.3:
                        break
                    px.append(px[-1] + uu * dt)
                    py.append(py[-1] + vv * dt)
                    pz.append(pz[-1] + ww * dt)
                    colors.append(ss)

                if len(px) > 2:
                    fig.add_trace(go.Scatter3d(
                        x=px, y=py, z=[zz * zex for zz in pz],
                        mode="lines",
                        line=dict(
                            color=colors if colors else "blue",
                            colorscale="RdBu_r",
                            cmin=0, cmax=12,
                            width=3,
                        ),
                        showlegend=(sl_idx == 0),
                        name="Streamlines" if sl_idx == 0 else None,
                        hoverinfo="skip",
                    ))

        # ------- Tower markers -------
        if show_towers.value and data.get("towers_ok"):
            # Key towers only (tse04, tse09, tse13)
            key = ["tse04", "tse09", "tse13"]
            for t in data["towers"]:
                if t["id"] in key:
                    fig.add_trace(go.Scatter3d(
                        x=[t["x"]], y=[t["y"]],
                        z=[(t["alt"] + t["max_h"] + 20) * zex],
                        mode="markers+text",
                        marker=dict(size=6, color="red", symbol="diamond"),
                        text=[t["id"]],
                        textposition="top center",
                        textfont=dict(size=12, color="black"),
                        showlegend=False,
                    ))
                    # Tower pole
                    fig.add_trace(go.Scatter3d(
                        x=[t["x"], t["x"]],
                        y=[t["y"], t["y"]],
                        z=[t["alt"] * zex, (t["alt"] + t["max_h"]) * zex],
                        mode="lines",
                        line=dict(color="red", width=3),
                        showlegend=False,
                        hoverinfo="skip",
                    ))

        # ------- Layout -------
        cx = 0.5 * (x.min() + x.max()) if data.get("fields_ok") else 0
        cy = 0.5 * (y.min() + y.max()) if data.get("fields_ok") else 0
        fig.update_layout(
            title=dict(
                text=f"Perdigão CFD — Δx = {res_label} — 2017-05-11 12:00 UTC, SSW 217°",
                font=dict(size=16),
            ),
            scene=dict(
                xaxis=dict(title="Easting [m]", showgrid=True, gridcolor="rgba(200,200,200,0.3)"),
                yaxis=dict(title="Northing [m]", showgrid=True, gridcolor="rgba(200,200,200,0.3)"),
                zaxis=dict(title="Altitude [m]", showgrid=True, gridcolor="rgba(200,200,200,0.3)"),
                bgcolor="#e8f0fe",
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=0.15 * zex),
                camera=dict(
                    eye=dict(x=-1.5, y=-1.0, z=0.6),
                    center=dict(x=0, y=0, z=-0.1),
                ),
            ),
            width=1200,
            height=800,
            margin=dict(l=0, r=0, t=50, b=0),
            legend=dict(x=0.01, y=0.99),
        )

        fig
    return


if __name__ == "__main__":
    app.run()
