import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import zarr
    import plotly.graph_objects as go
    from pathlib import Path
    from stl import mesh as stlmesh

    return Path, go, mo, np, stlmesh, zarr


@app.cell
def _(mo):
    mo.md("""
    # ERA5 Boundary Conditions — 3D Wireframe

    Grille ERA5 (25 km) x 10 niveaux de pression en 3D,
    avec le terrain STL Perdigao et le domaine CFD 10x10 km.
    """)
    return


@app.cell
def _(mo):
    timestamp_input = mo.ui.text(value="2017-05-04T22:00", label="Timestamp ERA5")
    var_selector = mo.ui.dropdown(
        options={"U (eastward) [m/s]": "u", "V (northward) [m/s]": "v",
                 "Temperature [K]": "t", "Wind speed [m/s]": "speed"},
        value="U (eastward) [m/s]",
        label="Variable",
    )
    mo.hstack([timestamp_input, var_selector])
    return timestamp_input, var_selector


@app.cell
def _(Path, go, np, stlmesh, timestamp_input, var_selector, zarr):
    # ---- Load ERA5 ----
    _zarr_path = str(Path(__file__).resolve().parent.parent / "data" / "raw" / "era5_perdigao.zarr")
    _store = zarr.open_group(_zarr_path, mode="r")

    _times_raw = np.array(_store["coords/time"][:])
    _times = _times_raw.astype("datetime64[ns]").astype("datetime64[s]")
    _levels = np.array(_store["coords/level"][:], dtype=float)
    _lats = np.array(_store["coords/lat"][:], dtype=float)
    _lons = np.array(_store["coords/lon"][:], dtype=float)

    _u_all = np.array(_store["pressure/u"][:], dtype=float)
    _v_all = np.array(_store["pressure/v"][:], dtype=float)
    _t_all = np.array(_store["pressure/t"][:], dtype=float)
    _z_all = np.array(_store["pressure/z"][:], dtype=float) / 9.81

    # Sort lats ascending (ERA5 convention: lat[0]=North, descending)
    _lat_order = np.argsort(_lats)
    _lats_asc = _lats[_lat_order]
    # Reorder data arrays to ascending lat
    _u_all = _u_all[:, :, _lat_order, :]
    _v_all = _v_all[:, :, _lat_order, :]
    _t_all = _t_all[:, :, _lat_order, :]
    _z_all = _z_all[:, :, _lat_order, :]

    # ---- Time index ----
    _ts = np.datetime64(timestamp_input.value, "s")
    _t_idx = int(np.argmin(np.abs(_times - _ts)))
    _actual_time = str(_times[_t_idx])

    # ---- Extract 3D cubes [level, lat_asc, lon] ----
    _n_lev = len(_levels)
    _n_lat = len(_lats_asc)
    _n_lon = len(_lons)

    _u3 = np.zeros((_n_lev, _n_lat, _n_lon))
    _v3 = np.zeros_like(_u3)
    _t3 = np.zeros_like(_u3)
    _z3 = np.zeros_like(_u3)

    for _li in range(_n_lev):
        _u3[_li] = _u_all[_t_idx, _li, :, :]
        _v3[_li] = _v_all[_t_idx, _li, :, :]
        _t3[_li] = _t_all[_t_idx, _li, :, :]
        _z3[_li] = _z_all[_t_idx, _li, :, :]

    _spd3 = np.sqrt(_u3**2 + _v3**2)

    # ---- Local coordinates ----
    _SITE_LAT, _SITE_LON = 39.716, -7.740
    _DEG_TO_M_LAT = 111_000.0
    _DEG_TO_M_LON = 111_000.0 * np.cos(np.radians(_SITE_LAT))

    _era5_x = (_lons - _SITE_LON) * _DEG_TO_M_LON
    _era5_y = (_lats_asc - _SITE_LAT) * _DEG_TO_M_LAT

    _half = 5000.0  # 10 km / 2

    # ---- Select variable ----
    _var_key = var_selector.value
    _field3 = {"u": _u3, "v": _v3, "t": _t3, "speed": _spd3}[_var_key]
    _var_label = {"u": "U [m/s]", "v": "V [m/s]", "t": "T [K]", "speed": "Speed [m/s]"}[_var_key]
    _cmap = {"u": "RdBu_r", "v": "RdBu_r", "t": "Inferno", "speed": "Viridis"}[_var_key]
    _vmin, _vmax = float(_field3.min()), float(_field3.max())

    # ---- Load terrain STL ----
    _stl_path = str(Path(__file__).resolve().parent.parent /
                     "data" / "cases" / "sf500m_venkatraman" / "constant" / "triSurface" / "terrain.stl")
    _terrain = stlmesh.Mesh.from_file(_stl_path)
    _tri = _terrain.vectors
    _stl_x = _tri[:, :, 0].flatten()
    _stl_y = _tri[:, :, 1].flatten()
    _stl_z = _tri[:, :, 2].flatten()
    _n_tri = len(_tri)
    _tri_i = np.arange(0, 3 * _n_tri, 3)
    _tri_j = np.arange(1, 3 * _n_tri, 3)
    _tri_k = np.arange(2, 3 * _n_tri, 3)

    # ==== BUILD FIGURE ====
    fig = go.Figure()

    # Terrain surface
    fig.add_trace(go.Mesh3d(
        x=_stl_x, y=_stl_y, z=_stl_z,
        i=_tri_i, j=_tri_j, k=_tri_k,
        color="sienna", opacity=0.6,
        name="Terrain", hoverinfo="skip",
    ))

    # ERA5 scatter points at each (lat, lon, level)
    _all_x, _all_y, _all_z, _all_c, _all_hover = [], [], [], [], []
    for _li, _lev in enumerate(_levels):
        _zl = _z3[_li]
        _fl = _field3[_li]
        for _i in range(_n_lat):
            for _j in range(_n_lon):
                _all_x.append(_era5_x[_j])
                _all_y.append(_era5_y[_i])
                _all_z.append(_zl[_i, _j])
                _all_c.append(_fl[_i, _j])
                _all_hover.append(
                    f"{_lev:.0f} hPa<br>"
                    f"x={_era5_x[_j]:.0f}m y={_era5_y[_i]:.0f}m z={_zl[_i,_j]:.0f}m<br>"
                    f"{_var_label}={_fl[_i,_j]:.2f}"
                )

    fig.add_trace(go.Scatter3d(
        x=_all_x, y=_all_y, z=_all_z,
        mode="markers",
        marker=dict(
            size=5, color=_all_c, colorscale=_cmap,
            cmin=_vmin, cmax=_vmax, showscale=True,
            colorbar=dict(title=_var_label, x=1.02),
        ),
        hovertext=_all_hover, hoverinfo="text",
        showlegend=True, name="ERA5 grid points",
    ))

    # ---- Trilinear interpolation helper ----
    # At each ERA5 column (i,j), vertical profile: z3[:,i,j] → field3[:,i,j]
    # 1) Vertical interp at each column to z_target
    # 2) Bilinear interp in (x,y) from 4 surrounding columns
    from scipy.interpolate import interp1d

    def _interp_column(_i, _j, _z_target):
        """Interpolate field vertically at ERA5 column (i,j) to z_target."""
        _zc = _z3[:, _i, _j]
        _fc = _field3[:, _i, _j]
        _order = np.argsort(_zc)
        _fn = interp1d(_zc[_order], _fc[_order], kind="linear",
                       bounds_error=False, fill_value=(_fc[_order[0]], _fc[_order[-1]]))
        return _fn(_z_target)

    def _trilinear(_xq, _yq, _zq):
        """Trilinear interpolation: bilinear(x,y) x vertical(z) from ERA5 grid."""
        _shape = _xq.shape
        _xf, _yf, _zf = _xq.ravel(), _yq.ravel(), _zq.ravel()
        _n = len(_xf)
        # Find bounding ERA5 indices in x and y
        _ix = np.searchsorted(_era5_x, _xf).clip(1, _n_lon - 1)
        _iy = np.searchsorted(_era5_y, _yf).clip(1, _n_lat - 1)
        # Bilinear weights
        _dx = _era5_x[_ix] - _era5_x[_ix - 1]
        _dy = _era5_y[_iy] - _era5_y[_iy - 1]
        _wx = np.where(_dx != 0, (_xf - _era5_x[_ix - 1]) / _dx, 0.5).clip(0, 1)
        _wy = np.where(_dy != 0, (_yf - _era5_y[_iy - 1]) / _dy, 0.5).clip(0, 1)
        # Pre-compute column interps for all needed (lon_idx, lat_idx) pairs
        _cache = {}
        for _di in [0, 1]:
            for _dj in [0, 1]:
                for _ci, _cj in set(zip(_ix - 1 + _di, _iy - 1 + _dj)):
                    if (_ci, _cj) not in _cache:
                        # _interp_column(lat_idx, lon_idx, z) — note: i=lat, j=lon
                        _cache[(_ci, _cj)] = _interp_column(_cj, _ci, _zf)
        # Gather values at each query point from 4 corners
        _f00 = np.array([_cache[(_ix[_k] - 1, _iy[_k] - 1)][_k] for _k in range(_n)])
        _f10 = np.array([_cache[(_ix[_k],     _iy[_k] - 1)][_k] for _k in range(_n)])
        _f01 = np.array([_cache[(_ix[_k] - 1, _iy[_k])    ][_k] for _k in range(_n)])
        _f11 = np.array([_cache[(_ix[_k],     _iy[_k])    ][_k] for _k in range(_n)])
        _result = (_f00 * (1 - _wx) * (1 - _wy) +
                   _f10 * _wx * (1 - _wy) +
                   _f01 * (1 - _wx) * _wy +
                   _f11 * _wx * _wy)
        return _result.reshape(_shape)

    # ---- CFD box faces with interpolated BC texture ----
    _h = _half
    _zt = 3500.0
    _nf = 20  # grid resolution per face

    # Each face: build (x, y, z) grids for query + display
    _faces = []
    # West (x=-h): varies in y and z
    _gy, _gz = np.meshgrid(np.linspace(-_h, _h, _nf), np.linspace(0, _zt, _nf))
    _faces.append(("West",  np.full_like(_gy, -_h), _gy, _gz))
    # East (x=+h)
    _faces.append(("East",  np.full_like(_gy,  _h), _gy, _gz))
    # South (y=-h): varies in x and z
    _gx, _gz2 = np.meshgrid(np.linspace(-_h, _h, _nf), np.linspace(0, _zt, _nf))
    _faces.append(("South", _gx, np.full_like(_gx, -_h), _gz2))
    # North (y=+h)
    _faces.append(("North", _gx, np.full_like(_gx,  _h), _gz2))
    # Top (z=zt): varies in x and y
    _gx3, _gy3 = np.meshgrid(np.linspace(-_h, _h, _nf), np.linspace(-_h, _h, _nf))
    _faces.append(("Top",   _gx3, _gy3, np.full_like(_gx3, _zt)))

    for _fname, _sx, _sy, _sz in _faces:
        _fc = _trilinear(_sx, _sy, _sz)
        fig.add_trace(go.Surface(
            x=_sx, y=_sy, z=_sz,
            surfacecolor=_fc,
            colorscale=_cmap, cmin=_vmin, cmax=_vmax,
            showscale=False, opacity=0.7,
            name=f"BC {_fname}",
            hovertemplate=(
                f"{_fname}<br>"
                f"x=%{{x:.0f}}m y=%{{y:.0f}}m z=%{{z:.0f}}m<br>"
                f"{_var_label}=%{{surfacecolor:.2f}}<extra></extra>"
            ),
        ))

    # Layout
    fig.update_layout(
        title=f"ERA5 3D + Terrain — {_var_label} — {_actual_time}",
        scene=dict(
            xaxis_title="X [m] (East)",
            yaxis_title="Y [m] (North)",
            zaxis_title="Z [m ASL]",
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.3),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8)),
        ),
        height=800, width=1100,
        template="plotly_white",
    )
    return (fig,)


@app.cell
def _(fig, mo):
    mo.ui.plotly(fig)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    **Legende** :
    - **Surface brune** = terrain STL Perdigao (z = 141-468 m ASL)
    - **Points colores** = ERA5 grid points (3x3 x 10 niveaux = 90 points)
    - **Boite rouge** = domaine CFD 10x10 km x 3.5 km
    - Le 1000 hPa (~50 m ASL) est **sous le terrain** — physiquement correct
      (niveau de pression extrapole sous la surface)
    """)
    return


if __name__ == "__main__":
    app.run()
