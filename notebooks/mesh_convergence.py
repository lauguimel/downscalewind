import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Mesh Convergence Study — Phase 0.2

    Resolution sweep: **500 m / 250 m / 100 m** — box domain, Perdigão, neutral BBSF

    - Solver: `buoyantBoussinesqSimpleFoam`, k-ε, cfMesh cartesianMesh
    - Inflow: 10 m/s @ 231°, neutral (Ri_b=0), ERA5 parametric
    - Domain: 10×10 km, 3000 m height
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import re
    import subprocess
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    return Path, mcolors, mo, np, pd, plt, re, subprocess


@app.cell(hide_code=True)
def _(mo):
    mo.md("## 1 · Download from HPC (rsync)")
    return


@app.cell
def _(mo):
    download_btn = mo.ui.run_button(label="Download from Aqua HPC")
    download_btn
    return (download_btn,)


@app.cell
def _(Path, download_btn, mo, subprocess):
    local_base = Path("../data/cases/phase0_resolution")
    remote_base = "maitreje@aqua.qut.edu.au:~/downscalewind/campaign"
    cases = ["case_res_500", "case_res_250", "case_res_100"]

    download_status = ""
    if download_btn.value:
        msgs = []
        for _c in cases:
            _local = local_base / _c
            _local.mkdir(parents=True, exist_ok=True)
            _result = subprocess.run(
                [
                    "rsync", "-az", "--progress",
                    "--include=log.*",
                    "--include=0/",
                    "--include=0/Cx", "--include=0/Cy", "--include=0/Cz",
                    "--include=postProcessing/",
                    "--include=postProcessing/**",
                    "--include=system/controlDict",
                    "--exclude=processor*/",
                    "--exclude=*",
                    f"{remote_base}/{_c}/",
                    str(_local) + "/",
                ],
                capture_output=True, text=True,
            )
            _status = "✓" if _result.returncode == 0 else f"✗ {_result.stderr[:80]}"
            msgs.append(f"{_c}: {_status}")
        download_status = "\n".join(msgs)
    else:
        # Just check what's available locally
        for _c in cases:
            _log = local_base / _c / "log.buoyantBoussinesqSimpleFoam"
            download_status += f"{_c}: {'✓ local' if _log.exists() else '⚠ not downloaded'}\n"

    mo.callout(mo.md(download_status), kind="info")
    return cases, download_status, local_base, remote_base


@app.cell(hide_code=True)
def _(mo):
    mo.md("## 2 · Mesh statistics")
    return


@app.cell
def _(Path, cases, local_base, pd, re):
    def parse_checkmesh(case_dir: Path) -> dict:
        log = case_dir / "log.checkMesh"
        if not log.exists():
            return {}
        txt = log.read_text()
        result = {}
        m = re.search(r"cells:\s+(\d+)", txt)
        if m:
            result["n_cells"] = int(m.group(1))
        m = re.search(r"points:\s+(\d+)", txt)
        if m:
            result["n_points"] = int(m.group(1))
        m = re.search(r"faces:\s+(\d+)", txt)
        if m:
            result["n_faces"] = int(m.group(1))
        # Max non-orthogonality
        m = re.search(r"Max non-orthogonality = ([\d.]+)", txt)
        if m:
            result["max_nonortho"] = float(m.group(1))
        # Max skewness
        m = re.search(r"Max skewness = ([\d.]+)", txt)
        if m:
            result["max_skewness"] = float(m.group(1))
        return result

    mesh_stats = {}
    res_labels = {"case_res_500": "500 m", "case_res_250": "250 m", "case_res_100": "100 m"}
    for _c in cases:
        _stats = parse_checkmesh(local_base / _c)
        if _stats:
            mesh_stats[res_labels[_c]] = _stats

    df_mesh = pd.DataFrame(mesh_stats).T
    df_mesh.index.name = "Resolution"
    df_mesh
    return df_mesh, mesh_stats, parse_checkmesh, res_labels


@app.cell
def _(df_mesh, mo, plt):
    if df_mesh.empty or "n_cells" not in df_mesh.columns:
        mo.callout(mo.md("⚠ No checkMesh data found — download first"), kind="warn")
    else:
        _fig, _axes = plt.subplots(1, 3, figsize=(12, 4))
        _colors = ["#3498db", "#2ecc71", "#e74c3c"]

        for _ax, _col, _label in zip(
            _axes,
            ["n_cells", "max_nonortho", "max_skewness"],
            ["Number of cells", "Max non-orthogonality (°)", "Max skewness"],
        ):
            if _col not in df_mesh.columns:
                continue
            _vals = df_mesh[_col].dropna()
            _ax.bar(_vals.index, _vals.values, color=_colors[: len(_vals)])
            _ax.set_title(_label)
            _ax.set_xlabel("Resolution")
            for _i, (_x, _v) in enumerate(zip(_vals.index, _vals.values)):
                _ax.text(_i, _v * 1.02, f"{_v:,.0f}", ha="center", fontsize=9)

        _fig.suptitle("Mesh quality — Phase 0.2 resolution sweep", fontweight="bold")
        _fig.tight_layout()
        _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("## 3 · Solver residuals")
    return


@app.cell
def _(Path, cases, local_base, np, pd, re, res_labels):
    def parse_solver_log(case_dir: Path) -> pd.DataFrame:
        """Parse buoyantBoussinesqSimpleFoam log into a DataFrame."""
        log = case_dir / "log.buoyantBoussinesqSimpleFoam"
        if not log.exists():
            return pd.DataFrame()
        txt = log.read_text()

        records = []
        current_time = None
        # Match "Time = N"
        time_pat = re.compile(r"^Time = (\d+)", re.MULTILINE)
        # Match residual lines
        resid_pat = re.compile(
            r"Solving for (\w+), Initial residual = ([\d.e+-]+),"
        )
        exec_pat = re.compile(r"ExecutionTime = ([\d.]+) s")

        row = {}
        for line in txt.splitlines():
            m = time_pat.match(line)
            if m:
                if row and current_time is not None:
                    row["time"] = current_time
                    records.append(row)
                current_time = int(m.group(1))
                row = {}
                continue
            m = resid_pat.search(line)
            if m:
                var, val = m.group(1), float(m.group(2))
                # Keep first residual for each variable (Ux/Uy/Uz → U_max)
                key = f"r_{var}"
                if key not in row or float(val) > row[key]:
                    row[key] = float(val)
            m = exec_pat.search(line)
            if m:
                row["exec_time_s"] = float(m.group(1))

        if row and current_time is not None:
            row["time"] = current_time
            records.append(row)

        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records).set_index("time")
        # Aggregate U components into scalar
        u_cols = [c for c in df.columns if c in ("r_Ux", "r_Uy", "r_Uz")]
        if u_cols:
            df["r_U"] = df[u_cols].max(axis=1)
            df.drop(columns=u_cols, inplace=True, errors="ignore")
        return df

    resid_data = {}
    for _c in cases:
        _df = parse_solver_log(local_base / _c)
        if not _df.empty:
            resid_data[res_labels[_c]] = _df

    resid_data
    return parse_solver_log, resid_data


@app.cell
def _(mo, plt, resid_data):
    if not resid_data:
        mo.callout(mo.md("⚠ No solver logs found — download first"), kind="warn")
    else:
        _colors = {"500 m": "#3498db", "250 m": "#2ecc71", "100 m": "#e74c3c"}
        _vars = ["r_U", "r_p_rgh", "r_T", "r_k", "r_epsilon"]
        _labels = {"r_U": "U", "r_p_rgh": "p_rgh", "r_T": "T", "r_k": "k", "r_epsilon": "ε"}

        # Find which variables exist
        _avail = [v for v in _vars if any(v in df.columns for df in resid_data.values())]
        _n = len(_avail)

        _fig, _axes = plt.subplots(1, _n, figsize=(4 * _n, 4), sharey=False)
        if _n == 1:
            _axes = [_axes]

        for _ax, _var in zip(_axes, _avail):
            for _res, _df in resid_data.items():
                if _var in _df.columns:
                    _ax.semilogy(
                        _df.index, _df[_var],
                        label=_res, color=_colors.get(_res, "gray"),
                        linewidth=1.2, alpha=0.85,
                    )
            _ax.set_title(_labels.get(_var, _var))
            _ax.set_xlabel("Iteration")
            _ax.set_ylabel("Initial residual" if _var == _avail[0] else "")
            _ax.axhline(1e-4, color="k", lw=0.8, ls="--", alpha=0.4, label="10⁻⁴")
            _ax.legend(fontsize=8)
            _ax.grid(True, which="both", alpha=0.3)

        _fig.suptitle("Solver convergence — residuals per iteration", fontweight="bold")
        _fig.tight_layout()
        _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("## 4 · Global field monitoring (fieldMinMax + volAverage)")
    return


@app.cell
def _(Path, cases, local_base, pd, re, res_labels):
    def parse_field_minmax(case_dir: Path) -> pd.DataFrame:
        """Parse postProcessing/fieldMinMax/0/fieldMinMax.dat."""
        pp = case_dir / "postProcessing" / "fieldMinMax" / "0"
        if not pp.exists():
            # Try alternative naming
            for _p in (case_dir / "postProcessing" / "fieldMinMax").glob("*/fieldMinMax.dat"):
                pp = _p.parent
                break
        dat = list(pp.glob("*.dat")) if pp.exists() else []
        if not dat:
            return pd.DataFrame()
        df = pd.read_csv(dat[0], comment="#", sep=r"\s+", header=None)
        # Typical columns: time field min max minLoc maxLoc
        df.columns = range(df.shape[1])
        return df

    def parse_vol_average(case_dir: Path) -> pd.DataFrame:
        """Parse postProcessing/volAverages/0/*.dat."""
        pp = case_dir / "postProcessing" / "volAverages"
        if not pp.exists():
            return pd.DataFrame()
        frames = []
        for dat in pp.glob("**/*.dat"):
            try:
                _df = pd.read_csv(dat, comment="#", sep=r"\s+", header=None)
                _df["field"] = dat.stem
                frames.append(_df)
            except Exception:
                pass
        return pd.concat(frames) if frames else pd.DataFrame()

    minmax_data = {}
    volavg_data = {}
    for _c in cases:
        _mm = parse_field_minmax(local_base / _c)
        if not _mm.empty:
            minmax_data[res_labels[_c]] = _mm
        _va = parse_vol_average(local_base / _c)
        if not _va.empty:
            volavg_data[res_labels[_c]] = _va

    _note = "ℹ fieldMinMax / volAverages available only for cases run with the updated controlDict (function objects added). Re-run cases to get these metrics."
    minmax_data if minmax_data else _note
    return (
        minmax_data,
        parse_field_minmax,
        parse_vol_average,
        volavg_data,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("## 5 · Mesh visualisation — cell centres cross-section")
    return


@app.cell
def _(mo):
    _res_options = {"500 m": "case_res_500", "250 m": "case_res_250", "100 m": "case_res_100"}
    viz_res = mo.ui.dropdown(
        options=_res_options,
        value="case_res_500",
        label="Resolution",
    )
    viz_plane = mo.ui.dropdown(
        options={"Y-Z (cross-valley)": "yz", "X-Z (along-wind)": "xz", "X-Y (plan view)": "xy"},
        value="xz",
        label="Cross-section plane",
    )
    viz_slice = mo.ui.slider(
        start=-4000, stop=4000, step=100, value=0,
        label="Slice position (m)",
    )
    mo.hstack([viz_res, viz_plane, viz_slice])
    return viz_plane, viz_res, viz_slice


@app.cell
def _(Path, local_base, mo, mcolors, np, plt, viz_plane, viz_res, viz_slice):
    def _read_of_scalar(path: Path) -> np.ndarray:
        """Read an OpenFOAM scalar field (internalField) from file."""
        txt = path.read_text()
        m_n = __import__("re").search(r"internalField\s+nonuniform List<scalar>\s*(\d+)\s*\((.+?)\)", txt, __import__("re").DOTALL)
        if m_n:
            return np.fromstring(m_n.group(2), sep="\n")
        m_u = __import__("re").search(r"internalField\s+uniform\s+([\d.e+-]+)", txt)
        if m_u:
            return np.array([float(m_u.group(1))])
        return np.array([])

    _case_dir = local_base / viz_res.value
    _cx_f = _case_dir / "0" / "Cx"
    _cy_f = _case_dir / "0" / "Cy"
    _cz_f = _case_dir / "0" / "Cz"

    if not (_cx_f.exists() and _cy_f.exists() and _cz_f.exists()):
        mo.callout(mo.md("⚠ Cell centres (0/Cx, 0/Cy, 0/Cz) not found — download first"), kind="warn")
    else:
        _cx = _read_of_scalar(_cx_f)
        _cy = _read_of_scalar(_cy_f)
        _cz = _read_of_scalar(_cz_f)

        _plane = viz_plane.value
        _pos = viz_slice.value
        _tol = 300.0  # slice thickness (m)

        if _plane == "xz":
            _mask = np.abs(_cy - _pos) < _tol
            _x_ax, _y_ax, _xlabel, _ylabel = _cx[_mask], _cz[_mask], "X (m)", "Z (m)"
            _title = f"X-Z slice at Y={_pos} m ±{_tol} m"
        elif _plane == "yz":
            _mask = np.abs(_cx - _pos) < _tol
            _x_ax, _y_ax, _xlabel, _ylabel = _cy[_mask], _cz[_mask], "Y (m)", "Z (m)"
            _title = f"Y-Z slice at X={_pos} m ±{_tol} m"
        else:
            _mask = np.abs(_cz - _pos) < _tol
            _x_ax, _y_ax, _xlabel, _ylabel = _cx[_mask], _cy[_mask], "X (m)", "Y (m)"
            _title = f"X-Y slice at Z={_pos} m ±{_tol} m"

        _fig, _ax = plt.subplots(figsize=(10, 5))
        if len(_x_ax) > 0:
            _sc = _ax.scatter(
                _x_ax, _y_ax,
                c=_cz[_mask] if _plane != "xy" else _cx[_mask],
                cmap="terrain", s=1.5, alpha=0.6,
                norm=mcolors.Normalize(vmin=_cz.min(), vmax=_cz.max()) if _plane != "xy" else None,
            )
            plt.colorbar(_sc, ax=_ax, label="Z elevation (m)" if _plane != "xy" else "X (m)")
            _ax.set_title(f"{_title}\n{len(_x_ax):,} cells shown", fontweight="bold")
        else:
            _ax.text(0.5, 0.5, "No cells in slice", ha="center", va="center", transform=_ax.transAxes)
        _ax.set_xlabel(_xlabel)
        _ax.set_ylabel(_ylabel)
        _ax.set_aspect("equal")
        _ax.grid(True, alpha=0.2)
        _fig.tight_layout()
        _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("## 6 · Execution time & cost")
    return


@app.cell
def _(cases, local_base, mo, pd, re, res_labels):
    _timing = {}
    for _c in cases:
        _log = local_base / _c / "log.buoyantBoussinesqSimpleFoam"
        if not _log.exists():
            continue
        _txt = _log.read_text()
        # Last ExecutionTime
        _matches = re.findall(r"ExecutionTime = ([\d.]+) s\s+ClockTime = ([\d.]+) s", _txt)
        if _matches:
            _exec, _clock = float(_matches[-1][0]), float(_matches[-1][1])
            # Last Time =
            _t_matches = re.findall(r"^Time = (\d+)", _txt, re.MULTILINE)
            _last_iter = int(_t_matches[-1]) if _t_matches else None
            _timing[res_labels[_c]] = {
                "last_iter": _last_iter,
                "cpu_time_s": _exec,
                "wall_time_s": _clock,
                "s_per_iter_cpu": round(_exec / _last_iter, 2) if _last_iter else None,
                "s_per_iter_wall": round(_clock / _last_iter, 2) if _last_iter else None,
            }

    if _timing:
        pd.DataFrame(_timing).T
    else:
        mo.callout(mo.md("⚠ No timing data yet"), kind="warn")
    return


if __name__ == "__main__":
    app.run()
