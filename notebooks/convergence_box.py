import marimo

__generated_with = "0.11.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import re
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    CASES_DIR = Path("../data/cases/phase0_resolution")
    RESOLUTIONS = {"case_res_500": 500, "case_res_250": 250, "case_res_100": 100}
    COLORS = {500: "#e07b54", 250: "#5b8db8", 100: "#4caf7d"}
    FIG_DIR = Path("../data/validation/convergence_figures")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    mo.md("# Convergence study — Box domain (500 / 250 / 100 m)")
    return CASES_DIR, COLORS, FIG_DIR, RESOLUTIONS, mo, np, pd, plt, re, Path


@app.cell
def _(CASES_DIR, RESOLUTIONS, pd, re):
    def _parse_checkmesh(case_name):
        _log = (CASES_DIR / case_name / "log.checkMesh").read_text()
        _r = {}
        _m = re.search(r"cells:\s+(\d+)", _log)
        _r["n_cells"] = int(_m.group(1)) if _m else None
        _m = re.search(r"Max non-orthogonality\s*=\s*([\d.]+)", _log)
        _r["max_nonorth"] = float(_m.group(1)) if _m else None
        _m = re.search(r"average non-orthogonality\s*=\s*([\d.]+)", _log)
        _r["avg_nonorth"] = float(_m.group(1)) if _m else None
        _m = re.search(r"Max skewness\s*=\s*([\d.]+)", _log)
        _r["max_skew"] = float(_m.group(1)) if _m else None
        _m = re.search(r"Max aspect ratio\s*=\s*([\d.]+)", _log)
        _r["max_aspect"] = float(_m.group(1)) if _m else None
        _r["mesh_ok"] = bool(re.search(r"Mesh OK\.", _log))
        return _r

    mesh_stats = {}
    for _cname, _res in RESOLUTIONS.items():
        try:
            mesh_stats[_res] = _parse_checkmesh(_cname)
        except FileNotFoundError:
            mesh_stats[_res] = {}

    df_mesh = pd.DataFrame(mesh_stats).T
    df_mesh.index.name = "resolution_m"
    df_mesh
    return df_mesh, mesh_stats


@app.cell
def _(CASES_DIR, RESOLUTIONS, pd, re):
    def _parse_solver(case_name):
        _log = (CASES_DIR / case_name / "log.buoyantBoussinesqSimpleFoam").read_text()
        _iters, _ux, _uy, _uz, _p, _T, _k, _eps = [], [], [], [], [], [], [], []
        _blocks = re.split(r"\nTime = (\d+)\n", _log)
        for _i in range(1, len(_blocks) - 1, 2):
            _t = int(_blocks[_i])
            _blk = _blocks[_i + 1]

            def _r(f, b=_blk):
                _mm = re.search(rf"Solving for {f},\s*Initial residual = ([\deE+\-.]+)", b)
                return float(_mm.group(1)) if _mm else float("nan")

            _iters.append(_t)
            _ux.append(_r("Ux")); _uy.append(_r("Uy")); _uz.append(_r("Uz"))
            _p.append(_r("p_rgh")); _T.append(_r("T"))
            _k.append(_r("k")); _eps.append(_r("epsilon"))

        return pd.DataFrame({
            "iter": _iters, "Ux": _ux, "Uy": _uy, "Uz": _uz,
            "p_rgh": _p, "T": _T, "k": _k, "epsilon": _eps,
        }).set_index("iter")

    solver_logs = {}
    for _cname, _res in RESOLUTIONS.items():
        try:
            solver_logs[_res] = _parse_solver(_cname)
            _df = solver_logs[_res]
            print(f"{_res}m: {len(_df)} iters, final Ux={_df['Ux'].dropna().iloc[-1]:.2e}, "
                  f"k={_df['k'].dropna().iloc[-1]:.2e}")
        except FileNotFoundError:
            print(f"{_res}m: log not found")

    solver_logs
    return (solver_logs,)


@app.cell
def _(COLORS, FIG_DIR, plt, solver_logs):
    _fig, _axes = plt.subplots(2, 2, figsize=(11, 7), sharex=False)
    for _ax, (_field, _label) in zip(
        _axes.flat,
        [("Ux", "U_x"), ("p_rgh", "p_rgh"), ("k", "k"), ("T", "T")]
    ):
        for _res, _df in sorted(solver_logs.items()):
            if _field in _df.columns:
                _v = _df[_field].dropna()
                _ax.semilogy(_v.index, _v.values,
                             color=COLORS[_res], linewidth=1.2,
                             label=f"{_res}m ({len(_v)} iter)")
        _ax.axhline(1e-4, color="k", linestyle="--", lw=0.8, alpha=0.5, label="1e-4")
        _ax.set_ylabel(f"Initial residual {_label}")
        _ax.set_xlabel("Iteration")
        _ax.legend(fontsize=8)
        _ax.grid(True, alpha=0.3)
    _fig.suptitle("Residual convergence — box domain", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _fig.savefig(FIG_DIR / "convergence_box_residuals.png", dpi=150, bbox_inches="tight")
    _fig
    return


@app.cell
def _(COLORS, FIG_DIR, mesh_stats, np, plt):
    _fig3, _axes3 = plt.subplots(1, 3, figsize=(11, 4))
    _metrics = [
        ("n_cells", "Cell count", None),
        ("max_nonorth", "Max non-orthogonality (°)\n[threshold: 70°]", 70),
        ("max_skew", "Max skewness\n[threshold: 4]", 4),
    ]
    for _ax, (_metric, _ylabel, _thresh) in zip(_axes3, _metrics):
        _ress = sorted(mesh_stats.keys(), reverse=True)
        _vals = [mesh_stats[_r].get(_metric, np.nan) for _r in _ress]
        _bars = _ax.bar(
            [f"{_r}m" for _r in _ress], _vals,
            color=[COLORS[_r] for _r in _ress], alpha=0.85
        )
        if _thresh is not None:
            _ax.axhline(_thresh, color="r", linestyle="--", lw=1, label=f"threshold {_thresh}")
            _ax.legend(fontsize=8)
        _ax.set_ylabel(_ylabel)
        _ax.set_xlabel("Resolution")
        _ax.grid(True, alpha=0.3, axis="y")
        for _bar, _val in zip(_bars, _vals):
            if not np.isnan(_val):
                _ax.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height() * 1.01,
                         f"{int(_val):,}" if _metric == "n_cells" else f"{_val:.1f}",
                         ha="center", va="bottom", fontsize=9)
    _fig3.suptitle("Mesh quality — box domain", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _fig3.savefig(FIG_DIR / "convergence_box_mesh_quality.png", dpi=150, bbox_inches="tight")
    _fig3
    return


@app.cell
def _(COLORS, FIG_DIR, pd, plt, solver_logs):
    _summary = []
    for _res, _df in sorted(solver_logs.items()):
        _row = {"resolution_m": _res, "n_iter": len(_df)}
        for _f in ["Ux", "p_rgh", "k", "T"]:
            _v = _df[_f].dropna()
            _row[f"final_{_f}"] = _v.iloc[-1] if len(_v) else float("nan")
        _summary.append(_row)
    df_summary = pd.DataFrame(_summary).set_index("resolution_m")

    _fig2, _ax2 = plt.subplots(figsize=(7, 4))
    _x = [500, 250, 100]
    for _f, _mk, _lb in [("final_Ux", "o", "U_x"), ("final_k", "s", "k"),
                          ("final_T", "^", "T"), ("final_p_rgh", "D", "p_rgh")]:
        _vals = [
            df_summary.loc[_r, _f]
            if _r in df_summary.index and _f in df_summary.columns
            else float("nan")
            for _r in _x
        ]
        _ax2.semilogy(_x, _vals, marker=_mk, label=_lb, linewidth=1.5, markersize=6)
    _ax2.axhline(1e-4, color="k", linestyle="--", lw=0.8, alpha=0.5, label="1e-4 target")
    _ax2.set_xlabel("Resolution (m)")
    _ax2.set_ylabel("Final initial residual")
    _ax2.set_xticks([100, 250, 500])
    _ax2.set_xticklabels(["100m", "250m", "500m"])
    _ax2.legend()
    _ax2.grid(True, alpha=0.3)
    _ax2.set_title("Final residuals vs resolution — box domain")
    plt.tight_layout()
    _fig2.savefig(FIG_DIR / "convergence_box_final_residuals.png", dpi=150, bbox_inches="tight")
    _fig2
    return (df_summary,)


@app.cell
def _(df_summary, mesh_stats, mo, solver_logs):
    _rows = []
    for _r in [500, 250, 100]:
        _m = mesh_stats.get(_r, {})
        _s = solver_logs.get(_r)
        _fx = _s["Ux"].dropna().iloc[-1] if _s is not None and len(_s["Ux"].dropna()) else float("nan")
        _fk = _s["k"].dropna().iloc[-1]  if _s is not None and len(_s["k"].dropna())  else float("nan")
        _ni = len(_s) if _s is not None else "?"
        _rows.append({
            "Res (m)": _r,
            "Cells": f"{_m.get('n_cells', '?'):,}" if isinstance(_m.get("n_cells"), int) else "?",
            "MaxNonOrth": f"{_m.get('max_nonorth', float('nan')):.1f}°",
            "MeshOK": "✓" if _m.get("mesh_ok") else "✗",
            "Iterations": _ni,
            "Final Ux resid": f"{_fx:.2e}",
            "Final k resid": f"{_fk:.2e}",
        })
    import pandas as _pd
    _df_table = _pd.DataFrame(_rows)
    mo.md(_df_table.to_markdown(index=False))
    return


@app.cell
def _(mo):
    mo.md("""
    ## Observations

    - **case_res_100** ran to t=600 (walltime limit) — re-submit with `walltime=4:00:00`.
    - **Function objects** (solverInfo, fieldMinMax, volAverages) only active for case_res_500.
      Re-generate 250m and 100m cases from updated template before next run.
    - **High k values** (mean ~2000–3000 m²/s²) are in top-corner cells — boundary artifact,
      not affecting the center zone (GNN output cube).

    ## Gate decision (Phase 0.2)

    After re-running case_res_100 to t=2000:
    - If RMSE(100m) < RMSE(250m) × 0.8 **and** cost < 2h → **use 100m for campaign**
    - If gains marginal (< 5%) → **250m sufficient** (4× cost reduction)
    """)
    return


if __name__ == "__main__":
    app.run()
