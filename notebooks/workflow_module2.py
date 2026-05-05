import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Module 2A — SF vs BBSF Convergence Study

    **Site**: Perdigão, Portugal — 2 observed IOP conditions
    **Domain**: 5×5 km centred on obs masts
    **Solvers**: simpleFoam (neutral) vs buoyantBoussinesqSimpleFoam (Boussinesq)
    **Platform**: OF2412 ESI via Apptainer on Aqua HPC

    4 resolutions × 2 directions × 2 solvers = **16 cases**
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    import yaml

    ROOT = Path(__file__).parent.parent.resolve() if "__file__" in dir() else Path("..").resolve()
    CASES_DIR = ROOT / "data" / "campaign" / "convergence_sf_bbsf"

    _cfd_path = str(ROOT / "services" / "module2a-cfd")
    if _cfd_path not in sys.path:
        sys.path.insert(0, _cfd_path)

    with open(ROOT / "configs" / "sites" / "perdigao.yaml") as _fh:
        SITE_CFG = yaml.safe_load(_fh)

    SRTM_TIF = ROOT / "data" / "raw" / "srtm_perdigao_30m.tif"

    print(f"Site: {SITE_CFG['site']['name']}, Cases: {CASES_DIR}")
    return CASES_DIR, ROOT, SITE_CFG, SRTM_TIF


# ---- Analysis section ----

@app.cell
def _(CASES_DIR, SITE_CFG):
    import numpy as np
    import pandas as pd

    # Tower elevations (ASL)
    _key_towers = (
        SITE_CFG.get("terrain", {}).get("key_towers", [])
        or SITE_CFG.get("measurement", {}).get("masts", {}).get("key_towers", [])
    )
    TOWER_Z = {t["id"]: t["elevation_m"] for t in _key_towers}
    TOWER_NAMES = {"T20": "Ridge", "T13": "Flank", "T25": "Valley"}

    RESOLUTIONS = [250, 100, 50, 30]
    DIRECTIONS = ["east", "sw"]
    DIR_LABELS = {"east": "East 91.7° / 7.88 m/s", "sw": "SW 228.7° / 10.46 m/s"}

    def load_profile(case_id: str, tower: str) -> pd.DataFrame | None:
        """Load a tower profile CSV from postProcessing."""
        _candidates = [
            CASES_DIR / case_id / "postProcessing" / "sampleDict" / "2000" / f"{tower}_epsilon_k_U.csv",
            CASES_DIR / "postProcessing" / case_id / "sampleDict" / "2000" / f"{tower}_epsilon_k_U.csv",
        ]
        for _p in _candidates:
            if _p.exists():
                _df = pd.read_csv(_p)
                if len(_df) > 0:
                    return _df
        return None

    def profile_speed(df: pd.DataFrame, z_ground: float):
        """Return (AGL heights, wind speed) from a profile DataFrame."""
        _agl = df["z"].values - z_ground
        _speed = np.sqrt(df["U_0"].values**2 + df["U_1"].values**2 + df["U_2"].values**2)
        return _agl, _speed

    # Collect all profiles
    profiles = {}
    for _direction in DIRECTIONS:
        for _solver_tag in ["sf", "bbsf"]:
            for _res in RESOLUTIONS:
                _cid = f"{_direction}_{_solver_tag}_{_res}m"
                for _tower in TOWER_Z:
                    _df = load_profile(_cid, _tower)
                    if _df is not None:
                        _agl, _speed = profile_speed(_df, TOWER_Z[_tower])
                        profiles[(_direction, _solver_tag, _res, _tower)] = (_agl, _speed)

    print(f"Loaded {len(profiles)} profiles ({len(TOWER_Z)} towers × {len(RESOLUTIONS)} res × 2 dirs × 2 solvers)")
    return DIRECTIONS, DIR_LABELS, RESOLUTIONS, TOWER_NAMES, TOWER_Z, np, pd, profiles


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Figure 1 — Vertical wind profiles: SF vs BBSF")
    return


@app.cell
def _(DIRECTIONS, DIR_LABELS, RESOLUTIONS, TOWER_NAMES, TOWER_Z, mo, np, profiles):
    import matplotlib.pyplot as plt

    _towers = list(TOWER_Z.keys())
    _fig, _axes = plt.subplots(len(DIRECTIONS), len(_towers), figsize=(14, 8),
                               sharey=True, squeeze=False)

    _colors = {250: "#1f77b4", 100: "#ff7f0e", 50: "#2ca02c", 30: "#d62728"}

    for _i, _dir in enumerate(DIRECTIONS):
        for _j, _tw in enumerate(_towers):
            _ax = _axes[_i, _j]
            for _res in RESOLUTIONS:
                _c = _colors[_res]
                # SF (solid)
                _key_sf = (_dir, "sf", _res, _tw)
                if _key_sf in profiles:
                    _agl, _spd = profiles[_key_sf]
                    _m = _agl > 0
                    _ax.plot(_spd[_m], _agl[_m], color=_c, ls="-", lw=1.5,
                             label=f"SF {_res}m")
                # BBSF (dashed)
                _key_bb = (_dir, "bbsf", _res, _tw)
                if _key_bb in profiles:
                    _agl, _spd = profiles[_key_bb]
                    _m = (_agl > 0) & (_spd < 50)
                    if np.sum(_m) > 2:
                        _ax.plot(_spd[_m], _agl[_m], color=_c, ls="--", lw=1.5,
                                 label=f"BBSF {_res}m")

            _ax.set_xlim(0, None)
            _ax.set_ylim(0, 200)
            _tname = TOWER_NAMES.get(_tw, _tw)
            _ax.set_title(f"{_tw} ({_tname}) — {DIR_LABELS.get(_dir, _dir)}", fontsize=9)
            if _j == 0:
                _ax.set_ylabel("Height AGL [m]")
            if _i == len(DIRECTIONS) - 1:
                _ax.set_xlabel("|U| [m/s]")
            _ax.grid(True, alpha=0.3)

    _handles, _labels = _axes[0, 0].get_legend_handles_labels()
    _fig.legend(_handles, _labels, loc="upper center", ncol=4, fontsize=8,
                bbox_to_anchor=(0.5, 1.02))
    _fig.suptitle("SF (solid) vs BBSF (dashed)", y=1.05, fontsize=13)
    _fig.tight_layout()
    fig_profiles = _fig
    return (fig_profiles,)


@app.cell
def _(fig_profiles):
    fig_profiles
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Figure 2 — Speed difference BBSF − SF by resolution")
    return


@app.cell
def _(DIRECTIONS, RESOLUTIONS, TOWER_NAMES, TOWER_Z, np, profiles):
    import matplotlib.pyplot as plt

    _towers = list(TOWER_Z.keys())
    _fig, _axes = plt.subplots(1, len(_towers), figsize=(14, 4.5), sharey=True, squeeze=False)
    _axes = _axes[0]

    _dcols = {"east": "#1f77b4", "sw": "#d62728"}
    _dmrk = {"east": "o", "sw": "s"}

    for _j, _tw in enumerate(_towers):
        _ax = _axes[_j]
        for _dir in DIRECTIONS:
            _d10, _d100, _vres = [], [], []
            for _res in RESOLUTIONS:
                _ksf = (_dir, "sf", _res, _tw)
                _kbb = (_dir, "bbsf", _res, _tw)
                if _ksf not in profiles or _kbb not in profiles:
                    continue
                _agl_sf, _spd_sf = profiles[_ksf]
                _agl_bb, _spd_bb = profiles[_kbb]
                if np.max(_spd_bb) > 50:
                    continue
                _spd_bb_i = np.interp(_agl_sf, _agl_bb, _spd_bb)
                _m = _agl_sf > 0
                _diff = _spd_bb_i[_m] - _spd_sf[_m]
                _agl_v = _agl_sf[_m]
                _i10 = np.argmin(np.abs(_agl_v - 10))
                _i100 = np.argmin(np.abs(_agl_v - 100))
                _d10.append(float(_diff[_i10]))
                _d100.append(float(_diff[_i100]))
                _vres.append(_res)

            if _vres:
                _ax.plot(_vres, _d10, color=_dcols[_dir], marker=_dmrk[_dir],
                         ls="-", label=f"{_dir} @10m")
                _ax.plot(_vres, _d100, color=_dcols[_dir], marker=_dmrk[_dir],
                         ls="--", alpha=0.6, label=f"{_dir} @100m")

        _ax.axhline(0, color="k", lw=0.5, ls=":")
        _ax.set_xlabel("Resolution [m]")
        _ax.set_title(f"{_tw} ({TOWER_NAMES.get(_tw, '')})", fontsize=10)
        _ax.invert_xaxis()
        _ax.grid(True, alpha=0.3)
        if _j == 0:
            _ax.set_ylabel("U_BBSF − U_SF [m/s]")
            _ax.legend(fontsize=7)

    _fig.suptitle("Difference BBSF − SF by resolution (solid=@10m, dashed=@100m)", fontsize=12)
    _fig.tight_layout()
    fig_diff = _fig
    return (fig_diff,)


@app.cell
def _(fig_diff):
    fig_diff
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Table — |U| comparison at 10m and 100m AGL")
    return


@app.cell
def _(DIRECTIONS, RESOLUTIONS, TOWER_Z, mo, np, pd, profiles):
    _rows = []
    for _dir in DIRECTIONS:
        for _tw in TOWER_Z:
            for _res in RESOLUTIONS:
                _row = {"direction": _dir, "tower": _tw, "resolution_m": _res}
                for _tag in ["sf", "bbsf"]:
                    _key = (_dir, _tag, _res, _tw)
                    if _key in profiles:
                        _agl, _spd = profiles[_key]
                        if np.max(_spd) > 50:
                            _row[f"{_tag}_10m"] = None
                            _row[f"{_tag}_100m"] = None
                            continue
                        _m = _agl > 0
                        if np.sum(_m) < 2:
                            continue
                        _i10 = np.argmin(np.abs(_agl[_m] - 10))
                        _i100 = np.argmin(np.abs(_agl[_m] - 100))
                        _row[f"{_tag}_10m"] = round(float(_spd[_m][_i10]), 2)
                        _row[f"{_tag}_100m"] = round(float(_spd[_m][_i100]), 2)
                _rows.append(_row)

    df_compare = pd.DataFrame(_rows)
    df_compare["delta_10m"] = df_compare.apply(
        lambda r: round(r["bbsf_10m"] - r["sf_10m"], 2)
        if pd.notna(r.get("bbsf_10m")) and pd.notna(r.get("sf_10m")) else None, axis=1)
    df_compare["delta_100m"] = df_compare.apply(
        lambda r: round(r["bbsf_100m"] - r["sf_100m"], 2)
        if pd.notna(r.get("bbsf_100m")) and pd.notna(r.get("sf_100m")) else None, axis=1)

    mo.ui.table(df_compare, selection=None)
    return (df_compare,)


@app.cell(hide_code=True)
def _(df_compare, mo):
    _valid = df_compare.dropna(subset=["delta_10m", "delta_100m"])
    _mean_10 = _valid["delta_10m"].mean()
    _mean_100 = _valid["delta_100m"].mean()

    mo.md(f"""
    ### Diagnostic

    - **BBSF systématiquement plus rapide que SF**: delta moyen = {_mean_10:.1f} m/s @10m, {_mean_100:.1f} m/s @100m
    - **Cause**: T = TRef partout → terme Boussinesq = 0. L'écart vient des BCs de pression
      différentes (`fixedFluxPressure` BBSF vs `fixedValue` SF), pas de la buoyance.
    - **Conclusion**: cette comparaison est invalide pour mesurer l'effet thermique.

    ### Prochaines étapes
    1. Aligner les BCs de pression entre SF et BBSF (mêmes BCs Robin/inletOutlet)
    2. Initialiser T depuis un profil ERA5 réaliste (gradient vertical, pas uniforme)
    3. Relancer BBSF avec T non-neutre pour mesurer le vrai effet Boussinesq
    """)
    return


if __name__ == "__main__":
    app.run()
