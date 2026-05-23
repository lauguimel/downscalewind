"""audit_surrogate_vs_obs.py — Phase G M_G8 audit: stratified comparison
of the surrogate v2 prediction vs in-situ OBS at the OBS stations.

Consumes the parquet produced by M_G7
(`data/inference/surrogate_at_stations.parquet`) and produces:

  data/validation/phase_G_obs_audit/<out>/
    csv/
      metrics_global.csv
      metrics_by_<strata>.csv  (source, class_topo, height_bucket,
                                wind_class, season, era5_freshness,
                                class_topo_x_wind_class,
                                season_x_height_bucket)
    figures/
      scatter_pred_vs_obs.png
      mae_by_strata.png
      distribution_by_source.png
      era5_freshness_impact.png
      bias_by_season_height.png
    REPORT.md

Stratification (per M_G8 mandate slice):
  class_topo, height_bucket, wind_class, season, era5_freshness, source

Smoke (CPU, ≤10 pairings):
  conda run -n downscalewind python services/validation/audit_surrogate_vs_obs.py \
    --parquet data/inference/smoke_surrogate_at_stations.parquet \
    --output-dir data/validation/phase_G_obs_audit/smoke/

Production:
  conda run -n downscalewind python services/validation/audit_surrogate_vs_obs.py \
    --parquet data/inference/surrogate_at_stations.parquet \
    --output-dir data/validation/phase_G_obs_audit/v1/
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
for _p in (_SCRIPT_DIR, _PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from services.validation.utils.strata import (  # noqa: E402
    HEIGHT_BUCKETS,
    add_strata,
    metrics_by_strata,
    pairing_metrics,
)

logger = logging.getLogger("audit_surrogate_vs_obs")


# ─── Verdict thresholds (M_G8 → Phase H gating) ─────────────────────────────

GO_MAE_THRESHOLD = 1.5      # m/s — median MAE under which Phase H is worth pursuing
GO_BIAS_THRESHOLD = 1.0     # m/s — |median bias|; if larger, surrogate has
                            #        an offset that should be DNN-correctable
NO_GO_MAE_PLATEAU = 3.0     # m/s — above this MAE everywhere = saturated
MIN_N_FOR_DECISION = 1000   # min pairings for a confident GO/NO-GO

STRATA_SETS: list[str | tuple[str, ...]] = [
    "source",
    "class_topo",
    "height_bucket",
    "wind_class",
    "season",
    "era5_freshness",
    ("class_topo", "wind_class"),
    ("season", "height_bucket"),
]


def _strata_filename(by: str | Sequence[str]) -> str:
    if isinstance(by, str):
        return f"metrics_by_{by}.csv"
    return "metrics_by_" + "_x_".join(by) + ".csv"


def _strata_label(by: str | Sequence[str]) -> str:
    if isinstance(by, str):
        return by
    return " × ".join(by)


# ─── IO ─────────────────────────────────────────────────────────────────────

def load_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {
        "station_id", "timestamp", "source", "lat", "lon", "elev", "height_obs",
        "u_obs", "v_obs", "speed_obs",
        "u_pred", "v_pred", "w_pred", "speed_pred",
        "era5_time_delta_minutes",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"parquet missing required columns: {sorted(missing)}")
    logger.info("Loaded %s | rows=%d | sources=%s | stations=%d",
                path, len(df), sorted(df["source"].unique()),
                df["station_id"].nunique())
    return df


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.4f")
    logger.info("wrote %s (%d rows)", path, len(df))


# ─── Figures ────────────────────────────────────────────────────────────────

def fig_scatter_pred_vs_obs(df: pd.DataFrame, out: Path) -> None:
    obs = df["speed_obs"].to_numpy()
    pred = df["speed_pred"].to_numpy()
    m = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[m], pred[m]
    n = obs.size
    fig, ax = plt.subplots(figsize=(6, 6))
    if n >= 200:
        ax.hexbin(obs, pred, gridsize=40, cmap="viridis",
                  mincnt=1, bins="log")
    else:
        for src, sub in df[m].groupby("source"):
            ax.scatter(sub["speed_obs"], sub["speed_pred"], s=12,
                       alpha=0.6, label=src)
        if n > 0 and df.loc[m, "source"].nunique() > 1:
            ax.legend(loc="upper left", fontsize=8)
    lo = float(np.nanmin([0.0, obs.min() if n else 0.0, pred.min() if n else 0.0]))
    hi = float(np.nanmax([5.0, obs.max() if n else 5.0, pred.max() if n else 5.0]))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    g = pairing_metrics(obs, pred)
    if np.isfinite(g["slope"]):
        x = np.array([lo, hi])
        ax.plot(x, g["slope"] * x + g["intercept"], "r-", lw=1.2,
                label=f"fit a={g['slope']:.2f} b={g['intercept']:.2f} R²={g['R2']:.2f}")
    ax.set_xlabel("speed_obs (m/s)")
    ax.set_ylabel("speed_pred (m/s)")
    ax.set_title(f"surrogate v2 vs OBS — N={n}\n"
                 f"MAE={g['MAE']:.2f} RMSE={g['RMSE']:.2f} bias={g['bias']:+.2f} m/s")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    logger.info("wrote %s", out)


def _heatmap(ax, mat: np.ndarray, xticks: list, yticks: list,
             title: str, cbar_label: str, cmap: str, vmin=None, vmax=None,
             fmt: str = ".2f") -> None:
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(xticks)))
    ax.set_xticklabels(xticks, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(yticks)))
    ax.set_yticklabels(yticks, fontsize=9)
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, format(v, fmt), ha="center", va="center",
                        fontsize=8, color="white" if abs(v) > 1.0 else "black")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label(cbar_label, fontsize=9)


def fig_mae_by_strata(df: pd.DataFrame, out: Path) -> None:
    """Heatmap MAE × (class_topo × wind_class)."""
    topo_order = ["plain", "foothill", "mountain", "summit"]
    wind_order = ["low", "mid", "high"]
    mat = np.full((len(topo_order), len(wind_order)), np.nan)
    n_mat = np.zeros_like(mat)
    for i, tc in enumerate(topo_order):
        for j, wc in enumerate(wind_order):
            sub = df[(df["class_topo"] == tc) & (df["wind_class"] == wc)]
            if len(sub) == 0:
                continue
            stats = pairing_metrics(sub["speed_obs"].to_numpy(),
                                    sub["speed_pred"].to_numpy())
            mat[i, j] = stats["MAE"]
            n_mat[i, j] = stats["N"]
    fig, ax = plt.subplots(figsize=(7, 5))
    _heatmap(ax, mat, wind_order, topo_order,
             "MAE (m/s) — class_topo × wind_class",
             "MAE (m/s)", cmap="YlOrRd", vmin=0.0,
             vmax=float(np.nanmax(mat)) if np.isfinite(mat).any() else 1.0)
    # overlay N counts in lower-right corner of each cell
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if n_mat[i, j] > 0:
                ax.text(j + 0.35, i + 0.35, f"N={int(n_mat[i, j])}",
                        ha="right", va="bottom", fontsize=7, color="gray")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    logger.info("wrote %s", out)


def fig_distribution_by_source(df: pd.DataFrame, out: Path) -> None:
    sources = sorted(df["source"].unique())
    n_src = len(sources)
    fig, axes = plt.subplots(1, max(n_src, 1), figsize=(4.0 * max(n_src, 1), 4.0),
                             squeeze=False)
    axes = axes[0]
    bins = np.linspace(0.0, max(15.0, float(df[["speed_obs", "speed_pred"]].max().max() or 15.0)), 30)
    for ax, src in zip(axes, sources):
        sub = df[df["source"] == src]
        ax.hist(sub["speed_obs"].dropna(), bins=bins, alpha=0.55,
                label="OBS", color="steelblue", density=True)
        ax.hist(sub["speed_pred"].dropna(), bins=bins, alpha=0.55,
                label="surrogate", color="firebrick", density=True)
        ax.set_title(f"{src} (N={len(sub)})")
        ax.set_xlabel("wind speed (m/s)")
        ax.set_ylabel("pdf")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    logger.info("wrote %s", out)


def fig_era5_freshness_impact(df: pd.DataFrame, out: Path) -> None:
    delta = np.abs(df["era5_time_delta_minutes"].to_numpy())
    err = np.abs(df["speed_pred"].to_numpy() - df["speed_obs"].to_numpy())
    m = np.isfinite(delta) & np.isfinite(err)
    delta, err = delta[m], err[m]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # (a) continuous scatter
    ax = axes[0]
    ax.scatter(delta, err, s=10, alpha=0.4, color="steelblue")
    if delta.size >= 5:
        # rolling MAE in bins
        bins = np.linspace(0.0, max(delta.max(), 1.0), 8)
        centers = 0.5 * (bins[:-1] + bins[1:])
        mae_bins = [float(np.nanmean(err[(delta >= bins[i]) & (delta < bins[i + 1])]))
                    if ((delta >= bins[i]) & (delta < bins[i + 1])).sum() > 0
                    else float("nan")
                    for i in range(len(bins) - 1)]
        ax.plot(centers, mae_bins, "ro-", lw=1.5, label="binned mean |err|")
        ax.legend(fontsize=8)
    ax.set_xlabel("|era5_time_delta_minutes|")
    ax.set_ylabel("|speed_pred − speed_obs| (m/s)")
    ax.set_title("Per-pairing error vs ERA5 freshness")
    ax.grid(True, alpha=0.3)

    # (b) box plot per freshness class
    ax = axes[1]
    order = ["on_time", "interpolated", "far"]
    data = []
    labels = []
    for fr in order:
        sub_err = np.abs(df.loc[df["era5_freshness"] == fr, "speed_pred"]
                         - df.loc[df["era5_freshness"] == fr, "speed_obs"]).dropna()
        if len(sub_err) > 0:
            data.append(sub_err.to_numpy())
            labels.append(f"{fr}\nN={len(sub_err)}")
    if data:
        ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylabel("|speed_pred − speed_obs| (m/s)")
    ax.set_title("Error distribution by ERA5 freshness")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    logger.info("wrote %s", out)


def fig_bias_by_season_height(df: pd.DataFrame, out: Path) -> None:
    season_order = ["winter", "spring", "summer", "autumn"]
    height_order = [int(h) for h in HEIGHT_BUCKETS]
    mat = np.full((len(season_order), len(height_order)), np.nan)
    n_mat = np.zeros_like(mat)
    for i, s in enumerate(season_order):
        for j, h in enumerate(height_order):
            sub = df[(df["season"] == s) & (df["height_bucket"] == h)]
            if len(sub) == 0:
                continue
            stats = pairing_metrics(sub["speed_obs"].to_numpy(),
                                    sub["speed_pred"].to_numpy())
            mat[i, j] = stats["bias"]
            n_mat[i, j] = stats["N"]
    fig, ax = plt.subplots(figsize=(7, 5))
    vmax = float(np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else 1.0
    _heatmap(ax, mat, [str(h) for h in height_order], season_order,
             "Bias (pred − obs) m/s — season × height_bucket",
             "bias (m/s)", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
             fmt="+.2f")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if n_mat[i, j] > 0:
                ax.text(j + 0.35, i + 0.35, f"N={int(n_mat[i, j])}",
                        ha="right", va="bottom", fontsize=7, color="gray")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    logger.info("wrote %s", out)


# ─── Verdict logic ──────────────────────────────────────────────────────────

def make_verdict(global_metrics: dict, per_source: pd.DataFrame,
                 by_topo: pd.DataFrame, n_pairings: int) -> tuple[str, str]:
    """Return (verdict, rationale) using thresholds defined at top of file."""
    mae = global_metrics["MAE"]
    bias = global_metrics["bias"]

    rationale_lines: list[str] = []
    rationale_lines.append(
        f"Global MAE={mae:.2f} m/s, bias={bias:+.2f} m/s, "
        f"R²={global_metrics['R2']:.2f}, N={n_pairings}."
    )

    # n trop petit = audit informatif mais pas décisif
    if n_pairings < MIN_N_FOR_DECISION:
        rationale_lines.append(
            f"N={n_pairings} < {MIN_N_FOR_DECISION} → audit smoke insuffisant "
            f"pour décision finale Phase H ; rapport YELLOW pour valider la "
            f"pipeline, attendre full prod parquet."
        )
        return "YELLOW", "\n".join(rationale_lines)

    if not np.isfinite(mae):
        return "RED", "MAE not finite — pipeline issue, investigate parquet."

    # patterns exploitables : strates dispersées en MAE → DNN correction utile
    topo_ok = by_topo.dropna(subset=["MAE"])
    if len(topo_ok) >= 2:
        spread = float(topo_ok["MAE"].max() - topo_ok["MAE"].min())
        rationale_lines.append(
            f"MAE spread across class_topo = {spread:.2f} m/s "
            f"({topo_ok['MAE'].min():.2f} → {topo_ok['MAE'].max():.2f})."
        )
        exploitable_pattern = spread > 0.5  # heuristique
    else:
        exploitable_pattern = False

    if mae < GO_MAE_THRESHOLD and abs(bias) < GO_BIAS_THRESHOLD:
        verdict = "GO"
        rationale_lines.append(
            f"MAE < {GO_MAE_THRESHOLD} m/s ET |bias| < {GO_BIAS_THRESHOLD} m/s "
            f"→ baseline surrogate déjà bonne, Phase H peut affiner stratifié."
        )
    elif mae > NO_GO_MAE_PLATEAU:
        verdict = "NO-GO"
        rationale_lines.append(
            f"MAE > {NO_GO_MAE_PLATEAU} m/s sur l'ensemble : surrogate saturé, "
            f"pas de marge exploitée par DNN correction → revoir stack CFD."
        )
    elif exploitable_pattern:
        verdict = "RESCOPE"
        rationale_lines.append(
            "MAE intermédiaire MAIS stratification montre patterns (classe topo "
            "ou autre) → Phase H sur strates ciblées (RESCOPE)."
        )
    else:
        verdict = "RESCOPE"
        rationale_lines.append(
            "MAE intermédiaire sans patterns stratifiés évidents → RESCOPE "
            "vers extension dataset OBS (Phase G+) avant DNN correction."
        )

    return verdict, "\n".join(rationale_lines)


# ─── Markdown report ────────────────────────────────────────────────────────

def write_report(
    *,
    df: pd.DataFrame,
    global_metrics: dict,
    metrics_by: dict[str, pd.DataFrame],
    verdict: str,
    rationale: str,
    output_dir: Path,
    parquet_path: Path,
) -> None:
    figures_dir = output_dir / "figures"
    csv_dir = output_dir / "csv"

    sources = sorted(df["source"].unique())
    n_unique_stations = int(df["station_id"].nunique())
    t_min = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").min()
    t_max = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").max()

    def _table_md(d: pd.DataFrame, cols: list[str]) -> str:
        if d.empty:
            return "_(empty)_\n"
        sub = d[cols].copy()
        for c in sub.columns:
            if sub[c].dtype.kind in "fc":
                sub[c] = sub[c].map(lambda v: "" if not np.isfinite(v) else f"{v:.3f}")
            else:
                sub[c] = sub[c].astype(str)
        # manual pipe table (avoids tabulate optional dep)
        header = "| " + " | ".join(sub.columns) + " |"
        sep = "| " + " | ".join("---" for _ in sub.columns) + " |"
        rows = ["| " + " | ".join(r) + " |" for r in sub.values.tolist()]
        return "\n".join([header, sep] + rows) + "\n"

    lines: list[str] = []
    lines.append(f"# M_G8 audit — surrogate v2 vs OBS\n")
    lines.append(f"_Auto-generated {datetime.utcnow().isoformat(timespec='seconds')}Z "
                 f"from `{parquet_path}`._\n")
    lines.append(f"## §1 Executive summary\n")
    lines.append(f"- n_pairings = **{len(df)}** ({n_unique_stations} unique "
                 f"stations across {len(sources)} sources: "
                 f"{', '.join(sources)})")
    lines.append(f"- time range = {t_min} → {t_max}")
    lines.append(f"- global MAE = **{global_metrics['MAE']:.3f} m/s** | "
                 f"RMSE = {global_metrics['RMSE']:.3f} | "
                 f"bias = {global_metrics['bias']:+.3f} | R² = {global_metrics['R2']:.3f}")
    lines.append(f"- affine fit `speed_pred = {global_metrics['slope']:.3f}·speed_obs "
                 f"+ {global_metrics['intercept']:.3f}`")
    lines.append(f"- p10/p90 of `speed_pred / speed_obs` = "
                 f"{global_metrics['p10_ratio']:.3f} / {global_metrics['p90_ratio']:.3f}")
    lines.append(f"")
    lines.append(f"**Comparison vs M17 baseline** (N=7 ICOS tall-tower 2020): "
                 f"M17 a montré `U_cfd = 0.54·U_obs + 1.88` (R²=0.43). "
                 f"L'audit présent ouvre la stratification à >7 sites pour vérifier "
                 f"si ce biais affine est universel ou strate-dépendant.")
    lines.append(f"")
    lines.append(f"**ERA5 baseline comparison**: skip — `speed_era5_baseline` "
                 f"absent du parquet M_G7. Caveat : la baseline ERA5 n'est pas "
                 f"comparée directement ici. Pour Phase H, ajouter `speed_era5_baseline` "
                 f"au parquet via `infer_at_stations.py` (extraire `era5_surface/u10,v10` "
                 f"du grid.zarr au point central).")
    lines.append(f"")

    lines.append(f"## §2 Methodology — stratification\n")
    lines.append(
        "Chaque pairing est étiqueté selon 6 axes :\n"
        "- **class_topo** : plain (<300m) / foothill (300-800m) / mountain (800-1500m) / "
        "summit (>1500m) (basé sur `elev`)\n"
        "- **height_bucket** : nearest in {10, 20, 50, 100} m AGL (`height_obs`)\n"
        "- **wind_class** : low (<3) / mid (3-7) / high (>7) m/s (`speed_obs`)\n"
        "- **season** : DJF / MAM / JJA / SON (timestamp month)\n"
        "- **era5_freshness** : on_time (Δt<30 min) / interpolated (30-180) / far (>180)\n"
        "- **source** : passthrough\n"
    )
    lines.append(
        "Métriques par strate : N, MAE, RMSE, bias, p10/p90 de "
        "`speed_pred / speed_obs`, slope+intercept+R² du fit `speed_pred = a·speed_obs + b`.\n"
    )

    lines.append(f"## §3 Per-strata results\n")
    metric_cols = ["N", "MAE", "RMSE", "bias", "p10_ratio", "p90_ratio",
                   "slope", "intercept", "R2"]

    for key, dframe in metrics_by.items():
        lines.append(f"### {key}\n")
        lines.append(_table_md(dframe, [c for c in dframe.columns
                                        if c not in metric_cols] + metric_cols))
        lines.append("")

    lines.append(f"## Figures\n")
    rel_fig = figures_dir.relative_to(output_dir)
    lines.append(f"![scatter]({rel_fig}/scatter_pred_vs_obs.png)\n")
    lines.append(f"![mae_by_strata]({rel_fig}/mae_by_strata.png)\n")
    lines.append(f"![distribution_by_source]({rel_fig}/distribution_by_source.png)\n")
    lines.append(f"![era5_freshness_impact]({rel_fig}/era5_freshness_impact.png)\n")
    lines.append(f"![bias_by_season_height]({rel_fig}/bias_by_season_height.png)\n")
    lines.append("")

    lines.append(f"## §4 Verdict Phase H : **{verdict}**\n")
    lines.append(rationale)
    lines.append("")
    lines.append(
        f"Seuils utilisés :\n"
        f"- GO si MAE médian < {GO_MAE_THRESHOLD} m/s ET |bias| < {GO_BIAS_THRESHOLD} m/s\n"
        f"- NO-GO si MAE > {NO_GO_MAE_PLATEAU} m/s (plateau saturé)\n"
        f"- RESCOPE sinon, en fonction de patterns stratifiés exploitables\n"
        f"- YELLOW si N < {MIN_N_FOR_DECISION} pairings (audit smoke, pas de décision finale)\n"
    )

    lines.append(f"## §5 Limits + next steps\n")
    n_summit = int((df.get("class_topo") == "summit").sum() if "class_topo" in df.columns else 0)
    n_coastal = "skipped (no `coastal` flag in parquet — needs distance-to-sea)"
    lines.append(
        f"- Couverture summit (alpine >1500m) dans cet audit : N={n_summit}. "
        f"Phase I (re-simulation domaine 10×10×5 km) reste pertinente si N_summit "
        f"insuffisant ou MAE > GO threshold pour cette strate.\n"
        f"- Couverture coastal : {n_coastal}. Optionnel pour M_G8, "
        f"à ajouter Phase G+ si discriminant attendu.\n"
        f"- ERA5 6h cadence (cf. engineer.md) : prédictions constantes par blocs "
        f"6h ; l'audit `era5_freshness_impact.png` quantifie l'effet. "
        f"Production Phase H devrait utiliser un ERA5 hourly store avec d2m.\n"
        f"- M17 baseline (N=7 ICOS) avait montré biais affine `U_cfd = 0.54·U_obs + "
        f"1.88` ; vérifier ici si la pente <1 est universelle ou tirée par strates "
        f"continentales spécifiques.\n"
        f"- Next steps : (a) si verdict GO → Phase H DNN correction stratifié sur "
        f"strates {['class_topo','wind_class']} ; (b) si RESCOPE → augmenter "
        f"dataset OBS (NOAA ISD full ingest, hourly ERA5) avant ré-audit.\n"
    )

    report_path = output_dir / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", report_path)


# ─── Driver ─────────────────────────────────────────────────────────────────

def run_audit(parquet: Path, output_dir: Path) -> dict:
    df_raw = load_parquet(parquet)
    df = add_strata(df_raw)

    csv_dir = output_dir / "csv"
    fig_dir = output_dir / "figures"
    csv_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # write the strata-tagged parquet (cheap, useful for downstream)
    df.to_parquet(output_dir / "pairings_with_strata.parquet", index=False)

    # global metrics
    g = pairing_metrics(df["speed_obs"].to_numpy(),
                        df["speed_pred"].to_numpy())
    pd.DataFrame([g]).to_csv(csv_dir / "metrics_global.csv",
                              index=False, float_format="%.4f")
    logger.info("global metrics: %s", json.dumps(g, indent=2))

    # per-strata
    metrics_by: dict[str, pd.DataFrame] = {}
    for by in STRATA_SETS:
        dframe = metrics_by_strata(df, "speed_obs", "speed_pred", by)
        write_csv(dframe, csv_dir / _strata_filename(by))
        metrics_by[_strata_label(by)] = dframe

    # figures
    fig_scatter_pred_vs_obs(df, fig_dir / "scatter_pred_vs_obs.png")
    fig_mae_by_strata(df, fig_dir / "mae_by_strata.png")
    fig_distribution_by_source(df, fig_dir / "distribution_by_source.png")
    fig_era5_freshness_impact(df, fig_dir / "era5_freshness_impact.png")
    fig_bias_by_season_height(df, fig_dir / "bias_by_season_height.png")

    # verdict
    by_topo = metrics_by["class_topo"]
    by_source = metrics_by["source"]
    verdict, rationale = make_verdict(g, by_source, by_topo, n_pairings=len(df))
    logger.info("verdict=%s", verdict)

    write_report(
        df=df,
        global_metrics=g,
        metrics_by=metrics_by,
        verdict=verdict,
        rationale=rationale,
        output_dir=output_dir,
        parquet_path=parquet,
    )

    return {"verdict": verdict, "global_metrics": g, "n_pairings": len(df)}


@click.command(context_settings={"show_default": True})
@click.option("--parquet", type=click.Path(exists=True, path_type=Path),
              required=True, help="Parquet from M_G7 infer_at_stations.py")
@click.option("--output-dir", type=click.Path(path_type=Path),
              required=True,
              help="Output directory (csv/, figures/, REPORT.md, "
                   "pairings_with_strata.parquet)")
@click.option("--verbose", "-v", is_flag=True, default=False)
def cli(parquet, output_dir, verbose):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    res = run_audit(Path(parquet), Path(output_dir))
    logger.info("done. verdict=%s n=%d MAE=%.3f bias=%+.3f",
                res["verdict"], res["n_pairings"],
                res["global_metrics"]["MAE"], res["global_metrics"]["bias"])


if __name__ == "__main__":
    cli()
