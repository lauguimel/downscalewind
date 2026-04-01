"""
make_paper_figures.py — Publication-quality figures for the NatComms paper

Figure A: FWI timeline at Pedrogao Grande and Perdigao (May-June 2017)
  Shows daily FWI with danger-class color bands and fire date annotation.

Usage:
    cd /Users/guillaume/Documents/Recherche/downscalewind
    conda run -n downscalewind python services/module2a-cfd/analysis/make_paper_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# -- Paths --
ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
FWI_PEDROGAO = DATA / "raw" / "era5land_fwi_pedrogao_grande.csv"
FWI_PERDIGAO = DATA / "raw" / "era5land_fwi_perdigao.csv"
OUTPUT_DIR = DATA / "validation" / "figures" / "paper"

# -- FWI danger classes (European Forest Fire Information System) --
FWI_CLASSES = [
    (0,  5,  "#2d8e2d", "Low"),
    (5,  12, "#f0c929", "Moderate"),
    (12, 24, "#e68a00", "High"),
    (24, 38, "#cc2222", "Very High"),
    (38, 80, "#7a0000", "Extreme"),
]

FIRE_DATE = pd.Timestamp("2017-06-17")


def load_fwi(path: Path) -> pd.DataFrame:
    """Load FWI CSV and parse dates."""
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def fwi_color(fwi_val: float) -> str:
    """Return color for a given FWI value."""
    for lo, hi, color, _ in FWI_CLASSES:
        if fwi_val < hi:
            return color
    return FWI_CLASSES[-1][2]


def plot_fwi_panel(ax, df, title, show_xlabel=True, show_legend=False,
                   show_fire_annotation=False):
    """Plot FWI time series on a single axis with danger-class coloring."""
    dates = df["date"].values
    fwi = df["fwi"].values

    # Background danger bands
    ymax = max(fwi.max() * 1.15, 16)
    for lo, hi, color, label in FWI_CLASSES:
        band_lo = lo
        band_hi = min(hi, ymax)
        if band_lo >= ymax:
            break
        ax.axhspan(band_lo, band_hi, color=color, alpha=0.10, zorder=0)

    # Bar plot colored by danger class
    colors = [fwi_color(v) for v in fwi]
    bar_width = 0.8  # days
    ax.bar(df["date"], fwi, width=bar_width, color=colors, edgecolor="none",
           alpha=0.85, zorder=2)

    # Thin line connecting tops
    ax.plot(df["date"], fwi, color="k", linewidth=0.6, alpha=0.5, zorder=3)

    # Heatwave shading (June 14-20)
    hw_start = pd.Timestamp("2017-06-14")
    hw_end = pd.Timestamp("2017-06-20")
    ax.axvspan(hw_start, hw_end, color="#cc2222", alpha=0.06, zorder=0)

    # Fire date vertical line
    ax.axvline(FIRE_DATE, color="#cc2222", linewidth=1.8, linestyle="--",
               zorder=4)

    # Fire annotation (only on top panel)
    if show_fire_annotation:
        fire_row = df[df["date"] == FIRE_DATE]
        if not fire_row.empty:
            fire_fwi = fire_row["fwi"].values[0]
            # Temperature on fire date
            fire_T = df.loc[df["date"] == FIRE_DATE, "T_C"].values[0]
            fire_RH = df.loc[df["date"] == FIRE_DATE, "RH"].values[0]
            ax.annotate(
                f"17 June: Pedrógão Grande fire\n"
                f"66 deaths — T={fire_T:.0f}°C, RH={fire_RH:.0f}%",
                xy=(FIRE_DATE, fire_fwi + 0.5),
                xytext=(FIRE_DATE - pd.Timedelta(days=14), ymax * 0.78),
                fontsize=8.5,
                ha="center",
                arrowprops=dict(arrowstyle="-|>", color="#cc2222", lw=1.2,
                                connectionstyle="arc3,rad=0.15"),
                color="#cc2222",
                fontweight="bold",
                zorder=5,
            )

    # Axes formatting
    ax.set_ylabel("FWI (daily)", fontsize=11)
    ax.set_ylim(0, ymax)
    ax.set_xlim(pd.Timestamp("2017-04-29"), pd.Timestamp("2017-07-02"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))
    ax.tick_params(axis="both", labelsize=9.5)

    if not show_xlabel:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("Date (2017)", fontsize=11)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # Thin horizontal lines at class boundaries
    for lo, hi, _, _ in FWI_CLASSES[1:]:
        if lo < ymax:
            ax.axhline(lo, color="grey", linewidth=0.3, linestyle=":", zorder=1)

    ax.set_title(title, fontsize=11.5, fontweight="bold", loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_legend:
        patches = [mpatches.Patch(color=c, alpha=0.7, label=f"{lbl} ({lo}–{hi})")
                   for lo, hi, c, lbl in FWI_CLASSES if lo < ymax]
        ax.legend(handles=patches, loc="upper left", fontsize=7.5,
                  framealpha=0.9, edgecolor="none", ncol=1,
                  title="FWI danger class", title_fontsize=8)


def make_fig_fwi_timeline():
    """Create the 2-panel FWI timeline figure."""
    df_pg = load_fwi(FWI_PEDROGAO)
    df_pd = load_fwi(FWI_PERDIGAO)

    # Publication style
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True,
        gridspec_kw={"hspace": 0.12},
    )

    plot_fwi_panel(ax1, df_pg,
                   "(a) Pedrógão Grande (fire origin, 50 km SW of Perdigão)",
                   show_xlabel=False, show_legend=True,
                   show_fire_annotation=True)
    plot_fwi_panel(ax2, df_pd,
                   "(b) Perdigão (IOP tower network, 50 km NE)",
                   show_xlabel=True, show_legend=False,
                   show_fire_annotation=False)

    fig.align_ylabels([ax1, ax2])

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = OUTPUT_DIR / f"fig_fwi_timeline.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    make_fig_fwi_timeline()
