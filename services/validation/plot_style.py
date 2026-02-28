"""
plot_style.py — Matplotlib publication style for DownscaleWind figures

Usage
-----
    from services.validation.plot_style import apply_style, COLORS, CMAPS
    apply_style()

All figures use:
  - serif font, 10pt body, 11pt axes labels, 12pt titles
  - 300 dpi on save
  - constrained_layout=True
"""

from __future__ import annotations
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
COLORS = {
    "cfd_1km":    "#1f77b4",   # blue  — CFD 1 km
    "cfd_500m":   "#2ca02c",   # green — CFD 500 m
    "era5":       "#ff7f0e",   # orange — ERA5
    "obs":        "#000000",   # black — observations
    "uncertainty":"#888888",   # gray  — ±1σ band
}

# ---------------------------------------------------------------------------
# Colormaps
# ---------------------------------------------------------------------------
CMAPS = {
    "speed":     "RdYlBu_r",   # wind speed (cool = slow, warm = fast)
    "w_vert":    "RdBu",       # vertical velocity (divergent, centred on 0)
    "terrain":   "terrain",    # elevation
    "tke":       "plasma",     # TKE
}

# ---------------------------------------------------------------------------
# Marker styles for mast positions
# ---------------------------------------------------------------------------
TOWER_MARKER = dict(marker="^", s=60, color="black", zorder=5, linewidths=0.8,
                    edgecolors="white")

# ---------------------------------------------------------------------------
# Main style function
# ---------------------------------------------------------------------------
def apply_style() -> None:
    """Apply publication-quality matplotlib settings."""
    mpl.rcParams.update({
        # Font
        "font.family":          "serif",
        "font.serif":           ["DejaVu Serif", "Times New Roman", "serif"],
        "font.size":            10,
        "axes.labelsize":       11,
        "axes.titlesize":       12,
        "legend.fontsize":      9,
        "xtick.labelsize":      9,
        "ytick.labelsize":      9,
        # Figure
        "figure.dpi":           150,
        "savefig.dpi":          300,
        "figure.constrained_layout.use": True,
        # Lines / axes
        "axes.linewidth":       0.8,
        "lines.linewidth":      1.5,
        "lines.markersize":     5,
        # Grid
        "axes.grid":            True,
        "grid.alpha":           0.3,
        "grid.linewidth":       0.5,
        # Save
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.05,
    })


def save(fig: "plt.Figure", path: str, **kwargs) -> None:
    """Save figure to path at 300 dpi with tight bbox."""
    import pathlib
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", **kwargs)
    print(f"Saved: {path}")
