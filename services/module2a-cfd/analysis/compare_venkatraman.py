"""
compare_venkatraman.py — Overlay our CFD profiles with Venkatraman et al. (2023).

Venkatraman: 2017-05-04 22:00 UTC, SW 231°, 88M cells, 12.5m resolution.
Our closest: ts00 = 2017-05-04 18:00 UTC (4h earlier, S 180°).

Since conditions differ (direction, time), this is NOT a direct benchmark
but shows our profiles alongside a published high-resolution reference.

Usage:
    cd services/module2a-cfd
    python analysis/compare_venkatraman.py
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "data" / "validation" / "poc_tbm_25ts_wc"

# Our case
CASE_DIR = ROOT / "data" / "cases" / "poc_tbm_25ts_wc" / "case_ts00"

# Venkatraman digitized data
VENK_CSV = ROOT / "references" / "venkatraman2023_digitized.csv"

# Towers in Venkatraman with our IDs
TOWER_MAP = {
    "tse04": "Tower 20 (SW ridge)",
    "tse09": "Tower 25 (valley)",
    "tse13": "Tower 29 (NE ridge)",
    "rsw03": "Tower 34 (SW ridge)",
    "tse06": "Tower 22 (valley)",
    "tse11": "Tower 27 (valley)",
    "tnw07": "Tower 7 (valley)",
    "tnw10": "Tower 10 (NE ridge)",
    "rsw06": "Tower 37 (SW ridge)",
}

SEARCH_RADIUS = 150.0


def read_of_scalar(filepath):
    text = filepath.read_text()
    m = re.search(r'nonuniform\s+List<scalar>\s*\n(\d+)\s*\n\(', text)
    if not m:
        return None
    n = int(m.group(1))
    start = text.index('(', m.start()) + 1
    end = text.index(')', start)
    return np.array([float(v) for v in text[start:end].split()[:n]])


def read_of_vector(filepath):
    text = filepath.read_text()
    m = re.search(r'nonuniform\s+List<vector>\s*\n(\d+)\s*\n\(', text)
    if not m:
        return None
    n = int(m.group(1))
    block = text[m.end():]
    coords = re.findall(r'\(\s*([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s*\)', block)
    return np.array([[float(a), float(b), float(c)] for a, b, c in coords[:n]])


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load Venkatraman data
    venk = pd.read_csv(VENK_CSV, comment='#')
    venk_speed = venk[venk['variable'] == 'wind_speed_ms']
    venk_tke = venk[venk['variable'] == 'TKE_m2s2']

    # Load our mesh + fields
    log.info("Loading mesh and fields...")
    cx = read_of_scalar(CASE_DIR / "0" / "Cx")
    cy = read_of_scalar(CASE_DIR / "0" / "Cy")
    cz = read_of_scalar(CASE_DIR / "0" / "Cz")
    cc = np.column_stack([cx, cy, cz])

    U = read_of_vector(CASE_DIR / "500" / "U")
    k_field = read_of_scalar(CASE_DIR / "500" / "k")
    speed = np.sqrt(U[:, 0]**2 + U[:, 1]**2)

    # Tower local coords (from pyproj)
    from pyproj import Transformer
    import zarr
    obs_store = zarr.open(str(ROOT / "data/raw/perdigao_obs.zarr"), mode='r')
    obs_coords = obs_store['coords']
    obs_sids = [s.decode() if isinstance(s, bytes) else s for s in np.array(obs_coords['site_id'])]
    obs_lats = np.array(obs_coords['lat'])
    obs_lons = np.array(obs_coords['lon'])
    obs_alts = np.array(obs_coords['altitude_m'])

    tr = Transformer.from_crs('EPSG:4326', 'EPSG:32629', always_xy=True)
    x0, y0 = tr.transform(-7.740, 39.716)

    tower_local = {}
    for i, sid in enumerate(obs_sids):
        xi, yi = tr.transform(obs_lons[i], obs_lats[i])
        tower_local[sid] = (xi - x0, yi - y0, obs_alts[i])

    # Build interpolators for Venkatraman towers
    tower_interp = {}
    for sid in TOWER_MAP:
        if sid not in tower_local:
            continue
        tx, ty, alt = tower_local[sid]
        dist = np.sqrt((cx - tx)**2 + (cy - ty)**2)
        mask = dist < SEARCH_RADIUS
        if mask.sum() < 10:
            continue
        pts = cc[mask]
        tower_interp[sid] = {
            "mask": mask, "alt": alt,
            "linear_s": LinearNDInterpolator(pts, speed[mask]),
            "nearest_s": NearestNDInterpolator(pts, speed[mask]),
            "linear_k": LinearNDInterpolator(pts, k_field[mask]),
            "nearest_k": NearestNDInterpolator(pts, k_field[mask]),
        }
    log.info("Built interpolators for %d towers", len(tower_interp))

    # Select towers with speed data in Venkatraman
    speed_towers = venk_speed['tower_id'].unique()
    tke_towers = venk_tke['tower_id'].unique() if len(venk_tke) > 0 else []
    plot_towers = [t for t in TOWER_MAP if t in speed_towers and t in tower_interp]

    # -----------------------------------------------------------------------
    # Plot: Wind speed profiles (our CFD + Venkatraman obs + their 5 models)
    # -----------------------------------------------------------------------
    n_towers = len(plot_towers)
    ncols = min(3, n_towers)
    nrows = (n_towers + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)

    for idx, sid in enumerate(plot_towers):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        info = tower_interp[sid]

        # Venkatraman obs + models
        vd = venk_speed[venk_speed['tower_id'] == sid]
        if len(vd) > 0:
            z = vd['z_m'].values
            ax.errorbar(vd['obs'], z, xerr=vd['obs_std'], fmt='ko', ms=5,
                       capsize=3, label='Obs (V2023)', zorder=10)
            ax.plot(vd['SF1'], z, 'b--', alpha=0.5, lw=1, label='V2023 SF1 (k-ε)')
            ax.plot(vd['SF3'], z, 'g--', alpha=0.5, lw=1, label='V2023 SF3 (canopy)')
            ax.plot(vd['BBSF1'], z, 'm--', alpha=0.5, lw=1, label='V2023 BBSF1')

        # Our CFD profile
        z_levels = np.arange(5, 300, 5)
        target = np.column_stack([
            np.full(len(z_levels), tower_local[sid][0]),
            np.full(len(z_levels), tower_local[sid][1]),
            z_levels + info["alt"],
        ])
        our_speed = info["linear_s"](target).ravel()
        nans = np.isnan(our_speed)
        if nans.any():
            our_speed[nans] = info["nearest_s"](target[nans]).ravel()

        ax.plot(our_speed, z_levels, 'r-', lw=2, label='DownscaleWind (WC z0)')

        ax.set_xlabel("Wind speed [m/s]")
        ax.set_ylabel("z AGL [m]")
        ax.set_ylim(0, 250)
        ax.set_title(TOWER_MAP[sid], fontsize=10)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7, loc='upper left')

    # Hide empty subplots
    for idx in range(n_towers, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle("Wind speed profiles: DownscaleWind (red) vs Venkatraman et al. 2023\n"
                 "Our ts00: 2017-05-04 18:00 (S 180°) | V2023: 22:00 (SW 231°)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "venkatraman_comparison.png", dpi=150, bbox_inches="tight")
    log.info("Saved venkatraman_comparison.png")

    # -----------------------------------------------------------------------
    # TKE comparison (if data available)
    # -----------------------------------------------------------------------
    tke_plot_towers = [t for t in TOWER_MAP if t in tke_towers and t in tower_interp]
    if tke_plot_towers:
        n_t = len(tke_plot_towers)
        nc = min(3, n_t)
        nr = (n_t + nc - 1) // nc
        fig2, axes2 = plt.subplots(nr, nc, figsize=(5 * nc, 5 * nr), squeeze=False)

        for idx, sid in enumerate(tke_plot_towers):
            row, col = divmod(idx, nc)
            ax = axes2[row, col]
            info = tower_interp[sid]

            vd = venk_tke[venk_tke['tower_id'] == sid]
            if len(vd) > 0:
                z = vd['z_m'].values
                ax.errorbar(vd['obs'], z, xerr=vd['obs_std'], fmt='ko', ms=5,
                           capsize=3, label='Obs', zorder=10)
                ax.plot(vd['SF1'], z, 'b--', alpha=0.5, lw=1, label='V2023 SF1')
                ax.plot(vd['SF3'], z, 'g--', alpha=0.5, lw=1, label='V2023 SF3')

            z_levels = np.arange(5, 300, 5)
            target = np.column_stack([
                np.full(len(z_levels), tower_local[sid][0]),
                np.full(len(z_levels), tower_local[sid][1]),
                z_levels + info["alt"],
            ])
            our_k = info["linear_k"](target).ravel()
            nans = np.isnan(our_k)
            if nans.any():
                our_k[nans] = info["nearest_k"](target[nans]).ravel()

            ax.plot(our_k, z_levels, 'r-', lw=2, label='DownscaleWind')
            ax.set_xlabel("TKE [m²/s²]")
            ax.set_ylabel("z AGL [m]")
            ax.set_ylim(0, 250)
            ax.set_title(TOWER_MAP[sid], fontsize=10)
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend(fontsize=7)

        for idx in range(n_t, nr * nc):
            row, col = divmod(idx, nc)
            axes2[row, col].set_visible(False)

        fig2.suptitle("TKE: DownscaleWind vs Venkatraman et al. 2023", fontsize=12, y=1.02)
        fig2.tight_layout()
        fig2.savefig(OUTPUT_DIR / "venkatraman_tke_comparison.png", dpi=150, bbox_inches="tight")
        log.info("Saved venkatraman_tke_comparison.png")


if __name__ == "__main__":
    main()
