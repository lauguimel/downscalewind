"""
validate_cfd_vs_obs.py — Compare CFD results with Perdigão tower observations.

For each of the 25 timestamps, interpolates CFD wind speed at all 48 tower
locations and heights, then compares with 10-min averaged observations.

Uses MeshInterpolator with shared Delaunay (built once for all towers).

Usage:
    cd services/module2a-cfd
    python analysis/validate_cfd_vs_obs.py
"""
from __future__ import annotations

import json
import logging
import re
import time as time_mod
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]
CASES_DIR = ROOT / "data" / "cases" / "poc_tbm_25ts_wc"
OBS_ZARR = ROOT / "data" / "raw" / "perdigao_obs.zarr"
OUTPUT_DIR = ROOT / "data" / "validation" / "poc_tbm_25ts_wc"
TIMESTAMPS_CSV = ROOT / "data" / "campaign" / "sf_poc" / "timestamps.csv"

SEARCH_RADIUS = 200.0


def read_of_scalar(filepath: Path) -> np.ndarray | None:
    text = filepath.read_text()
    m = re.search(r'nonuniform\s+List<scalar>\s*\n(\d+)\s*\n\(', text)
    if not m:
        return None
    n = int(m.group(1))
    start = text.index('(', m.start()) + 1
    end = text.index(')', start)
    return np.array([float(v) for v in text[start:end].split()[:n]])


def read_of_vector(filepath: Path) -> np.ndarray | None:
    text = filepath.read_text()
    m = re.search(r'nonuniform\s+List<vector>\s*\n(\d+)\s*\n\(', text)
    if not m:
        return None
    n = int(m.group(1))
    block = text[m.end():]
    coords = re.findall(r'\(\s*([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s*\)', block)
    return np.array([[float(a), float(b), float(c)] for a, b, c in coords[:n]])


def load_obs(obs_path: Path) -> dict:
    """Load Perdigão observations from zarr."""
    import zarr
    store = zarr.open(str(obs_path), mode='r')
    coords = store['coords']
    return {
        'time': pd.to_datetime(np.array(coords['time'])),
        'site_id': [s.decode() if isinstance(s, bytes) else s
                    for s in np.array(coords['site_id'])],
        'height_m': np.array(coords['height_m']),
        'lat': np.array(coords['lat']),
        'lon': np.array(coords['lon']),
        'altitude_m': np.array(coords['altitude_m']),
        'u': np.array(store['sites']['u']),  # (time, site, height)
        'v': np.array(store['sites']['v']),
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load observations
    log.info("Loading observations...")
    obs = load_obs(OBS_ZARR)
    obs_speed = np.sqrt(obs['u']**2 + obs['v']**2)
    n_sites = len(obs['site_id'])
    n_heights = len(obs['height_m'])
    log.info("  %d sites, %d heights, %d timesteps", n_sites, n_heights, len(obs['time']))

    # Tower local coordinates
    from pyproj import Transformer
    tr = Transformer.from_crs('EPSG:4326', 'EPSG:32629', always_xy=True)
    x0, y0 = tr.transform(-7.740, 39.716)
    tower_xy = {}
    for i, sid in enumerate(obs['site_id']):
        xi, yi = tr.transform(obs['lon'][i], obs['lat'][i])
        tower_xy[sid] = (xi - x0, yi - y0, obs['altitude_m'][i])

    # Load mesh (shared across all cases)
    ref_case = CASES_DIR / "case_ts00"
    log.info("Loading mesh...")
    cx = read_of_scalar(ref_case / "0" / "Cx")
    cy = read_of_scalar(ref_case / "0" / "Cy")
    cz = read_of_scalar(ref_case / "0" / "Cz")
    cc = np.column_stack([cx, cy, cz])
    log.info("  %d cells", len(cx))

    # Build interpolators per tower (all heights in one shot)
    log.info("Building interpolators for %d towers...", n_sites)
    tower_interp = {}
    for sid, (tx, ty, alt) in tower_xy.items():
        dist = np.sqrt((cx - tx)**2 + (cy - ty)**2)
        mask = dist < SEARCH_RADIUS
        if mask.sum() < 10:
            continue
        pts = cc[mask]
        z_terrain = alt  # use known altitude
        # Target: all obs heights AGL at this tower
        z_abs = np.array([alt + h for h in obs['height_m']])
        target = np.column_stack([
            np.full(n_heights, tx),
            np.full(n_heights, ty),
            z_abs,
        ])
        tower_interp[sid] = {
            'mask': mask,
            'target': target,
            'linear': LinearNDInterpolator(pts, np.zeros(mask.sum())),
            'nearest': NearestNDInterpolator(pts, np.zeros(mask.sum())),
        }
    log.info("  %d towers with interpolators", len(tower_interp))

    # Load timestamps
    ts_df = pd.read_csv(TIMESTAMPS_CSV)

    # Process each case
    all_pairs = []  # (obs_speed, cfd_speed, site, height, timestamp)

    for i, row in ts_df.iterrows():
        dt = pd.Timestamp(row['datetime'])
        case_id = f"ts{i:02d}"
        case_dir = CASES_DIR / f"case_{case_id}" / "500"

        if not (case_dir / "U").exists():
            continue

        # Find nearest obs time
        time_diff = abs(obs['time'] - dt)
        t_idx = time_diff.argmin()
        if time_diff[t_idx].total_seconds() > 600:  # >10 min → no obs
            continue

        # Read CFD field
        U = read_of_vector(case_dir / "U")
        if U is None:
            continue
        cfd_speed = np.sqrt(U[:, 0]**2 + U[:, 1]**2)

        # Interpolate at all towers
        for s_idx, sid in enumerate(obs['site_id']):
            if sid not in tower_interp:
                continue
            info = tower_interp[sid]
            vals = cfd_speed[info['mask']]
            info['linear'].values = vals.reshape(-1, 1)
            info['nearest'].values = vals.reshape(-1, 1)
            cfd_at_tower = info['linear'](info['target']).ravel()
            nans = np.isnan(cfd_at_tower)
            if nans.any():
                cfd_at_tower[nans] = info['nearest'](info['target'][nans]).ravel()

            for h_idx in range(n_heights):
                obs_val = obs_speed[t_idx, s_idx, h_idx]
                cfd_val = cfd_at_tower[h_idx]
                if np.isfinite(obs_val) and np.isfinite(cfd_val) and obs_val > 0.1:
                    all_pairs.append({
                        'obs': obs_val, 'cfd': cfd_val,
                        'site': sid, 'height': obs['height_m'][h_idx],
                        'timestamp': str(dt), 'case': case_id,
                    })

        log.info("  %s (%s): %d valid pairs", case_id, str(dt)[:16],
                 sum(1 for p in all_pairs if p['case'] == case_id))

    df = pd.DataFrame(all_pairs)
    log.info("Total: %d obs-CFD pairs across %d timestamps", len(df), df['case'].nunique())

    if len(df) == 0:
        log.error("No valid pairs — check obs/CFD overlap")
        return

    # Global metrics
    bias = (df['cfd'] - df['obs']).mean()
    rmse = np.sqrt(((df['cfd'] - df['obs'])**2).mean())
    mae = (df['cfd'] - df['obs']).abs().mean()
    r2 = np.corrcoef(df['obs'], df['cfd'])[0, 1]**2
    print(f"\n=== CFD vs Obs (all pairs, N={len(df)}) ===")
    print(f"  Bias: {bias:+.2f} m/s")
    print(f"  RMSE: {rmse:.2f} m/s")
    print(f"  MAE:  {mae:.2f} m/s")
    print(f"  R²:   {r2:.3f}")

    # Per-height metrics
    print(f"\n{'Height':>8s} {'N':>6s} {'Bias':>8s} {'RMSE':>8s} {'MAE':>8s}")
    for h in sorted(df['height'].unique()):
        dh = df[df['height'] == h]
        b = (dh['cfd'] - dh['obs']).mean()
        r = np.sqrt(((dh['cfd'] - dh['obs'])**2).mean())
        m = (dh['cfd'] - dh['obs']).abs().mean()
        print(f"{h:8.0f} {len(dh):6d} {b:+8.2f} {r:8.2f} {m:8.2f}")

    # Save CSV
    df.to_csv(OUTPUT_DIR / "cfd_vs_obs_pairs.csv", index=False)

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------

    # 1. Scatter plot (all pairs)
    fig, ax = plt.subplots(figsize=(7, 7))
    vmax = max(df['obs'].max(), df['cfd'].max()) * 1.1
    ax.scatter(df['obs'], df['cfd'], s=3, alpha=0.3, c='steelblue')
    ax.plot([0, vmax], [0, vmax], 'k--', lw=1)
    ax.set_xlabel("Observed wind speed [m/s]")
    ax.set_ylabel("CFD wind speed [m/s]")
    ax.set_xlim(0, vmax)
    ax.set_ylim(0, vmax)
    ax.set_aspect('equal')
    ax.set_title(f"CFD vs Obs — {len(df)} pairs, {df['case'].nunique()} timestamps\n"
                 f"Bias={bias:+.2f}, RMSE={rmse:.2f}, MAE={mae:.2f} m/s, R²={r2:.3f}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "scatter_cfd_vs_obs.png", dpi=150)
    log.info("Saved scatter_cfd_vs_obs.png")

    # 2. RMSE by height
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    heights_sorted = sorted(df['height'].unique())
    rmse_h = [np.sqrt(((df[df['height']==h]['cfd'] - df[df['height']==h]['obs'])**2).mean())
              for h in heights_sorted]
    bias_h = [(df[df['height']==h]['cfd'] - df[df['height']==h]['obs']).mean()
              for h in heights_sorted]
    ax2.barh(range(len(heights_sorted)), rmse_h, color='steelblue', alpha=0.7, label='RMSE')
    ax2.barh(range(len(heights_sorted)), bias_h, color='coral', alpha=0.7, label='Bias')
    ax2.set_yticks(range(len(heights_sorted)))
    ax2.set_yticklabels([f"{h:.0f}m" for h in heights_sorted])
    ax2.set_xlabel("m/s")
    ax2.set_title("RMSE and Bias by observation height")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')
    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / "rmse_by_height.png", dpi=150)
    log.info("Saved rmse_by_height.png")

    # 3. RMSE by timestamp
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ts_metrics = []
    for case in sorted(df['case'].unique()):
        dc = df[df['case'] == case]
        ts_metrics.append({
            'case': case,
            'rmse': np.sqrt(((dc['cfd'] - dc['obs'])**2).mean()),
            'bias': (dc['cfd'] - dc['obs']).mean(),
            'n': len(dc),
        })
    ts_m = pd.DataFrame(ts_metrics)
    x = range(len(ts_m))
    ax3.bar(x, ts_m['rmse'], color='steelblue', alpha=0.7, label='RMSE')
    ax3.bar(x, ts_m['bias'], color='coral', alpha=0.7, label='Bias')
    ax3.set_xticks(x)
    ax3.set_xticklabels(ts_m['case'], rotation=45, fontsize=8)
    ax3.set_ylabel("m/s")
    ax3.set_title("RMSE and Bias per timestamp")
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    fig3.tight_layout()
    fig3.savefig(OUTPUT_DIR / "rmse_by_timestamp.png", dpi=150)
    log.info("Saved rmse_by_timestamp.png")


if __name__ == "__main__":
    main()
