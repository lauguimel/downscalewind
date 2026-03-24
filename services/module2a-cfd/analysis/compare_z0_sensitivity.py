"""
compare_z0_sensitivity.py — Compare z0-uniform vs z0-WorldCover cases pairwise.

Reads CFD results for paired cases, extracts vertical profiles at tower
locations via 3D griddata interpolation, and produces comparison figures.

The mesh is shared across all cases: Delaunay triangulation is computed once.

Usage:
    cd services/module2a-cfd
    python analysis/compare_z0_sensitivity.py
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]
CASES_DIR = ROOT / "data" / "cases" / "poc_tbm_z0_sensitivity"
OUTPUT_DIR = ROOT / "data" / "validation" / "poc_tbm_z0_sensitivity"

PAIRS = [
    ("u00", "w00"),
    ("u01", "w01"),
    ("u02", "w02"),
    ("u03", "w03"),
    ("u04", "w04"),
]

# Tower locations (local coords from perdigao.yaml)
TOWERS = {"T20": (-461, 955), "T25": (1025, 333), "T13": (-854, -444)}

# Interpolation target heights (AGL)
Z_LEVELS = np.concatenate([
    np.arange(5, 200, 5),
    np.arange(200, 1000, 20),
    np.arange(1000, 5100, 100),
])

SEARCH_RADIUS = 150.0  # m


# -----------------------------------------------------------------------
# OF field readers (no fluidfoam dependency)
# -----------------------------------------------------------------------

def read_of_scalar(filepath: Path) -> np.ndarray:
    """Parse OpenFOAM volScalarField → (N,) array."""
    text = filepath.read_text()
    m = re.search(r'nonuniform\s+List<scalar>\s*\n(\d+)\s*\n\(', text)
    if not m:
        # Try uniform
        m2 = re.search(r'internalField\s+uniform\s+([\d.eE+\-]+)', text)
        if m2:
            return None  # uniform — caller handles
        return None
    n = int(m.group(1))
    start = text.index('(', m.start()) + 1
    end = text.index(')', start)
    return np.array([float(v) for v in text[start:end].split()[:n]])


def read_of_vector(filepath: Path) -> np.ndarray:
    """Parse OpenFOAM volVectorField → (N, 3) array."""
    text = filepath.read_text()
    m = re.search(r'nonuniform\s+List<vector>\s*\n(\d+)\s*\n\(', text)
    if not m:
        return None
    n = int(m.group(1))
    block = text[m.end():]
    coords = re.findall(r'\(\s*([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s*\)', block)
    return np.array([[float(a), float(b), float(c)] for a, b, c in coords[:n]])


# -----------------------------------------------------------------------
# Shared mesh + interpolator
# -----------------------------------------------------------------------

class MeshInterpolator:
    """Pre-computed 3D interpolation from shared mesh to tower profiles."""

    def __init__(self, ref_case: Path, towers: dict, search_radius: float, z_levels: np.ndarray):
        log.info("Loading mesh from %s", ref_case.name)
        t0 = time.time()

        cx = read_of_scalar(ref_case / "0" / "Cx")
        cy = read_of_scalar(ref_case / "0" / "Cy")
        cz = read_of_scalar(ref_case / "0" / "Cz")
        self.cc = np.column_stack([cx, cy, cz])
        self.n_cells = len(cx)
        log.info("  %d cells loaded in %.1fs", self.n_cells, time.time() - t0)

        # Pre-compute masks + interpolators per tower
        self.tower_interp = {}
        for tname, (tx, ty) in towers.items():
            dist_h = np.sqrt((cx - tx)**2 + (cy - ty)**2)
            mask = dist_h < search_radius
            n_sel = mask.sum()
            if n_sel < 10:
                log.warning("  %s: only %d cells within %.0fm — skipped", tname, n_sel, search_radius)
                continue

            pts = self.cc[mask]
            z_terrain = pts[:, 2].min()

            # Target points: vertical line at tower location
            z_tgt = z_levels[z_levels <= (pts[:, 2].max() - z_terrain)]
            target = np.column_stack([
                np.full(len(z_tgt), tx),
                np.full(len(z_tgt), ty),
                z_tgt + z_terrain,
            ])

            log.info("  %s: %d cells, z_terrain=%.0fm, %d levels, building Delaunay...",
                     tname, n_sel, z_terrain, len(z_tgt))
            t1 = time.time()

            # Build interpolators (reused for all fields)
            linear = LinearNDInterpolator(pts, np.zeros(n_sel))
            nearest = NearestNDInterpolator(pts, np.zeros(n_sel))

            self.tower_interp[tname] = {
                "mask": mask,
                "target": target,
                "z_agl": z_tgt,
                "z_terrain": z_terrain,
                "linear": linear,
                "nearest": nearest,
                "n_cells": n_sel,
            }
            log.info("    Delaunay built in %.1fs", time.time() - t1)

    def interpolate(self, field: np.ndarray) -> dict[str, np.ndarray]:
        """Interpolate a field at all tower locations. Returns {tower: values}."""
        result = {}
        for tname, info in self.tower_interp.items():
            vals = field[info["mask"]]

            # Update interpolator values (reuse Delaunay structure)
            info["linear"].values = vals.reshape(-1, 1)
            info["nearest"].values = vals.reshape(-1, 1)

            interp = info["linear"](info["target"]).ravel()

            # Fill NaN (outside convex hull) with nearest
            nans = np.isnan(interp)
            if nans.any():
                interp[nans] = info["nearest"](info["target"][nans]).ravel()

            result[tname] = interp
        return result


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Use first case as mesh reference
    ref_case = CASES_DIR / "case_u00"
    interp = MeshInterpolator(ref_case, TOWERS, SEARCH_RADIUS, Z_LEVELS)

    # Collect profiles for all cases
    all_profiles = {}
    for uid, wid in PAIRS:
        for cid in (uid, wid):
            case_dir = CASES_DIR / f"case_{cid}"
            time_dir = case_dir / "500"
            if not (time_dir / "U").exists():
                log.warning("%s: no results at 500/", cid)
                continue

            log.info("Reading fields for %s", cid)
            U = read_of_vector(time_dir / "U")
            k = read_of_scalar(time_dir / "k")
            T = read_of_scalar(time_dir / "T")

            speed = np.sqrt(U[:, 0]**2 + U[:, 1]**2)

            profiles = {
                "speed": interp.interpolate(speed),
                "k": interp.interpolate(k),
            }
            if T is not None:
                profiles["T"] = interp.interpolate(T)

            # Read inflow metadata
            with open(case_dir / "inflow.json") as f:
                inflow = json.load(f)

            all_profiles[cid] = {
                "profiles": profiles,
                "z_agl": {t: info["z_agl"] for t, info in interp.tower_interp.items()},
                "timestamp": inflow.get("timestamp", cid),
                "u_hub": inflow.get("u_hub", 0),
                "wind_dir": inflow.get("wind_dir", 0),
            }

    # -----------------------------------------------------------------------
    # Plot 1: Profile comparison (5 rows × 3 towers × 2 variables)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(len(PAIRS), len(TOWERS), figsize=(4 * len(TOWERS), 3.5 * len(PAIRS)),
                              squeeze=False)

    for row, (uid, wid) in enumerate(PAIRS):
        if uid not in all_profiles or wid not in all_profiles:
            continue

        ts = all_profiles[uid]["timestamp"]
        u_hub = all_profiles[uid]["u_hub"]
        w_dir = all_profiles[uid]["wind_dir"]

        for col, tname in enumerate(TOWERS):
            ax = axes[row, col]
            z_u = all_profiles[uid]["z_agl"][tname]
            z_w = all_profiles[wid]["z_agl"][tname]

            s_u = all_profiles[uid]["profiles"]["speed"][tname]
            s_w = all_profiles[wid]["profiles"]["speed"][tname]

            ax.plot(s_u, z_u, 'b-', lw=1.5, label="z0=0.05")
            ax.plot(s_w, z_w, 'r--', lw=1.5, label="z0=WC")

            ax.set_ylim(0, 1000)
            ax.set_xlabel("Speed [m/s]")
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel("z AGL [m]")
                ax.annotate(f"{ts[:16]}\n{u_hub:.1f} m/s, {w_dir:.0f}°",
                           xy=(0.02, 0.98), xycoords="axes fraction",
                           va="top", fontsize=7, color="gray")
            if row == 0:
                ax.set_title(tname, fontsize=11)
            if row == 0 and col == len(TOWERS) - 1:
                ax.legend(fontsize=8)

    fig.suptitle("Wind speed profiles: z0 uniform (blue) vs WorldCover (red)", y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "z0_speed_profiles.png", dpi=150, bbox_inches="tight")
    log.info("Saved z0_speed_profiles.png")

    # -----------------------------------------------------------------------
    # Plot 2: TKE profiles
    # -----------------------------------------------------------------------
    fig2, axes2 = plt.subplots(len(PAIRS), len(TOWERS), figsize=(4 * len(TOWERS), 3.5 * len(PAIRS)),
                                squeeze=False)

    for row, (uid, wid) in enumerate(PAIRS):
        if uid not in all_profiles or wid not in all_profiles:
            continue
        for col, tname in enumerate(TOWERS):
            ax = axes2[row, col]
            z_u = all_profiles[uid]["z_agl"][tname]
            z_w = all_profiles[wid]["z_agl"][tname]

            k_u = all_profiles[uid]["profiles"]["k"][tname]
            k_w = all_profiles[wid]["profiles"]["k"][tname]

            ax.plot(k_u, z_u, 'b-', lw=1.5, label="z0=0.05")
            ax.plot(k_w, z_w, 'r--', lw=1.5, label="z0=WC")

            ax.set_ylim(0, 500)
            ax.set_xlabel("TKE [m²/s²]")
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel("z AGL [m]")
            if row == 0:
                ax.set_title(tname, fontsize=11)
            if row == 0 and col == len(TOWERS) - 1:
                ax.legend(fontsize=8)

    fig2.suptitle("TKE profiles: z0 uniform (blue) vs WorldCover (red)", y=1.01)
    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / "z0_tke_profiles.png", dpi=150, bbox_inches="tight")
    log.info("Saved z0_tke_profiles.png")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n=== z0 Sensitivity: speed at 80m AGL ===")
    print(f"{'Pair':10s}", end="")
    for tname in TOWERS:
        print(f"  {tname:>8s}_uni  {tname:>8s}_WC   Δ", end="")
    print()

    for uid, wid in PAIRS:
        if uid not in all_profiles or wid not in all_profiles:
            continue
        print(f"{uid}/{wid:5s}", end="")
        for tname in TOWERS:
            z = all_profiles[uid]["z_agl"][tname]
            i80 = np.argmin(np.abs(z - 80))
            su = all_profiles[uid]["profiles"]["speed"][tname][i80]
            sw = all_profiles[wid]["profiles"]["speed"][tname][i80]
            print(f"  {su:8.2f}     {sw:8.2f}  {sw - su:+.2f}", end="")
        print()


if __name__ == "__main__":
    main()
