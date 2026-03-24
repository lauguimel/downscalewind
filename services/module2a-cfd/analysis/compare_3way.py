"""
compare_3way.py — Compare z0-uniform vs WorldCover vs Canopy (3-way).

Same 5 timestamps, 3 roughness approaches, shared mesh.
Uses MeshInterpolator (Delaunay built once, reused for all 15 cases).

Usage:
    cd services/module2a-cfd
    python analysis/compare_3way.py
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
OUTPUT_DIR = ROOT / "data" / "validation" / "poc_tbm_3way"

# 3 study dirs
DIRS = {
    "z0=0.05":      ROOT / "data" / "cases" / "poc_tbm_z0_sensitivity",
    "z0=WC":        ROOT / "data" / "cases" / "poc_tbm_z0_sensitivity",
    "canopy":       ROOT / "data" / "cases" / "poc_tbm_canopy",
}

# Case ID mapping per timestamp index
CASE_IDS = {
    "z0=0.05":  ["u00", "u01", "u02", "u03", "u04"],
    "z0=WC":    ["w00", "w01", "w02", "w03", "w04"],
    "canopy":   ["c00", "c01", "c02", "c03", "c04"],
}

STYLES = {
    "z0=0.05":  {"color": "royalblue",  "ls": "-",  "lw": 1.5},
    "z0=WC":    {"color": "crimson",    "ls": "--", "lw": 1.5},
    "canopy":   {"color": "forestgreen","ls": "-.", "lw": 1.5},
}

TOWERS = {"T20": (-461, 955), "T25": (1025, 333), "T13": (-854, -444)}

Z_LEVELS = np.concatenate([
    np.arange(5, 200, 5),
    np.arange(200, 1000, 20),
    np.arange(1000, 5100, 100),
])
SEARCH_RADIUS = 150.0


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


class MeshInterpolator:
    def __init__(self, ref_case: Path):
        cx = read_of_scalar(ref_case / "0" / "Cx")
        cy = read_of_scalar(ref_case / "0" / "Cy")
        cz = read_of_scalar(ref_case / "0" / "Cz")
        self.cc = np.column_stack([cx, cy, cz])
        self.n = len(cx)
        self.towers = {}

        for tname, (tx, ty) in TOWERS.items():
            dist = np.sqrt((cx - tx)**2 + (cy - ty)**2)
            mask = dist < SEARCH_RADIUS
            pts = self.cc[mask]
            z_terr = pts[:, 2].min()
            z_tgt = Z_LEVELS[Z_LEVELS <= (pts[:, 2].max() - z_terr)]
            target = np.column_stack([np.full(len(z_tgt), tx),
                                       np.full(len(z_tgt), ty),
                                       z_tgt + z_terr])
            self.towers[tname] = {
                "mask": mask, "target": target, "z_agl": z_tgt, "z_terrain": z_terr,
                "linear": LinearNDInterpolator(pts, np.zeros(mask.sum())),
                "nearest": NearestNDInterpolator(pts, np.zeros(mask.sum())),
            }

    def interpolate(self, field: np.ndarray) -> dict[str, np.ndarray]:
        result = {}
        for tname, info in self.towers.items():
            vals = field[info["mask"]]
            info["linear"].values = vals.reshape(-1, 1)
            info["nearest"].values = vals.reshape(-1, 1)
            interp = info["linear"](info["target"]).ravel()
            nans = np.isnan(interp)
            if nans.any():
                interp[nans] = info["nearest"](info["target"][nans]).ravel()
            result[tname] = interp
        return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ref = ROOT / "data" / "cases" / "poc_tbm_z0_sensitivity" / "case_u00"
    log.info("Building mesh interpolator...")
    interp = MeshInterpolator(ref)

    # Load all 15 cases
    data = {}  # data[config][ts_idx] = {"speed": {tower: arr}, "k": {...}, "meta": {...}}
    for config in CASE_IDS:
        data[config] = {}
        for ts_idx, cid in enumerate(CASE_IDS[config]):
            case_dir = DIRS[config] / f"case_{cid}"
            time_dir = case_dir / "500"
            if not (time_dir / "U").exists():
                log.warning("%s/%s: no results", config, cid)
                continue
            U = read_of_vector(time_dir / "U")
            k = read_of_scalar(time_dir / "k")
            speed = np.sqrt(U[:, 0]**2 + U[:, 1]**2)
            with open(case_dir / "inflow.json") as f:
                inflow = json.load(f)
            data[config][ts_idx] = {
                "speed": interp.interpolate(speed),
                "k": interp.interpolate(k),
                "timestamp": inflow.get("timestamp", cid),
                "u_hub": inflow.get("u_hub", 0),
                "wind_dir": inflow.get("wind_dir", 0),
            }
    log.info("Loaded %d cases", sum(len(v) for v in data.values()))

    n_ts = 5
    tower_names = list(TOWERS.keys())

    # -----------------------------------------------------------------------
    # Plot 1: Speed profiles (5 rows × 3 towers)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(n_ts, len(tower_names),
                              figsize=(4 * len(tower_names), 3.2 * n_ts), squeeze=False)
    for row in range(n_ts):
        for col, tname in enumerate(tower_names):
            ax = axes[row, col]
            for config in CASE_IDS:
                if row not in data[config]:
                    continue
                d = data[config][row]
                z = interp.towers[tname]["z_agl"]
                s = d["speed"][tname]
                ax.plot(s, z, label=config, **STYLES[config])
            ax.set_ylim(0, 800)
            ax.set_xlabel("Speed [m/s]")
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel("z AGL [m]")
                ts = data["z0=0.05"].get(row, {}).get("timestamp", "?")
                u_h = data["z0=0.05"].get(row, {}).get("u_hub", 0)
                w_d = data["z0=0.05"].get(row, {}).get("wind_dir", 0)
                ax.annotate(f"{str(ts)[:16]}\n{u_h:.1f} m/s, {w_d:.0f}°",
                           xy=(0.02, 0.98), xycoords="axes fraction",
                           va="top", fontsize=7, color="gray")
            if row == 0:
                ax.set_title(tname, fontsize=11)
            if row == 0 and col == len(tower_names) - 1:
                ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Wind speed: z0=0.05 (blue) vs WorldCover (red) vs Canopy (green)", y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "3way_speed_profiles.png", dpi=150, bbox_inches="tight")
    log.info("Saved 3way_speed_profiles.png")

    # -----------------------------------------------------------------------
    # Plot 2: TKE profiles
    # -----------------------------------------------------------------------
    fig2, axes2 = plt.subplots(n_ts, len(tower_names),
                                figsize=(4 * len(tower_names), 3.2 * n_ts), squeeze=False)
    for row in range(n_ts):
        for col, tname in enumerate(tower_names):
            ax = axes2[row, col]
            for config in CASE_IDS:
                if row not in data[config]:
                    continue
                d = data[config][row]
                z = interp.towers[tname]["z_agl"]
                ax.plot(d["k"][tname], z, label=config, **STYLES[config])
            ax.set_ylim(0, 500)
            ax.set_xlabel("TKE [m²/s²]")
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel("z AGL [m]")
            if row == 0:
                ax.set_title(tname, fontsize=11)
            if row == 0 and col == len(tower_names) - 1:
                ax.legend(fontsize=7, loc="upper right")
    fig2.suptitle("TKE: z0=0.05 (blue) vs WorldCover (red) vs Canopy (green)", y=1.01)
    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / "3way_tke_profiles.png", dpi=150, bbox_inches="tight")
    log.info("Saved 3way_tke_profiles.png")

    # -----------------------------------------------------------------------
    # Summary: speed at 80m AGL
    # -----------------------------------------------------------------------
    print("\n=== 3-way comparison: speed at 80m AGL ===")
    print(f"{'TS':3s}", end="")
    for tname in tower_names:
        print(f"  {tname:>6s}_uni  {tname:>6s}_WC  {tname:>6s}_can", end="")
    print()
    for row in range(n_ts):
        ts = str(data["z0=0.05"].get(row, {}).get("timestamp", "?"))[:10]
        print(f"{ts}", end="")
        for tname in tower_names:
            z = interp.towers[tname]["z_agl"]
            i80 = np.argmin(np.abs(z - 80))
            vals = []
            for config in ["z0=0.05", "z0=WC", "canopy"]:
                if row in data[config]:
                    vals.append(data[config][row]["speed"][tname][i80])
                else:
                    vals.append(float('nan'))
            print(f"  {vals[0]:8.2f}  {vals[1]:8.2f}  {vals[2]:8.2f}", end="")
        print()


if __name__ == "__main__":
    main()
