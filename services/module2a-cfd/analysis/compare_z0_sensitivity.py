"""
compare_z0_sensitivity.py — Compare z0-uniform vs z0-WorldCover cases pairwise.

Reads CFD results (500/U, 500/k) for paired cases (u00/w00, u01/w01, ...),
extracts vertical profiles at tower locations, and produces comparison figures.

Usage:
    python analysis/compare_z0_sensitivity.py
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]
CASES_DIR = ROOT / "data" / "cases" / "poc_tbm_z0_sensitivity"
OUTPUT_DIR = ROOT / "data" / "validation" / "poc_tbm_z0_sensitivity"

PAIRS = [("u00", "w00"), ("u01", "w01"), ("u02", "w02"), ("u03", "w03"), ("u04", "w04")]


def read_of_vector_field(filepath: Path) -> np.ndarray:
    """Parse OpenFOAM volVectorField → (N, 3) array."""
    text = filepath.read_text()
    match = re.search(r'internalField\s+nonuniform\s+List<vector>\s*\n(\d+)\s*\n\(', text)
    if not match:
        return np.array([])
    n = int(match.group(1))
    block = text[match.end():]
    coords = re.findall(r'\(\s*([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s*\)', block)
    return np.array([[float(x), float(y), float(z)] for x, y, z in coords[:n]])


def read_of_scalar_field(filepath: Path) -> np.ndarray:
    """Parse OpenFOAM volScalarField → (N,) array."""
    text = filepath.read_text()
    match = re.search(r'internalField\s+nonuniform\s+List<scalar>\s*\n(\d+)\s*\n\(', text)
    if not match:
        return np.array([])
    n = int(match.group(1))
    start = match.end()
    end = text.index(')', start)
    values = text[start:end].split()
    return np.array([float(v) for v in values[:n]])


def read_cell_centres(case_dir: Path) -> np.ndarray:
    """Read cell centres (Cx, Cy, Cz) → (N, 3)."""
    cx = read_of_scalar_field(case_dir / "0" / "Cx")
    cy = read_of_scalar_field(case_dir / "0" / "Cy")
    cz = read_of_scalar_field(case_dir / "0" / "Cz")
    return np.column_stack([cx, cy, cz])


def extract_profile(cc: np.ndarray, field: np.ndarray, x0: float, y0: float,
                     radius: float = 200.0) -> tuple[np.ndarray, np.ndarray]:
    """Extract a vertical profile at (x0, y0) within radius."""
    dist = np.sqrt((cc[:, 0] - x0) ** 2 + (cc[:, 1] - y0) ** 2)
    mask = dist < radius
    z = cc[mask, 2]
    vals = field[mask]
    order = np.argsort(z)
    return z[order], vals[order]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Tower locations (local coords) — from perdigao.yaml
    towers = {"T20": (-461, 955), "T25": (1025, 333), "T13": (-854, -444)}

    fig_profiles, axes_p = plt.subplots(len(PAIRS), len(towers), figsize=(4 * len(towers), 4 * len(PAIRS)),
                                         squeeze=False, sharex=False)

    summary = []

    for row, (uid, wid) in enumerate(PAIRS):
        u_dir = CASES_DIR / f"case_{uid}"
        w_dir = CASES_DIR / f"case_{wid}"

        if not (u_dir / "500" / "U").exists() or not (w_dir / "500" / "U").exists():
            log.warning("Missing results for %s/%s", uid, wid)
            continue

        # Read fields
        cc = read_cell_centres(u_dir)  # same mesh for both
        U_u = read_of_vector_field(u_dir / "500" / "U")
        U_w = read_of_vector_field(w_dir / "500" / "U")
        speed_u = np.linalg.norm(U_u[:, :2], axis=1)  # horizontal wind speed
        speed_w = np.linalg.norm(U_w[:, :2], axis=1)

        # Read inflow info
        with open(u_dir / "inflow.json") as f:
            inflow = json.load(f)
        ts = inflow.get("timestamp", uid)

        # Profile comparison at each tower
        for col, (tname, (tx, ty)) in enumerate(towers.items()):
            z_u, s_u = extract_profile(cc, speed_u, tx, ty, radius=250)
            z_w, s_w = extract_profile(cc, speed_w, tx, ty, radius=250)

            ax = axes_p[row, col]
            ax.plot(s_u, z_u, 'b-', label=f"z0=0.05 ({uid})", linewidth=1.5)
            ax.plot(s_w, z_w, 'r--', label=f"z0=WC ({wid})", linewidth=1.5)
            ax.set_ylabel("z [m]" if col == 0 else "")
            ax.set_xlabel("Wind speed [m/s]")
            if row == 0:
                ax.set_title(tname)
            if col == 0:
                ax.annotate(f"{ts[:16]}", xy=(0.02, 0.98), xycoords="axes fraction",
                           va="top", fontsize=8, color="gray")
            ax.legend(fontsize=7, loc="upper right")
            ax.set_ylim(0, 2000)
            ax.grid(True, alpha=0.3)

        # Global stats
        diff = speed_w - speed_u
        summary.append({
            "pair": f"{uid}/{wid}",
            "timestamp": ts[:16],
            "mean_diff": float(diff.mean()),
            "rms_diff": float(np.sqrt((diff ** 2).mean())),
            "max_abs_diff": float(np.abs(diff).max()),
            "u_mean": float(speed_u.mean()),
            "w_mean": float(speed_w.mean()),
        })

    fig_profiles.suptitle("z0 sensitivity: uniform (blue) vs WorldCover (red)", fontsize=14, y=1.01)
    fig_profiles.tight_layout()
    fig_profiles.savefig(OUTPUT_DIR / "z0_profiles_comparison.png", dpi=150, bbox_inches="tight")
    log.info("Saved %s", OUTPUT_DIR / "z0_profiles_comparison.png")

    # Summary table
    fig_tab, ax_tab = plt.subplots(figsize=(10, 3))
    ax_tab.axis("off")
    headers = ["Pair", "Timestamp", "ΔU mean [m/s]", "ΔU rms [m/s]", "U_uniform", "U_WorldCover"]
    rows = [[s["pair"], s["timestamp"], f"{s['mean_diff']:+.3f}", f"{s['rms_diff']:.3f}",
             f"{s['u_mean']:.2f}", f"{s['w_mean']:.2f}"] for s in summary]
    table = ax_tab.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    fig_tab.tight_layout()
    fig_tab.savefig(OUTPUT_DIR / "z0_summary_table.png", dpi=150, bbox_inches="tight")
    log.info("Saved %s", OUTPUT_DIR / "z0_summary_table.png")

    # Print summary
    print("\n=== z0 Sensitivity Summary ===")
    print(f"{'Pair':8s} {'Timestamp':16s} {'ΔU mean':>10s} {'ΔU rms':>10s} {'U_unif':>8s} {'U_WC':>8s}")
    for s in summary:
        print(f"{s['pair']:8s} {s['timestamp']:16s} {s['mean_diff']:+10.3f} {s['rms_diff']:10.3f} "
              f"{s['u_mean']:8.2f} {s['w_mean']:8.2f}")


if __name__ == "__main__":
    main()
