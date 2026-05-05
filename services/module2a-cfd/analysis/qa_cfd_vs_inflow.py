"""
qa_cfd_vs_inflow.py — Coherence QA for complex_terrain_v1 CFD outputs.

For each exported case (grid.zarr + inflow.json), check:

  Gate A — Max principle on passive scalars (T, q):
    T_min_cfd >= T_min_inflow_profile
    T_max_cfd <= T_max_inflow_profile
    Same for q. Violations indicate numerical overshoot or BC bug.

  Gate B — Mass-conservation lower bound on |U|:
    mean(|U|_cfd over domain volume) >= mean(|U|_inflow over corresponding heights)
    A drop of more than tol (e.g. 20%) flags excessive momentum loss.

  Gate C — BC sanity at upstream boundary:
    Mean wind speed on upstream face matches inflow profile within tol.

  Gate D — Flow direction at top of domain ~ inflow.wind_dir.

Output: 1 CSV row per case (qa_cfd_vs_inflow.csv) and a markdown summary.

Usage (on Aqua):
    python qa_cfd_vs_inflow.py \\
        --cases-dir /scratch/maitreje/dsw/complex_terrain_v1/cases \\
        --output qa_cfd_vs_inflow.csv \\
        --summary qa_cfd_vs_inflow.md
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import click
import numpy as np


def load_inflow_profile(case_dir: Path) -> dict | None:
    p = case_dir / "inflow.json"
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    keys = ("z_levels", "ux_profile", "uy_profile", "T_profile")
    if not all(k in d for k in keys):
        return None
    out = {
        "z": np.asarray(d["z_levels"], dtype=float),
        "u": np.asarray(d["ux_profile"], dtype=float),
        "v": np.asarray(d["uy_profile"], dtype=float),
        "T": np.asarray(d["T_profile"], dtype=float),
    }
    out["q"] = np.asarray(d["q_profile"], dtype=float) if "q_profile" in d else None
    out["spd"] = np.hypot(out["u"], out["v"])
    out["wind_dir"] = float(d.get("wind_dir", float("nan")))
    out["u_hub"] = float(d.get("u_hub", float("nan")))
    return out


def gate_max_principle(cfd_field: np.ndarray, inflow_vals: np.ndarray,
                       tol_rel: float = 0.0):
    """Check max principle: cfd values must lie within [min, max] of inflow.

    Returns dict with passed flag and the worst violations (absolute and rel).
    """
    inflow_min = float(np.nanmin(inflow_vals))
    inflow_max = float(np.nanmax(inflow_vals))
    cfd_min = float(np.nanmin(cfd_field))
    cfd_max = float(np.nanmax(cfd_field))
    span = max(inflow_max - inflow_min, 1e-9)
    under = max(0.0, inflow_min - cfd_min)        # how far below inflow_min
    over = max(0.0, cfd_max - inflow_max)         # how far above inflow_max
    under_rel = under / span
    over_rel = over / span
    passed = (under_rel <= tol_rel) and (over_rel <= tol_rel)
    return {
        "passed": passed,
        "inflow_min": inflow_min, "inflow_max": inflow_max,
        "cfd_min": cfd_min, "cfd_max": cfd_max,
        "under": under, "over": over,
        "under_rel": under_rel, "over_rel": over_rel,
    }


def gate_mass_lower_bound(spd_cfd_3d: np.ndarray,
                          z_agl_cfd: np.ndarray,
                          inflow: dict,
                          tol_rel: float = 0.20):
    """Check that domain-averaged |U| is not significantly below inflow average.

    We compare:
      - vol_mean_cfd = volume average of |U|_cfd over the whole domain (128x128xnz)
      - inflow_mean  = vertical average of |U|_inflow over the cfd z range
    """
    # Volume average of speed
    vol_mean_cfd = float(np.nanmean(spd_cfd_3d))
    # Inflow vertical mean over cfd z range (linear interp on common z grid)
    z_lo, z_hi = float(z_agl_cfd.min()), float(z_agl_cfd.max())
    z_inflow = inflow["z"]
    spd_inflow = inflow["spd"]
    mask = (z_inflow >= z_lo) & (z_inflow <= z_hi)
    if mask.sum() < 2:
        # interp inflow onto cfd z to be safe
        spd_on_cfd_z = np.interp(z_agl_cfd, z_inflow, spd_inflow)
        inflow_mean = float(np.mean(spd_on_cfd_z))
    else:
        inflow_mean = float(np.mean(spd_inflow[mask]))
    deficit = inflow_mean - vol_mean_cfd
    deficit_rel = deficit / max(inflow_mean, 1e-6)
    passed = deficit_rel <= tol_rel
    return {
        "passed": passed,
        "vol_mean_cfd": vol_mean_cfd,
        "inflow_mean": inflow_mean,
        "deficit": deficit,
        "deficit_rel": deficit_rel,
    }


def gate_bc_upstream(U_cfd_3d: np.ndarray,
                     z_agl_cfd: np.ndarray,
                     inflow: dict,
                     tol_ms: float = 1.0):
    """Inflow boundary face check: mean speed on upstream column matches inflow.

    Picks the upstream face based on inflow wind direction.
    Returns the mean abs error on the speed profile (m/s).
    """
    wd = inflow["wind_dir"]
    if not np.isfinite(wd):
        return {"passed": False, "detail": "wind_dir missing", "mae_ms": float("nan")}
    # wind_dir = direction the wind comes FROM, in meteorological convention
    # Convert to math angle for upstream identification
    # We pick the boundary cells on the upstream side. With grid (ny, nx, nz, 3):
    # axis 0 = y (north), axis 1 = x (east).
    ny, nx, nz, _ = U_cfd_3d.shape
    # unit wind vector (incoming) — wd in degrees, 0=N, 90=E, etc.
    wd_rad = np.deg2rad(wd)
    # incoming unit vector (where wind blows TO)
    ix = -np.sin(wd_rad)
    iy = -np.cos(wd_rad)
    # upstream face: the side opposite to incoming direction
    if abs(ix) >= abs(iy):
        # x dominant — upstream is x=0 if ix>0 else x=nx-1
        if ix > 0:
            face_uv = U_cfd_3d[:, 0, :, :2]
        else:
            face_uv = U_cfd_3d[:, -1, :, :2]
    else:
        if iy > 0:
            face_uv = U_cfd_3d[0, :, :, :2]
        else:
            face_uv = U_cfd_3d[-1, :, :, :2]
    # face_uv shape: (n_lateral, nz, 2)
    spd_face = np.hypot(face_uv[..., 0], face_uv[..., 1])  # (n_lat, nz)
    mean_face_z = np.nanmean(spd_face, axis=0)             # (nz,)
    # inflow on cfd z grid
    spd_in_z = np.interp(z_agl_cfd, inflow["z"], inflow["spd"])
    mae = float(np.nanmean(np.abs(mean_face_z - spd_in_z)))
    passed = mae <= tol_ms
    return {"passed": passed, "mae_ms": mae,
            "face_mean_top": float(mean_face_z[-1]),
            "inflow_top": float(spd_in_z[-1])}


def gate_topdir(U_cfd_3d: np.ndarray, inflow: dict, tol_deg: float = 20.0):
    """Direction at top of domain should match inflow.wind_dir within tol."""
    top = U_cfd_3d[:, :, -1, :2]   # (ny, nx, 2)
    u = float(np.nanmean(top[..., 0]))
    v = float(np.nanmean(top[..., 1]))
    # meteo direction (where wind comes FROM)
    direction = (np.rad2deg(np.arctan2(-u, -v)) + 360.0) % 360.0
    err = float((direction - inflow["wind_dir"] + 540.0) % 360.0 - 180.0)
    passed = abs(err) <= tol_deg
    return {"passed": passed, "cfd_dir": direction, "err_deg": err}


def qa_one_case(case_dir: Path) -> dict | None:
    import zarr
    grid_path = case_dir / "grid.zarr"
    if not grid_path.exists():
        return None
    inflow = load_inflow_profile(case_dir)
    if inflow is None:
        return {"case": case_dir.name, "error": "no_inflow_json"}

    try:
        g = zarr.open_group(str(grid_path), mode="r")
        z_agl = np.array(g["coords/z_agl"][:])
        U = np.array(g["target/U"][:])    # (ny, nx, nz, 3)
        T = np.array(g["target/T"][:])    # (ny, nx, nz)
        q = np.array(g["target/q"][:]) if "target/q" in g else None
        attrs = dict(g.attrs)
    except Exception as e:
        return {"case": case_dir.name, "error": f"zarr_read:{e}"}

    if not np.all(np.isfinite(U)) or not np.all(np.isfinite(T)):
        return {"case": case_dir.name, "error": "nan_inf"}

    spd = np.sqrt(np.sum(U[..., :2] ** 2, axis=-1))   # (ny, nx, nz)

    # Gates
    g_T = gate_max_principle(T, inflow["T"], tol_rel=0.02)
    g_q = (gate_max_principle(q, inflow["q"], tol_rel=0.02)
           if q is not None and inflow["q"] is not None else None)
    g_U = gate_mass_lower_bound(spd, z_agl, inflow, tol_rel=0.20)
    g_bc = gate_bc_upstream(U, z_agl, inflow, tol_ms=1.5)
    g_dir = gate_topdir(U, inflow, tol_deg=25.0)

    # Speed-up at center, hub-height
    iy, ix = U.shape[0] // 2, U.shape[1] // 2
    spd_center = spd[iy, ix, :]
    z_hub = 100.0
    u_center_hub = float(np.interp(z_hub, z_agl, spd_center))
    u_inflow_hub = float(np.interp(z_hub, inflow["z"], inflow["spd"]))
    speedup = u_center_hub / max(u_inflow_hub, 1e-3)

    return {
        "case": case_dir.name,
        "site_id": attrs.get("site_id", ""),
        "timestamp": attrs.get("timestamp_iso", ""),
        "wind_dir": inflow["wind_dir"],
        "u_hub_inflow_ms": u_inflow_hub,
        "u_hub_cfd_center_ms": u_center_hub,
        "speedup_center_hub": speedup,
        # Gate T
        "T_passed": int(g_T["passed"]),
        "T_under": g_T["under"],
        "T_over": g_T["over"],
        "T_under_rel": g_T["under_rel"],
        "T_over_rel": g_T["over_rel"],
        "T_inflow_min": g_T["inflow_min"],
        "T_inflow_max": g_T["inflow_max"],
        "T_cfd_min": g_T["cfd_min"],
        "T_cfd_max": g_T["cfd_max"],
        # Gate q
        "q_passed": int(g_q["passed"]) if g_q else -1,
        "q_under": g_q["under"] if g_q else float("nan"),
        "q_over": g_q["over"] if g_q else float("nan"),
        "q_under_rel": g_q["under_rel"] if g_q else float("nan"),
        "q_over_rel": g_q["over_rel"] if g_q else float("nan"),
        # Gate U mass lower bound
        "U_passed": int(g_U["passed"]),
        "U_vol_mean_cfd": g_U["vol_mean_cfd"],
        "U_inflow_mean": g_U["inflow_mean"],
        "U_deficit_rel": g_U["deficit_rel"],
        # Gate BC
        "BC_passed": int(g_bc["passed"]),
        "BC_mae_ms": g_bc["mae_ms"],
        # Gate direction
        "Dir_passed": int(g_dir["passed"]),
        "Dir_cfd": g_dir["cfd_dir"],
        "Dir_err_deg": g_dir["err_deg"],
        # Solver
        "solver_converged": int(bool(attrs.get("solver.converged", False))),
        "res_U": attrs.get("solver.final_residual_U", float("nan")),
        "res_p": attrs.get("solver.final_residual_p", float("nan")),
        "n_iter": attrs.get("solver.n_iter", -1),
    }


@click.command()
@click.option("--cases-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--output", default="qa_cfd_vs_inflow.csv", type=click.Path())
@click.option("--summary", default="qa_cfd_vs_inflow.md", type=click.Path())
@click.option("--limit", default=0, type=int, help="Process only first N cases (debug)")
def main(cases_dir, output, summary, limit):
    cases_dir = Path(cases_dir)
    cases = sorted([d for d in cases_dir.iterdir() if d.is_dir()])
    if limit > 0:
        cases = cases[:limit]
    print(f"Found {len(cases)} cases")

    rows = []
    for i, c in enumerate(cases):
        if i % 25 == 0:
            print(f"[{i}/{len(cases)}] {c.name}")
        r = qa_one_case(c)
        if r is not None:
            rows.append(r)

    if not rows:
        print("No QA results")
        sys.exit(1)

    # Write CSV
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"CSV: {output}  ({len(rows)} rows)")

    # Summary
    n = len(rows)
    err = sum(1 for r in rows if "error" in r)
    valid = [r for r in rows if "error" not in r]
    nv = len(valid)
    def pct(key):
        ok = sum(1 for r in valid if r.get(key, 0) == 1)
        return ok, nv, (100.0 * ok / nv if nv else 0.0)
    tT_ok, _, tT_pct = pct("T_passed")
    tq_ok, _, tq_pct = pct("q_passed")
    tU_ok, _, tU_pct = pct("U_passed")
    tBC_ok, _, tBC_pct = pct("BC_passed")
    tD_ok, _, tD_pct = pct("Dir_passed")
    tConv_ok, _, tConv_pct = pct("solver_converged")

    def stats(key):
        vals = np.array([r[key] for r in valid if np.isfinite(r.get(key, np.nan))])
        if vals.size == 0:
            return "no data"
        return (f"median={np.median(vals):.3f}  mean={np.mean(vals):.3f}  "
                f"p90={np.percentile(vals, 90):.3f}  max={np.max(vals):.3f}")

    md = []
    md.append("# QA report — complex_terrain_v1 CFD vs inflow\n")
    md.append(f"- Total cases: **{n}**  (errors: {err}, valid: {nv})")
    md.append("")
    md.append("## Gate pass rates\n")
    md.append(f"- T max-principle (tol 2%): **{tT_ok}/{nv}** ({tT_pct:.1f}%)")
    md.append(f"- q max-principle (tol 2%): **{tq_ok}/{nv}** ({tq_pct:.1f}%)")
    md.append(f"- U mass lower bound (tol 20% deficit): **{tU_ok}/{nv}** ({tU_pct:.1f}%)")
    md.append(f"- Upstream BC profile (tol 1.5 m/s MAE): **{tBC_ok}/{nv}** ({tBC_pct:.1f}%)")
    md.append(f"- Top-of-domain direction (tol 25 deg): **{tD_ok}/{nv}** ({tD_pct:.1f}%)")
    md.append(f"- Solver converged: **{tConv_ok}/{nv}** ({tConv_pct:.1f}%)")
    md.append("")
    md.append("## Distributions\n")
    md.append(f"- speedup_center_hub @100m: {stats('speedup_center_hub')}")
    md.append(f"- U deficit_rel: {stats('U_deficit_rel')}")
    md.append(f"- T under_rel: {stats('T_under_rel')}")
    md.append(f"- T over_rel: {stats('T_over_rel')}")
    md.append(f"- q under_rel: {stats('q_under_rel')}")
    md.append(f"- q over_rel: {stats('q_over_rel')}")
    md.append(f"- BC MAE (m/s): {stats('BC_mae_ms')}")
    md.append(f"- Dir err (deg): {stats('Dir_err_deg')}")
    md.append("")
    md.append("## Worst offenders (top 10)\n")
    for label, key in [("T over", "T_over_rel"), ("T under", "T_under_rel"),
                       ("q over", "q_over_rel"), ("q under", "q_under_rel"),
                       ("U deficit", "U_deficit_rel"), ("BC MAE", "BC_mae_ms"),
                       ("Dir err (abs)", "Dir_err_deg")]:
        md.append(f"### {label}")
        if key == "Dir_err_deg":
            sorted_r = sorted(valid, key=lambda r: -abs(r.get(key, 0.0)))
        else:
            sorted_r = sorted(valid, key=lambda r: -r.get(key, 0.0))
        for r in sorted_r[:10]:
            md.append(f"- {r['case']:<48s} {r.get(key, float('nan')):.4f}  "
                      f"(speedup={r.get('speedup_center_hub', float('nan')):.2f})")
        md.append("")

    with open(summary, "w") as f:
        f.write("\n".join(md))
    print(f"Summary: {summary}")
    print()
    print("\n".join(md[:18]))


if __name__ == "__main__":
    main()
