#!/usr/bin/env python3
"""Audit OpenFOAM terrain-wall roughness consistency.

The check is intentionally local to the wall:

* read the z0 used by atm* wall functions in ``0/nut``;
* compute terrain face-to-owner-cell distance as a practical first-cell y;
* compare owner-cell wind speed against the OpenFOAM rough log-law
  ``u*/kappa * log((y + z0) / z0)``;
* compare wall-cell ``k`` implied ``u*`` against ``inflow.json`` ``u_star``.

This targets the failure mode where the lateral inflow is correct but the
terrain wall source terms drain momentum too aggressively.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np


KAPPA = 0.4
CMU = 0.09


def parse_numeric_block(path: Path, count: int | None = None) -> str:
    text = path.read_text(errors="replace")
    match = re.search(r"^\s*(\d+)\s*\n\s*\(", text, re.MULTILINE)
    if not match:
        raise ValueError(f"Cannot find OpenFOAM list block in {path}")
    n = int(match.group(1))
    if count is not None and n != count:
        raise ValueError(f"{path}: expected {count} entries, got {n}")
    start = match.end()
    end = text.rfind(")")
    if end <= start:
        raise ValueError(f"Cannot find list end in {path}")
    return text[start:end]


def parse_poly_scalar_list(path: Path) -> np.ndarray:
    text = path.read_text(errors="replace")
    match = re.search(r"^\s*(\d+)\s*\n\s*\(", text, re.MULTILINE)
    if not match:
        raise ValueError(f"Cannot parse scalar list: {path}")
    n = int(match.group(1))
    start = match.end()
    end = text.rfind(")")
    arr = np.fromstring(text[start:end], sep="\n", count=n)
    if len(arr) != n:
        raise ValueError(f"{path}: expected {n} scalars, got {len(arr)}")
    return arr


def parse_points(path: Path) -> np.ndarray:
    block = parse_numeric_block(path)
    triples = re.findall(r"\(([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\)", block)
    return np.asarray(triples, dtype=np.float64)


def read_boundary_info(case_dir: Path) -> dict[str, dict[str, int]]:
    path = case_dir / "constant" / "polyMesh" / "boundary"
    text = path.read_text(errors="replace")
    match = re.search(r"^\s*(\d+)\s*\(", text, re.MULTILINE)
    if not match:
        raise ValueError(f"Cannot parse boundary file: {path}")
    block = text[match.end():]
    patches: dict[str, dict[str, int]] = {}
    for m in re.finditer(r"(\w+)\s*\{([^}]+)\}", block):
        name = m.group(1)
        body = m.group(2)
        nf = re.search(r"nFaces\s+(\d+)", body)
        sf = re.search(r"startFace\s+(\d+)", body)
        if nf and sf:
            patches[name] = {"nFaces": int(nf.group(1)), "startFace": int(sf.group(1))}
    return patches


def parse_patch_face_centres(case_dir: Path, patch: str) -> tuple[np.ndarray, np.ndarray]:
    poly = case_dir / "constant" / "polyMesh"
    patches = read_boundary_info(case_dir)
    if patch not in patches:
        raise KeyError(f"Patch {patch!r} not found; available={sorted(patches)}")
    start = patches[patch]["startFace"]
    n_faces = patches[patch]["nFaces"]

    points = parse_points(poly / "points")
    owner_all = parse_poly_scalar_list(poly / "owner").astype(np.int64)

    faces_text = (poly / "faces").read_text(errors="replace")
    match = re.search(r"^\s*(\d+)\s*\n\s*\(", faces_text, re.MULTILINE)
    if not match:
        raise ValueError(f"Cannot parse faces: {poly / 'faces'}")
    block = faces_text[match.end():faces_text.rfind(")")]

    centres = np.empty((n_faces, 3), dtype=np.float64)
    found = 0
    face_i = 0
    for m in re.finditer(r"\d+\(([^)]*)\)", block):
        if face_i >= start + n_faces:
            break
        if face_i >= start:
            verts = np.fromstring(m.group(1), sep=" ", dtype=np.int64)
            centres[found] = points[verts].mean(axis=0)
            found += 1
        face_i += 1
    if found != n_faces:
        raise ValueError(f"Expected {n_faces} {patch} faces, found {found}")

    owners = owner_all[start:start + n_faces]
    return centres, owners


def parse_of_scalar_field(path: Path) -> np.ndarray:
    text = path.read_text(errors="replace")
    m = re.search(r"internalField\s+nonuniform\s+List<scalar>\s*\n(\d+)\s*\n\(", text)
    if not m:
        m_uniform = re.search(r"internalField\s+uniform\s+([\d.eE+-]+)\s*;", text)
        if m_uniform:
            raise ValueError(f"{path} is uniform; cannot infer field length")
        raise ValueError(f"Cannot parse scalar internalField: {path}")
    n = int(m.group(1))
    start = m.end()
    end = text.index(")", start)
    arr = np.fromstring(text[start:end], sep="\n", count=n)
    if len(arr) != n:
        raise ValueError(f"{path}: expected {n} scalars, got {len(arr)}")
    return arr


def parse_of_vector_field(path: Path) -> np.ndarray:
    text = path.read_text(errors="replace")
    m = re.search(r"internalField\s+nonuniform\s+List<vector>\s*\n(\d+)\s*\n\(", text)
    if not m:
        m_uniform = re.search(
            r"internalField\s+uniform\s+\(([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\)\s*;",
            text,
        )
        if m_uniform:
            raise ValueError(f"{path} is uniform; cannot infer field length")
        raise ValueError(f"Cannot parse vector internalField: {path}")
    n = int(m.group(1))
    start = m.end()
    end = text.index("\n)", start)
    triples = re.findall(r"\(([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\)", text[start:end])
    if len(triples) != n:
        raise ValueError(f"{path}: expected {n} vectors, got {len(triples)}")
    return np.asarray(triples, dtype=np.float64)


def latest_numeric_time(case_dir: Path) -> str:
    times: list[tuple[float, str]] = []
    for child in case_dir.iterdir():
        if child.is_dir():
            try:
                times.append((float(child.name), child.name))
            except ValueError:
                pass
    if not times:
        raise ValueError(f"No numeric time directories in {case_dir}")
    return max(times)[1]


def patch_block(text: str, patch: str) -> str:
    m = re.search(rf"\b{re.escape(patch)}\s*\{{", text)
    if not m:
        raise KeyError(f"Patch {patch!r} block not found")
    depth = 0
    for i in range(m.end() - 1, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[m.end():i]
    raise ValueError(f"Unclosed patch block {patch!r}")


def read_wall_z0(case_dir: Path, patch: str, n_faces: int) -> tuple[np.ndarray, str]:
    nut_text = (case_dir / "0" / "nut").read_text(errors="replace")
    block = patch_block(nut_text, patch)
    m = re.search(r"z0\s+uniform\s+([\d.eE+-]+)\s*;", block)
    if m:
        return np.full(n_faces, float(m.group(1)), dtype=np.float64), "uniform"

    if "fieldTable" in block or "mappedFile" in block:
        z0_file = case_dir / "constant" / "boundaryData" / patch / "0" / "z0"
        arr = parse_poly_scalar_list(z0_file).astype(np.float64)
        if len(arr) != n_faces:
            raise ValueError(f"{z0_file}: expected {n_faces} entries, got {len(arr)}")
        return arr, "mappedFile"

    raise ValueError(f"Cannot identify z0 mode in {case_dir / '0' / 'nut'} patch {patch}")


def read_inflow(case_dir: Path) -> dict:
    path = case_dir / "inflow.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def interp_profile(inflow: dict, height: float) -> float:
    z = inflow.get("z_levels") or inflow.get("z_m") or inflow.get("z")
    spd = inflow.get("u_profile") or inflow.get("speed_ms") or inflow.get("spd")
    if not z or not spd:
        return math.nan
    return float(np.interp(height, np.asarray(z, dtype=float), np.asarray(spd, dtype=float)))


def stat(prefix: str, values: np.ndarray, out: dict[str, float]) -> None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        for name in ("mean", "median", "p10", "p90", "min", "max"):
            out[f"{prefix}_{name}"] = math.nan
        return
    out[f"{prefix}_mean"] = float(values.mean())
    out[f"{prefix}_median"] = float(np.median(values))
    out[f"{prefix}_p10"] = float(np.percentile(values, 10))
    out[f"{prefix}_p90"] = float(np.percentile(values, 90))
    out[f"{prefix}_min"] = float(values.min())
    out[f"{prefix}_max"] = float(values.max())


def audit_case(case_dir: Path, patch: str = "terrain", time_name: str | None = None) -> dict[str, float | str | int]:
    case_dir = case_dir.resolve()
    time_name = time_name or latest_numeric_time(case_dir)
    inflow = read_inflow(case_dir)

    face_centres, owners = parse_patch_face_centres(case_dir, patch)
    n_faces = len(owners)
    z0, z0_mode = read_wall_z0(case_dir, patch, n_faces)

    cx = parse_of_scalar_field(case_dir / "0" / "Cx")
    cy = parse_of_scalar_field(case_dir / "0" / "Cy")
    cz = parse_of_scalar_field(case_dir / "0" / "Cz")
    cell_centres = np.column_stack([cx[owners], cy[owners], cz[owners]])
    y = np.linalg.norm(cell_centres - face_centres, axis=1)

    U = parse_of_vector_field(case_dir / time_name / "U")
    k = parse_of_scalar_field(case_dir / time_name / "k")
    nut = parse_of_scalar_field(case_dir / time_name / "nut") if (case_dir / time_name / "nut").exists() else None

    U_wall = U[owners]
    speed_wall = np.linalg.norm(U_wall, axis=1)
    k_wall = np.maximum(k[owners], 0.0)

    u_star_inflow = float(inflow.get("u_star", math.nan))
    z0_eff = float(inflow.get("z0_eff", inflow.get("z0", math.nan)))
    u_star_from_k = CMU ** 0.25 * np.sqrt(k_wall)
    with np.errstate(divide="ignore", invalid="ignore"):
        y_over_z0 = y / z0
        log_denom = np.log(np.maximum((y + z0) / z0, 1.0 + 1e-4))
        eq_speed = (u_star_inflow / KAPPA) * log_denom
        speed_over_eq = speed_wall / eq_speed
        ustar_k_over_inflow = u_star_from_k / u_star_inflow

    out: dict[str, float | str | int] = {
        "case": case_dir.name,
        "case_dir": str(case_dir),
        "time": time_name,
        "patch": patch,
        "n_faces": int(n_faces),
        "z0_mode": z0_mode,
        "inflow_z0_eff": z0_eff,
        "inflow_u_star": u_star_inflow,
        "inflow_u2_ms": interp_profile(inflow, 2.0),
        "inflow_u10_ms": interp_profile(inflow, 10.0),
        "inflow_u100_ms": interp_profile(inflow, 100.0),
        "frac_y_lt_z0": float(np.mean(y < z0)),
        "frac_y_lt_2z0": float(np.mean(y < 2.0 * z0)),
        "frac_wall_speed_lt_half_equil": float(np.mean(speed_over_eq < 0.5)),
    }
    stat("z0_m", z0, out)
    stat("wall_y_m", y, out)
    stat("wall_y_over_z0", y_over_z0, out)
    stat("wall_speed_ms", speed_wall, out)
    stat("wall_equil_log_speed_ms", eq_speed, out)
    stat("wall_speed_over_equil", speed_over_eq, out)
    stat("wall_ustar_from_k_ms", u_star_from_k, out)
    stat("wall_ustar_k_over_inflow", ustar_k_over_inflow, out)
    if nut is not None:
        stat("wall_owner_nut_m2s", nut[owners], out)
    return out


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cases", nargs="+", type=Path)
    ap.add_argument("--patch", default="terrain")
    ap.add_argument("--time", default=None)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(list(argv) if argv is not None else None)

    rows = [audit_case(case, patch=args.patch, time_name=args.time) for case in args.cases]
    if not rows:
        return 0

    fieldnames = list(rows[0].keys())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
