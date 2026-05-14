#!/usr/bin/env python3
"""Approximate boundary fluxes from reconstructed OpenFOAM internal U fields.

Production exports currently reconstruct internal fields only and do not retain
``phi``. This audit computes a diagnostic proxy for boundary flux using the
owner-cell velocity and boundary face area vector:

    phi_proxy = U_owner · Sf

For inletOutlet top cases this is a useful leakage proxy. For strict slip top
cases it should be interpreted as a near-top internal vertical velocity proxy,
not the exact OpenFOAM boundary flux, since the slip patch value enforces
zero normal velocity at the boundary.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable

import numpy as np


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
    text = path.read_text(errors="replace")
    match = re.search(r"^\s*(\d+)\s*\n\s*\(", text, re.MULTILINE)
    if not match:
        raise ValueError(f"Cannot parse points: {path}")
    block = text[match.end():text.rfind(")")]
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


def parse_faces_for_range(case_dir: Path, start: int, n_faces: int) -> list[np.ndarray]:
    path = case_dir / "constant" / "polyMesh" / "faces"
    text = path.read_text(errors="replace")
    match = re.search(r"^\s*(\d+)\s*\n\s*\(", text, re.MULTILINE)
    if not match:
        raise ValueError(f"Cannot parse faces: {path}")
    block = text[match.end():text.rfind(")")]
    out: list[np.ndarray] = []
    face_i = 0
    for m in re.finditer(r"\d+\(([^)]*)\)", block):
        if face_i >= start + n_faces:
            break
        if face_i >= start:
            out.append(np.fromstring(m.group(1), sep=" ", dtype=np.int64))
        face_i += 1
    if len(out) != n_faces:
        raise ValueError(f"Expected {n_faces} faces, got {len(out)}")
    return out


def face_area_vector(points: np.ndarray, face: np.ndarray) -> np.ndarray:
    verts = points[face]
    if len(verts) < 3:
        return np.zeros(3)
    origin = verts[0]
    area = np.zeros(3)
    for i in range(1, len(verts) - 1):
        area += 0.5 * np.cross(verts[i] - origin, verts[i + 1] - origin)
    return area


def parse_of_vector_field(path: Path) -> np.ndarray:
    text = path.read_text(errors="replace")
    m = re.search(r"internalField\s+nonuniform\s+List<vector>\s*\n(\d+)\s*\n\(", text)
    if not m:
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


def audit_case(case_dir: Path, time_name: str | None = None) -> list[dict[str, object]]:
    case_dir = case_dir.resolve()
    time_name = time_name or latest_numeric_time(case_dir)
    patches = read_boundary_info(case_dir)
    owners = parse_poly_scalar_list(case_dir / "constant" / "polyMesh" / "owner").astype(np.int64)
    points = parse_points(case_dir / "constant" / "polyMesh" / "points")
    U = parse_of_vector_field(case_dir / time_name / "U")

    rows: list[dict[str, object]] = []
    lateral_in = 0.0
    top_out = 0.0
    patch_fluxes: dict[str, tuple[float, float]] = {}

    for patch, info in patches.items():
        start, n_faces = info["startFace"], info["nFaces"]
        faces = parse_faces_for_range(case_dir, start, n_faces)
        owner_idx = owners[start:start + n_faces]
        sf = np.asarray([face_area_vector(points, face) for face in faces], dtype=np.float64)
        area = np.linalg.norm(sf, axis=1)
        phi = np.einsum("ij,ij->i", U[owner_idx], sf)
        pos = float(phi[phi > 0].sum())
        neg = float(phi[phi < 0].sum())
        net = float(phi.sum())
        patch_fluxes[patch] = (pos, neg)
        if patch not in {"top", "terrain", "bottom"}:
            lateral_in += -neg
        if patch == "top":
            top_out += pos

        rows.append(
            {
                "case": case_dir.name,
                "case_dir": str(case_dir),
                "time": time_name,
                "patch": patch,
                "n_faces": n_faces,
                "area_m2": float(area.sum()),
                "proxy_phi_net_m3s": net,
                "proxy_phi_pos_m3s": pos,
                "proxy_phi_neg_m3s": neg,
                "proxy_un_mean_ms": net / max(float(area.sum()), 1e-12),
                "proxy_abs_un_mean_ms": float(np.sum(np.abs(phi)) / max(float(area.sum()), 1e-12)),
                "proxy_un_p10_ms": float(np.percentile(phi / np.maximum(area, 1e-12), 10)),
                "proxy_un_p90_ms": float(np.percentile(phi / np.maximum(area, 1e-12), 90)),
            }
        )

    ratio = top_out / lateral_in if lateral_in > 0 else float("nan")
    for row in rows:
        row["top_out_over_lateral_in"] = ratio
    return rows


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cases", nargs="+", type=Path)
    ap.add_argument("--time", default=None)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(list(argv) if argv is not None else None)

    rows: list[dict[str, object]] = []
    for case in args.cases:
        rows.extend(audit_case(case, args.time))

    if not rows:
        return 0
    fields = list(rows[0].keys())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    else:
        import sys
        writer = csv.DictWriter(sys.stdout, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
