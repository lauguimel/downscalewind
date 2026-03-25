"""
export_campaign_zarr.py — Export a solved OF campaign to a single stacked Zarr.

Reads all case_ts* directories, loads fields via fluidfoam, stacks them into
arrays indexed by (case, cell), and writes a single Zarr store with:
  - CFD fields: U(n_cases, n_cells, 3), T/q/k/epsilon/nut(n_cases, n_cells)
  - Shared coordinates: x/y/z/z_agl/elev(n_cells)
  - Per-case metadata: timestamp, u_hub, wind_dir, T_ref, q_ref, ...
  - ERA5 inflow profiles: z_levels/u_profile/T_profile/q_profile(n_cases, n_levels)
  - Terrain: STL points, z0 field from WorldCover

Designed for GNN training: all cases share the same mesh, so coordinates
are stored once. Fields are chunked by case for efficient batch loading.

Usage
-----
    python export_campaign_zarr.py \
        --cases-dir /path/to/cases/poc_100ts_q \
        --output /path/to/poc_100ts_q.zarr \
        --time 500

Dependencies: numpy, zarr, fluidfoam, scipy
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def find_cases(cases_dir: Path, prefix: str = "case_ts") -> list[Path]:
    """Find and sort case directories."""
    cases = sorted([
        d for d in cases_dir.iterdir()
        if d.is_dir() and d.name.startswith(prefix)
    ])
    return cases


def load_mesh(case_dir: Path) -> dict:
    """Load mesh cell centres from Cx/Cy/Cz files (written by writeCellCentres)."""
    # Try 0/Cx first (from writeCellCentres postProcess)
    cx_path = case_dir / "0" / "Cx"
    if cx_path.exists():
        x = _parse_of_scalar(cx_path)
        y = _parse_of_scalar(case_dir / "0" / "Cy")
        z = _parse_of_scalar(case_dir / "0" / "Cz")
        if x is not None and y is not None and z is not None:
            return {"x": x, "y": y, "z": z}

    # Fallback to fluidfoam
    import fluidfoam
    case_str = str(case_dir)
    x, y, z = fluidfoam.readmesh(case_str)
    return {
        "x": np.asarray(x, dtype=np.float32),
        "y": np.asarray(y, dtype=np.float32),
        "z": np.asarray(z, dtype=np.float32),
    }


def compute_terrain_elevation(x, y, z) -> tuple[np.ndarray, np.ndarray]:
    """Compute terrain elevation and z_agl from cell coordinates.

    Groups cells by horizontal position (binned to 10m) and takes min z
    as terrain elevation for that column.
    """
    x_bin = np.round(x / 10.0) * 10.0
    y_bin = np.round(y / 10.0) * 10.0
    col_id = x_bin * 1e7 + y_bin
    _, inverse = np.unique(col_id, return_inverse=True)

    elev = np.empty_like(z)
    for col_idx in range(inverse.max() + 1):
        mask = inverse == col_idx
        elev[mask] = z[mask].min()

    z_agl = z - elev
    return elev.astype(np.float32), z_agl.astype(np.float32)


def _parse_of_scalar(filepath: Path) -> np.ndarray | None:
    """Parse an OpenFOAM scalar field (nonuniform List<scalar>)."""
    text = filepath.read_text()
    # Find the data block: N\n(\nval\nval\n...\n)
    import re
    m = re.search(r'nonuniform\s+List<scalar>\s*\n(\d+)\s*\n\(', text)
    if not m:
        # Try uniform
        m2 = re.search(r'internalField\s+uniform\s+([\d.eE+-]+)', text)
        if m2:
            return None  # uniform — caller should handle
        return None
    n = int(m.group(1))
    start = m.end()
    end = text.index(')', start)
    vals = text[start:end].split()
    return np.array([float(v) for v in vals[:n]], dtype=np.float32)


def _parse_of_vector(filepath: Path) -> np.ndarray | None:
    """Parse an OpenFOAM vector field (nonuniform List<vector>)."""
    text = filepath.read_text()
    import re
    m = re.search(r'nonuniform\s+List<vector>\s*\n(\d+)\s*\n\(', text)
    if not m:
        return None
    n = int(m.group(1))
    start = m.end()
    end = text.index('\n)', start)
    lines = text[start:end].strip().split('\n')
    result = np.zeros((n, 3), dtype=np.float32)
    for i, line in enumerate(lines[:n]):
        line = line.strip().strip('()')
        parts = line.split()
        result[i] = [float(parts[0]), float(parts[1]), float(parts[2])]
    return result


def load_fields(case_dir: Path, time_name: str) -> dict | None:
    """Load CFD fields from a specific time step (direct parsing, no fluidfoam)."""
    time_dir = case_dir / time_name

    if not time_dir.exists():
        logger.warning("Time dir %s not found in %s", time_name, case_dir)
        return None

    result = {}

    # Velocity (vector)
    u_path = time_dir / "U"
    if not u_path.exists():
        logger.error("U not found in %s/%s", case_dir.name, time_name)
        return None
    U = _parse_of_vector(u_path)
    if U is None:
        logger.error("Cannot parse U from %s", case_dir.name)
        return None
    result["U"] = U

    # Scalar fields
    for fname in ["T", "q", "k", "epsilon", "nut"]:
        fpath = time_dir / fname
        if fpath.exists():
            val = _parse_of_scalar(fpath)
            if val is not None:
                result[fname] = val

    return result


def load_inflow(case_dir: Path) -> dict | None:
    """Load inflow.json metadata."""
    inflow_path = case_dir / "inflow.json"
    if not inflow_path.exists():
        return None
    with open(inflow_path) as f:
        return json.load(f)


def load_stl(cases_dir: Path) -> np.ndarray | None:
    """Load STL terrain points from any case (handles ASCII and binary STL)."""
    import struct

    for case_dir in sorted(cases_dir.iterdir()):
        stl_path = case_dir / "constant" / "triSurface" / "terrain.stl"
        if not stl_path.exists():
            continue

        data = stl_path.read_bytes()

        # Try ASCII first
        if data[:5] == b"solid":
            try:
                text = data.decode("utf-8")
                points = []
                for line in text.splitlines():
                    line = line.strip()
                    if line.startswith("vertex"):
                        parts = line.split()
                        points.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if points:
                    return np.array(points, dtype=np.float32)
            except (UnicodeDecodeError, ValueError):
                pass  # fall through to binary

        # Binary STL: 80-byte header, 4-byte triangle count, then 50 bytes per triangle
        if len(data) > 84:
            n_triangles = struct.unpack_from("<I", data, 80)[0]
            points = np.zeros((n_triangles * 3, 3), dtype=np.float32)
            for i in range(n_triangles):
                offset = 84 + i * 50
                # Skip normal (12 bytes), read 3 vertices (36 bytes)
                for v in range(3):
                    voff = offset + 12 + v * 12
                    points[i * 3 + v] = struct.unpack_from("<fff", data, voff)
            # Deduplicate vertices
            points = np.unique(points, axis=0)
            return points

    return None


def load_z0_field(cases_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load z0 WorldCover field from boundaryData (shared mesh)."""
    for case_dir in sorted(cases_dir.iterdir()):
        z0_path = case_dir / "constant" / "boundaryData" / "terrain" / "0" / "z0"
        pts_path = case_dir / "constant" / "boundaryData" / "terrain" / "points"
        if z0_path.exists() and pts_path.exists():
            # Parse z0 values
            z0_vals = []
            with open(z0_path) as f:
                in_list = False
                for line in f:
                    line = line.strip()
                    if line == "(":
                        in_list = True
                        continue
                    if line == ")":
                        break
                    if in_list:
                        try:
                            z0_vals.append(float(line))
                        except ValueError:
                            pass
            if z0_vals:
                return np.array(z0_vals, dtype=np.float32)
    return None


def export_zarr(
    cases_dir: Path,
    output_path: Path,
    time_name: str,
    case_prefix: str = "case_ts",
) -> None:
    """Main export pipeline."""
    import zarr

    cases = find_cases(cases_dir, case_prefix)
    if not cases:
        logger.error("No cases found in %s", cases_dir)
        return

    # Filter to solved cases only
    solved = [c for c in cases if (c / time_name / "U").exists()]
    logger.info("Found %d cases (%d solved) in %s", len(cases), len(solved), cases_dir)

    if not solved:
        logger.error("No solved cases found")
        return

    # --- Load shared mesh from first case ---
    logger.info("Loading mesh from %s", solved[0].name)
    mesh = load_mesh(solved[0])
    n_cells = len(mesh["x"])
    logger.info("Mesh: %d cells", n_cells)

    elev, z_agl = compute_terrain_elevation(mesh["x"], mesh["y"], mesh["z"])

    # --- Load all cases ---
    n_cases = len(solved)
    all_fields = {}
    all_inflows = []
    field_names = ["U", "T", "q", "k", "epsilon", "nut"]

    for i, case_dir in enumerate(solved):
        logger.info("  [%d/%d] %s", i + 1, n_cases, case_dir.name)

        fields = load_fields(case_dir, time_name)
        if fields is None:
            logger.error("    FAILED — skipping")
            continue

        # Verify cell count matches
        if len(fields["U"]) != n_cells:
            logger.error("    Cell count mismatch: %d vs %d", len(fields["U"]), n_cells)
            continue

        for fname in field_names:
            if fname not in all_fields:
                if fname == "U":
                    all_fields[fname] = np.zeros((n_cases, n_cells, 3), dtype=np.float32)
                else:
                    all_fields[fname] = np.zeros((n_cases, n_cells), dtype=np.float32)
            if fname in fields:
                all_fields[fname][i] = fields[fname]

        inflow = load_inflow(case_dir)
        all_inflows.append(inflow)

    # --- Collect metadata ---
    meta_keys = ["u_hub", "u_star", "z0_eff", "T_ref", "q_ref", "Ri_b", "wind_dir"]
    meta = {k: np.zeros(n_cases, dtype=np.float32) for k in meta_keys}
    timestamps = []

    # Inflow profiles (variable length → pad to max)
    max_levels = 0
    for inf in all_inflows:
        if inf and "z_levels" in inf:
            max_levels = max(max_levels, len(inf["z_levels"]))

    profile_keys = ["z_levels", "u_profile", "T_profile", "q_profile",
                    "ux_profile", "uy_profile"]
    profiles = {k: np.full((n_cases, max_levels), np.nan, dtype=np.float32)
                for k in profile_keys}

    for i, inf in enumerate(all_inflows):
        if inf is None:
            timestamps.append("")
            continue

        # Extract timestamp from case directory name
        case_name = solved[i].name  # e.g. case_ts042
        timestamps.append(case_name)

        for k in meta_keys:
            if k in inf:
                val = inf[k]
                meta[k][i] = float(val) if val is not None else np.nan

        for k in profile_keys:
            if k in inf:
                arr = np.array(inf[k], dtype=np.float32)
                profiles[k][i, :len(arr)] = arr

    # --- Load terrain assets ---
    logger.info("Loading terrain assets...")
    stl_points = load_stl(cases_dir)
    z0_field = load_z0_field(cases_dir)

    # --- Write Zarr ---
    logger.info("Writing Zarr → %s", output_path)
    store = zarr.open_group(str(output_path), mode="w")

    # CFD fields (chunked by case for efficient batch loading)
    for fname, data in all_fields.items():
        if fname == "U":
            chunks = (1, n_cells, 3)
        else:
            chunks = (1, n_cells)
        store.create_array(fname, data=data, chunks=chunks)

    # Coordinates (shared)
    coords = store.create_group("coords")
    coords.create_array("x", data=mesh["x"])
    coords.create_array("y", data=mesh["y"])
    coords.create_array("z", data=mesh["z"])
    coords.create_array("z_agl", data=z_agl)
    coords.create_array("elev", data=elev)

    # Metadata
    meta_grp = store.create_group("meta")
    for k, v in meta.items():
        meta_grp.create_array(k, data=v)
    # Timestamps as bytes (zarr doesn't support variable-length strings well)
    ts_arr = np.array(timestamps, dtype="U64")
    meta_grp.create_array("case_id", data=ts_arr)

    # Inflow profiles
    inflow_grp = store.create_group("inflow")
    for k, v in profiles.items():
        inflow_grp.create_array(k, data=v)

    # Terrain
    terrain_grp = store.create_group("terrain")
    if stl_points is not None:
        terrain_grp.create_array("stl_points", data=stl_points)
        logger.info("  STL: %d vertices", len(stl_points))
    if z0_field is not None:
        terrain_grp.create_array("z0_field", data=z0_field)
        logger.info("  z0 field: %d faces", len(z0_field))

    # Global attributes
    store.attrs.update({
        "n_cases": n_cases,
        "n_cells": n_cells,
        "n_inflow_levels": max_levels,
        "time_step": time_name,
        "site": "perdigao",
        "solver": "simpleFoam",
        "physics": "SF + Coriolis + T_passive + q_passive + WorldCover_z0",
        "mesh_type": "terrainBlockMesher",
    })

    # Summary
    total_bytes = sum(
        store[k].nbytes for k in store.keys()
        if hasattr(store[k], "nbytes")
    )
    logger.info("Done: %d cases, %d cells, %.1f MB uncompressed",
                n_cases, n_cells, total_bytes / 1e6)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Export solved OF campaign to stacked Zarr"
    )
    parser.add_argument("--cases-dir", required=True, type=Path,
                        help="Directory with case_ts* subdirs")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output Zarr path")
    parser.add_argument("--time", default="500",
                        help="Time step to export (default: 500)")
    parser.add_argument("--prefix", default="case_ts",
                        help="Case directory prefix (default: case_ts)")
    args = parser.parse_args()

    export_zarr(args.cases_dir, args.output, args.time, args.prefix)


if __name__ == "__main__":
    main()
