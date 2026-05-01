"""
export_to_grid_zarr_v2.py — Native-grid OF → grid.zarr export for campaign v2.

One grid.zarr per OF case, on the *native* mesh inner-block grid (180×180×40).
No IDW interpolation: 1 OF cell ↔ 1 PyTorch voxel. Z(i,j,k) is preserved as the
real cell-center altitude (terrain-following).

Convention (validated 2026-05-01):
    coords/x                       (180,)            x of voxel column [m]
    coords/y                       (180,)            y of voxel column [m]
    coords/z                       (180, 180, 40)    real altitude [m]
    input/terrain                  (180, 180)        Z[:, :, 0] [m]
    input/z0_eff                   scalar            roughness from inflow.json
    input/lat                      scalar            site latitude (Coriolis)
    input/lon                      scalar            site longitude
    input/era5_pressure_levels     (N_press,)        hPa
    input/era5_3d/{u,v,T,q}        (3, 3, N_press)   ERA5 9-col profile
    input/era5_surface/{t2m,d2m,u10,v10}  (3, 3)
    input/inflow_meta              attrs             u_hub, u_star, T_ref, ...
    target/U                       (180, 180, 40, 3) m/s
    target/T                       (180, 180, 40)    K
    target/q                       (180, 180, 40)    kg/kg
    target/k                       (180, 180, 40)    m²/s²    (optional)
    target/epsilon                 (180, 180, 40)    m²/s³    (optional)
    target/nut                     (180, 180, 40)    m²/s     (optional)

Usage
-----
    python export_to_grid_zarr_v2.py \\
        --case-dir /scratch/.../sites/ct_d_fire_0001/case_ts000 \\
        --site-id ct_d_fire_0001 \\
        --site-lat 41.5 --site-lon -2.3 \\
        --era5-zarr /scratch/.../era5_campaign_v3/era5_ct_d_fire_0001.zarr \\
        --timestamp 2017-06-15T12:00:00 \\
        --time 300 \\
        --output /scratch/.../training_v2/ct_d_fire_0001_case_ts000/grid.zarr \\
        --include-turb k epsilon nut
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Native inner-block grid (TBM blocks 30×30×1 × cells 6×6×40)
NI, NJ, NK = 180, 180, 40
HALF_EXTENT_M = 3000.0       # inner box half-size [m] (inner = 6 km × 6 km)
DX = 2 * HALF_EXTENT_M / NI  # 33.333 m horizontal spacing
# Inner-block cell centers are strictly within (-2983.33, +2983.33). Outer cells
# are at |x| or |y| ≥ 3050, so a strict "<= HALF_EXTENT_M" cleanly separates them.

# Optional turbulence targets (subset of {k, epsilon, nut})
DEFAULT_TURB_TARGETS: tuple[str, ...] = ()


def _parse_of_scalar(filepath: Path) -> np.ndarray:
    """Parse an OpenFOAM scalar internalField (nonuniform List<scalar>)."""
    import re
    text = filepath.read_text(errors="replace")
    m = re.search(r"nonuniform\s+List<scalar>\s*\n(\d+)\s*\n\(", text)
    if not m:
        raise ValueError(f"Cannot parse scalar field: {filepath}")
    n = int(m.group(1))
    start = m.end()
    end = text.index(")", start)
    arr = np.fromstring(text[start:end], sep="\n", count=n)
    if len(arr) != n:
        raise ValueError(f"Expected {n} scalars, got {len(arr)}: {filepath}")
    return arr.astype(np.float32)


def _parse_of_vector(filepath: Path) -> np.ndarray:
    """Parse an OpenFOAM vector internalField (nonuniform List<vector>)."""
    import re
    text = filepath.read_text(errors="replace")
    m = re.search(r"nonuniform\s+List<vector>\s*\n(\d+)\s*\n\(", text)
    if not m:
        raise ValueError(f"Cannot parse vector field: {filepath}")
    n = int(m.group(1))
    start = m.end()
    end = text.index("\n)", start)
    block = text[start:end]
    triples = re.findall(r"\(([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\)", block)
    if len(triples) != n:
        raise ValueError(f"Expected {n} vectors, got {len(triples)}: {filepath}")
    return np.asarray(triples, dtype=np.float32)


def load_cell_centers(case_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read 0/Cx, 0/Cy, 0/Cz produced by writeCellCentres."""
    cx = _parse_of_scalar(case_dir / "0" / "Cx")
    cy = _parse_of_scalar(case_dir / "0" / "Cy")
    cz = _parse_of_scalar(case_dir / "0" / "Cz")
    if not (len(cx) == len(cy) == len(cz)):
        raise ValueError(f"Cx/Cy/Cz length mismatch: {len(cx)},{len(cy)},{len(cz)}")
    return cx, cy, cz


def build_inner_block_permutation(
    cx: np.ndarray, cy: np.ndarray, cz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Reshape OF inner-block cells into (NI, NJ, NK) logical grid.

    Returns
    -------
    perm : int64 array, shape (NI, NJ, NK)
        perm[i, j, k] = index into the global OF cell array.
    z_grid : float32 array, shape (NI, NJ, NK)
        Real altitude at each voxel (= cz[perm]).
    """
    # 1. Filter cells inside inner block (strict: outer cylinder cells are
    # at |x|>=3050, inner cell centers at |x|<=2983.3, no overlap risk).
    mask = (np.abs(cx) < HALF_EXTENT_M) & (np.abs(cy) < HALF_EXTENT_M)
    n_inner = int(mask.sum())
    expected = NI * NJ * NK
    if n_inner != expected:
        raise ValueError(
            f"Inner-block cell count mismatch: got {n_inner}, expected {expected}. "
            f"Mesh may not match TBM blocks (30 30 1) × cells (6 6 40)."
        )

    inner_idx = np.flatnonzero(mask)        # global indices of the 1.296M inner cells
    xi = cx[inner_idx]
    yi = cy[inner_idx]
    zi = cz[inner_idx]

    # 2. Bin (x, y) → (i, j). Cell centres are at x = -3000 + (i + 0.5)*DX.
    i = np.round((xi + HALF_EXTENT_M) / DX - 0.5).astype(np.int32)
    j = np.round((yi + HALF_EXTENT_M) / DX - 0.5).astype(np.int32)
    if i.min() < 0 or i.max() >= NI or j.min() < 0 or j.max() >= NJ:
        raise ValueError(
            f"(i,j) out of range: i in [{i.min()},{i.max()}], j in [{j.min()},{j.max()}]"
        )

    # 3. For each (i,j) column, sort by z ascending, assign k = 0..NK-1.
    # Use a structured sort key: (i, j, z) → lex order, then bucket.
    col_id = i.astype(np.int64) * NJ + j.astype(np.int64)
    order = np.lexsort((zi, col_id))   # primary col_id, secondary z
    col_id_sorted = col_id[order]

    # Verify each column has exactly NK cells
    counts = np.bincount(col_id_sorted, minlength=NI * NJ)
    if not np.all(counts == NK):
        bad = np.flatnonzero(counts != NK)
        raise ValueError(
            f"Inner-block column count mismatch: {len(bad)} columns with "
            f"!= {NK} cells (e.g. col {bad[0]}: {counts[bad[0]]})"
        )

    # 4. Build permutation [NI, NJ, NK] of global indices
    perm = np.empty((NI, NJ, NK), dtype=np.int64)
    inner_idx_sorted = inner_idx[order]
    perm.reshape(-1, NK)[:, :] = inner_idx_sorted.reshape(NI * NJ, NK)
    z_grid = cz[perm]
    return perm, z_grid


def load_field(case_dir: Path, time_name: str, var: str, kind: str) -> np.ndarray | None:
    """Load a scalar or vector field at a given time."""
    fpath = case_dir / time_name / var
    if not fpath.exists():
        return None
    if kind == "scalar":
        return _parse_of_scalar(fpath)
    return _parse_of_vector(fpath)


def reshape_to_grid(field: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Apply logical permutation: [N_global] → [NI, NJ, NK[, ...]]."""
    if field.ndim == 1:
        return field[perm]
    if field.ndim == 2:
        # vector field [N_global, 3] → [NI, NJ, NK, 3]
        out = field[perm.reshape(-1)]
        return out.reshape(NI, NJ, NK, field.shape[1])
    raise ValueError(f"Unsupported field ndim={field.ndim}")


def load_era5_at_timestamp(era5_zarr: Path, timestamp_iso: str) -> dict:
    """Load ERA5 3×3 + surface at a given timestamp."""
    import zarr
    g = zarr.open_group(str(era5_zarr), mode="r")

    times = g["coords/time"][:]
    target = np.datetime64(timestamp_iso).astype("datetime64[ns]").astype(np.int64)
    idx = int(np.argmin(np.abs(times.astype(np.int64) - target)))

    levels = np.asarray(g["coords/level"][:], dtype=np.float32)  # hPa
    # pressure subgroup → [time, level, lat, lon] with lat,lon being 3×3
    out = {"era5_pressure_levels": levels}

    pres = g["pressure"]
    out_3d = {}
    for var in ("u", "v", "t", "q"):
        if var in pres:
            arr = np.asarray(pres[var][idx, :, :, :], dtype=np.float32)  # (level, lat, lon)
            arr = np.transpose(arr, (1, 2, 0))                           # (lat, lon, level)
            out_3d["T" if var == "t" else var] = arr
    out["era5_3d"] = out_3d

    surf = g["surface"]
    out_surf = {}
    for var in ("t2m", "d2m", "u10", "v10"):
        if var in surf:
            arr = np.asarray(surf[var][idx, :, :], dtype=np.float32)
            out_surf[var] = arr
    out["era5_surface"] = out_surf

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case-dir", type=Path, required=True)
    ap.add_argument("--site-id", required=True)
    ap.add_argument("--site-lat", type=float, required=True)
    ap.add_argument("--site-lon", type=float, required=True)
    ap.add_argument("--era5-zarr", type=Path, required=True)
    ap.add_argument("--timestamp", required=True, help="ISO 8601, e.g. 2017-06-15T12:00:00")
    ap.add_argument("--time", default="300", help="OF time directory name (default 300)")
    ap.add_argument("--output", type=Path, required=True, help="Output grid.zarr path")
    ap.add_argument("--include-turb", nargs="*", default=list(DEFAULT_TURB_TARGETS),
                    choices=["k", "epsilon", "nut"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    if args.output.exists() and not args.overwrite:
        logger.info("Skip (exists): %s", args.output)
        return 0

    case_dir: Path = args.case_dir
    if not case_dir.is_dir():
        logger.error("Case dir not found: %s", case_dir)
        return 2

    # 1. Mesh + permutation
    cx, cy, cz = load_cell_centers(case_dir)
    perm, z_grid = build_inner_block_permutation(cx, cy, cz)

    # 2. Coords (1D x/y from any column, z is 3D)
    x_1d = (np.arange(NI) + 0.5) * DX - HALF_EXTENT_M
    y_1d = (np.arange(NJ) + 0.5) * DX - HALF_EXTENT_M
    terrain = z_grid[:, :, 0].copy()

    # 3. Targets — U (vector), T, q, optionally k, epsilon, nut
    U_glob = load_field(case_dir, args.time, "U", "vector")
    if U_glob is None:
        logger.error("U not found in %s/%s/", case_dir.name, args.time)
        return 2
    targets: dict[str, np.ndarray] = {"U": reshape_to_grid(U_glob, perm)}

    for var in ("T", "q"):
        f = load_field(case_dir, args.time, var, "scalar")
        if f is not None:
            targets[var] = reshape_to_grid(f, perm)
    for var in args.include_turb:
        f = load_field(case_dir, args.time, var, "scalar")
        if f is not None:
            targets[var] = reshape_to_grid(f, perm)

    # 4. Inflow metadata
    inflow_path = case_dir / "inflow.json"
    inflow = json.loads(inflow_path.read_text()) if inflow_path.exists() else {}

    # 5. ERA5 inputs
    era5 = load_era5_at_timestamp(args.era5_zarr, args.timestamp)

    # 6. Write Zarr
    import zarr
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.overwrite:
        import shutil
        shutil.rmtree(args.output)
    g = zarr.open_group(str(args.output), mode="w")

    coords = g.create_group("coords")
    coords.create_dataset("x", data=x_1d.astype(np.float32))
    coords.create_dataset("y", data=y_1d.astype(np.float32))
    coords.create_dataset("z", data=z_grid.astype(np.float32))

    inp = g.create_group("input")
    inp.create_dataset("terrain", data=terrain.astype(np.float32))
    inp.attrs["z0_eff"] = float(inflow.get("z0_eff", 0.0))
    inp.attrs["lat"] = float(args.site_lat)
    inp.attrs["lon"] = float(args.site_lon)
    inp.create_dataset("era5_pressure_levels", data=era5["era5_pressure_levels"])

    e3d = inp.create_group("era5_3d")
    for var, arr in era5["era5_3d"].items():
        e3d.create_dataset(var, data=arr.astype(np.float32))

    esrf = inp.create_group("era5_surface")
    for var, arr in era5["era5_surface"].items():
        esrf.create_dataset(var, data=arr.astype(np.float32))

    meta = inp.create_group("inflow_meta")
    for k_, v_ in inflow.items():
        if isinstance(v_, (int, float)) or v_ is None:
            meta.attrs[k_] = v_ if v_ is not None else float("nan")
    meta.attrs["timestamp"] = args.timestamp
    meta.attrs["site_id"] = args.site_id

    tgt = g.create_group("target")
    for var, arr in targets.items():
        tgt.create_dataset(var, data=arr.astype(np.float32))

    g.attrs.update({
        "schema_version": "v2.0",
        "site_id": args.site_id,
        "case_dir": str(case_dir),
        "time": args.time,
        "grid_shape": [NI, NJ, NK],
        "dx_m": DX,
        "half_extent_m": HALF_EXTENT_M,
    })

    logger.info(
        "OK %s/%s → %s  [shape=(%d,%d,%d), targets=%s]",
        args.site_id, case_dir.name, args.output,
        NI, NJ, NK, list(targets),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
