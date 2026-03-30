"""
export_cfd.py — Extract OpenFOAM results → Zarr + mast CSV

For each completed OpenFOAM case, this script:
  1. Reads the latest time-step from the case directory using fluidfoam
  2. Interpolates u, v, w, T, k, nut to mast positions (trilinear)
  3. Writes at-mast values to a CSV file
  4. Writes full 3-D fields to a Zarr store (for GNN training)

Output layout
-------------
data/cfd-database/
  perdigao/
    {case_id}/
      at_masts.csv         # interpolated values at tower positions
      fields.zarr/         # full 3-D fields (u, v, w, T, k, nut)
      metadata.yaml        # case metadata (resolution, context, BC params)

Usage (CLI)
-----------
    python export_cfd.py \
        --case-dir   data/cases/perdigao_1000m_3x3 \
        --towers     configs/sites/perdigao_towers.yaml \
        --site-cfg   configs/sites/perdigao.yaml \
        --case-id    "2017-05-15T12:00_1000m_3x3" \
        --output-dir data/cfd-database/perdigao
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)

FIELDS_TO_EXPORT = ["U", "T", "k", "nut"]   # p_rgh excluded (large, less useful for GNN)
MAST_HEIGHTS_M   = [10, 20, 40, 60, 80, 100]


# ---------------------------------------------------------------------------
# fluidfoam helpers
# ---------------------------------------------------------------------------

def _read_of_internal_vector(filepath: Path) -> np.ndarray:
    """Parse OpenFOAM volVectorField internalField → (N, 3) array.

    Works even when boundaryField is empty (reconstruct_fields.py output).
    """
    import re

    text = filepath.read_text()
    # Find "internalField   nonuniform List<vector>" followed by N and data
    m = re.search(r"internalField\s+nonuniform\s+List<vector>\s*\n(\d+)\s*\n\(", text)
    if not m:
        raise ValueError(f"Cannot parse vector field: {filepath}")

    n = int(m.group(1))
    start = m.end()
    end = text.index("\n)", start)
    block = text[start:end]

    pattern = re.compile(r"\(([^)]+)\)")
    vecs = []
    for match in pattern.finditer(block):
        parts = match.group(1).split()
        vecs.append([float(parts[0]), float(parts[1]), float(parts[2])])

    arr = np.array(vecs, dtype=np.float64)
    if len(arr) != n:
        raise ValueError(f"Expected {n} vectors, got {len(arr)} in {filepath}")
    return arr


def _read_of_internal_scalar(filepath: Path) -> np.ndarray | None:
    """Parse OpenFOAM volScalarField internalField → (N,) array.

    Returns None if the field is uniform or cannot be parsed.
    """
    import re

    if not filepath.exists():
        return None

    text = filepath.read_text()

    # Try nonuniform first
    m = re.search(r"internalField\s+nonuniform\s+List<scalar>\s*\n(\d+)\s*\n\(", text)
    if m:
        n = int(m.group(1))
        start = m.end()
        end = text.index("\n)", start)
        block = text[start:end]
        vals = [float(v) for v in block.split()]
        arr = np.array(vals, dtype=np.float64)
        if len(arr) != n:
            return None
        return arr

    # Try uniform
    m = re.search(r"internalField\s+uniform\s+([-+eE.\d]+)", text)
    if m:
        return None  # uniform field — caller should handle

    return None


def _read_mesh_centres(case_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read cell centres from 0/Cx, 0/Cy, 0/Cz or constant/polyMesh via fluidfoam."""
    # Try pre-computed cell centres first (faster, always available after writeCellCentres)
    cx_path = case_dir / "0" / "Cx"
    if cx_path.exists():
        cx = _read_of_internal_scalar(cx_path)
        cy = _read_of_internal_scalar(case_dir / "0" / "Cy")
        cz = _read_of_internal_scalar(case_dir / "0" / "Cz")
        if cx is not None and cy is not None and cz is not None:
            return cx, cy, cz

    # Fallback to fluidfoam mesh reader
    import fluidfoam
    return fluidfoam.readmesh(str(case_dir))


def _load_openfoam_fields(case_dir: Path) -> dict:
    """Read the latest time-step fields from an OpenFOAM case.

    Uses direct ASCII parsing (handles empty boundaryField from reconstruct_fields.py).
    Falls back to fluidfoam if direct parsing fails.

    Returns
    -------
    dict with keys:
        x, y, z : 1-D coordinate arrays [m] (cell centres)
        U        : (N,3) velocity [m/s]
        T, q, k, epsilon, nut : (N,) scalar fields (if available)
    """
    # Find latest time directory
    time_dirs = sorted(
        [d for d in case_dir.iterdir() if d.is_dir() and _is_time_dir(d.name)],
        key=lambda d: float(d.name),
    )
    if not time_dirs:
        raise FileNotFoundError(f"No time directories found in {case_dir}")
    latest = time_dirs[-1].name
    logger.info("Reading time step: %s from %s", latest, case_dir)

    time_dir = case_dir / latest

    # Cell centre coordinates
    x, y, z = _read_mesh_centres(case_dir)

    # Velocity — try direct parser first (handles empty boundaryField)
    U_path = time_dir / "U"
    try:
        U = _read_of_internal_vector(U_path)
    except Exception:
        import fluidfoam
        U = fluidfoam.readvector(str(case_dir), latest, "U")
        U = np.asarray(U).T if np.asarray(U).shape[0] == 3 else np.asarray(U)

    result = {"x": np.asarray(x), "y": np.asarray(y), "z": np.asarray(z), "U": U}

    # Scalar fields — try direct parser, fallback to fluidfoam
    for field_name in ["T", "q", "k", "epsilon", "nut"]:
        field_path = time_dir / field_name
        val = _read_of_internal_scalar(field_path)
        if val is not None:
            result[field_name] = val
        elif field_path.exists():
            try:
                import fluidfoam
                val = fluidfoam.readscalar(str(case_dir), latest, field_name)
                result[field_name] = np.asarray(val)
            except Exception:
                if field_name in ("k", "nut"):
                    result[field_name] = np.zeros_like(x)
        else:
            if field_name in ("k", "nut"):
                result[field_name] = np.zeros_like(x)

    return result


def _is_time_dir(name: str) -> bool:
    """Return True if the directory name looks like an OpenFOAM time step."""
    try:
        float(name)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Trilinear interpolation at mast positions
# ---------------------------------------------------------------------------

def _interpolate_at_mast(
    fields: dict,
    tower_x: float,
    tower_y: float,
    heights_m: list[float],
) -> dict:
    """Interpolate CFD fields at one mast location and multiple heights.

    Uses a simple nearest-neighbour column selection (find the column of
    cells closest to tower_x, tower_y), then linear interpolation in z.

    Parameters
    ----------
    fields:
        Dict from _load_openfoam_fields().
    tower_x, tower_y:
        Tower coordinates in the CFD coordinate system [m].
        (origin = domain centre = site centre)
    heights_m:
        Heights above ground [m] at which to interpolate.

    Returns
    -------
    dict: height → {u, v, w, T, k, nut}
    """
    x = fields["x"]
    y = fields["y"]
    z = fields["z"]
    U = fields["U"]
    T = fields["T"]
    k = fields["k"]
    nut = fields["nut"]

    # Horizontal distance from tower
    dist_h = np.hypot(x - tower_x, y - tower_y)

    result = {}
    for h in heights_m:
        # Cells within a horizontal radius (search in concentric rings)
        for radius_m in [200, 500, 1000, 2000, 5000]:
            mask = dist_h <= radius_m
            if mask.sum() >= 4:
                break
        if not mask.any():
            mask = np.ones(len(x), dtype=bool)

        # Among candidate cells, find those nearest to target height
        z_candidates = z[mask]
        idx_z = np.argsort(np.abs(z_candidates - h))[:4]
        idxs = np.where(mask)[0][idx_z]

        # Weighted interpolation (inverse distance in z)
        dz = np.abs(z[idxs] - h)
        if dz.min() < 1e-3:
            w = np.zeros(len(idxs))
            w[dz.argmin()] = 1.0
        else:
            w = 1.0 / dz
        w /= w.sum()

        result[h] = {
            "u":   float(np.dot(w, U[idxs, 0])),
            "v":   float(np.dot(w, U[idxs, 1])),
            "w":   float(np.dot(w, U[idxs, 2])),
            "T":   float(np.dot(w, T[idxs])),
            "k":   float(np.dot(w, k[idxs])),
            "nut": float(np.dot(w, nut[idxs])),
        }

    return result


# ---------------------------------------------------------------------------
# At-mast CSV
# ---------------------------------------------------------------------------

def export_at_masts(
    fields: dict,
    towers: list[dict],
    site_lat: float,
    site_lon: float,
    output_csv: Path,
) -> None:
    """Interpolate fields at all towers and write to CSV.

    Parameters
    ----------
    fields:
        Dict from _load_openfoam_fields().
    towers:
        List of tower dicts with keys: id, lat, lon, elevation_m.
    site_lat, site_lon:
        Site centre coordinates (CFD origin).
    output_csv:
        Output CSV path.
    """
    DEG_PER_M_LAT = 1.0 / 111_000.0
    DEG_PER_M_LON = 1.0 / (111_000.0 * np.cos(np.radians(site_lat)))

    rows = []
    for tower in towers:
        t_lat = tower["lat"]
        t_lon = tower["lon"]
        tower_x = (t_lon - site_lon) / DEG_PER_M_LON
        tower_y = (t_lat - site_lat) / DEG_PER_M_LAT

        interp = _interpolate_at_mast(fields, tower_x, tower_y, MAST_HEIGHTS_M)

        for h, vals in interp.items():
            rows.append({
                "tower_id":    tower["id"],
                "height_m":    h,
                "u_ms":        vals["u"],
                "v_ms":        vals["v"],
                "w_ms":        vals["w"],
                "speed_ms":    float(np.hypot(vals["u"], vals["v"])),
                "dir_deg":     float((270.0 - np.degrees(np.arctan2(vals["v"], vals["u"]))) % 360),
                "T_K":         vals["T"],
                "k_m2s2":      vals["k"],
                "nut_m2s":     vals["nut"],
            })

    import csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    logger.info("At-mast CSV written: %s (%d rows)", output_csv, len(rows))


# ---------------------------------------------------------------------------
# 3-D Zarr fields
# ---------------------------------------------------------------------------

def export_zarr_fields(
    fields: dict,
    output_zarr: Path,
    case_id: str,
    metadata: dict,
) -> None:
    """Write full 3-D CFD fields to a Zarr store.

    Parameters
    ----------
    fields:
        Dict from _load_openfoam_fields().
    output_zarr:
        Path to output Zarr store.
    case_id:
        Unique case identifier string.
    metadata:
        Dict of case metadata to store as Zarr attributes.
    """
    try:
        import zarr
        from numcodecs import Blosc
    except ImportError as exc:
        raise ImportError(
            "zarr and numcodecs are required: pip install zarr numcodecs"
        ) from exc

    compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)

    store = zarr.open_group(str(output_zarr), mode="w")

    def _write(name: str, data: np.ndarray, **attrs: str) -> None:
        arr = store.create_array(name, shape=data.shape, dtype=data.dtype,
                                 overwrite=True)
        arr[...] = data
        arr.attrs.update(attrs)

    # Coordinates
    _write("x", fields["x"].astype(np.float32), units="m")
    _write("y", fields["y"].astype(np.float32), units="m")
    _write("z", fields["z"].astype(np.float32), units="m (height above datum)")

    # Velocity
    _write("U", fields["U"].astype(np.float32),
           units="m s-1", description="(N,3) velocity components [u,v,w]")

    # Scalar fields
    for name, units, desc in [
        ("T",   "K",      "Temperature"),
        ("k",   "m2 s-2", "Turbulent kinetic energy"),
        ("nut", "m2 s-1", "Turbulent viscosity"),
    ]:
        if name in fields and fields[name] is not None:
            _write(name, fields[name].astype(np.float32), units=units, description=desc)

    # Global metadata
    store.attrs.update({
        "case_id":   case_id,
        "n_cells":   int(len(fields["x"])),
        **{k: str(v) for k, v in metadata.items()},
    })

    logger.info(
        "Zarr store written: %s (%d cells, fields: x,y,z,U,T,k,nut)",
        output_zarr, len(fields["x"]),
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def export_cfd(
    case_dir: Path | str,
    towers_yaml: Path | str,
    site_cfg: dict,
    case_id: str,
    output_dir: Path | str,
    metadata: dict | None = None,
) -> None:
    """Full export pipeline: OpenFOAM case → CSV + Zarr.

    Parameters
    ----------
    case_dir:
        Path to OpenFOAM case directory.
    towers_yaml:
        Path to perdigao_towers.yaml.
    site_cfg:
        Parsed perdigao.yaml.
    case_id:
        Unique identifier (used as subdirectory name).
    output_dir:
        Root output directory (data/cfd-database/<site>/).
    metadata:
        Optional additional metadata stored in YAML and Zarr attrs.
    """
    case_dir   = Path(case_dir)
    output_dir = Path(output_dir) / case_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tower metadata
    with open(towers_yaml) as f:
        towers_data = yaml.safe_load(f)
    towers_raw = towers_data.get("towers", {})
    # Support both list-of-dicts and dict-of-dicts (ISFS-generated YAML format)
    if isinstance(towers_raw, dict):
        towers = [{"id": k, **v} for k, v in towers_raw.items()]
    else:
        towers = towers_raw

    site_lat = site_cfg["site"]["coordinates"]["latitude"]
    site_lon = site_cfg["site"]["coordinates"]["longitude"]

    # Load OpenFOAM fields
    logger.info("Loading OpenFOAM results from %s", case_dir)
    fields = _load_openfoam_fields(case_dir)

    # At-mast interpolation → CSV
    export_at_masts(
        fields,
        towers,
        site_lat,
        site_lon,
        output_dir / "at_masts.csv",
    )

    # 3-D fields → Zarr
    export_zarr_fields(
        fields,
        output_dir / "fields.zarr",
        case_id=case_id,
        metadata=metadata or {},
    )

    # Case metadata YAML
    meta = {"case_id": case_id, "case_dir": str(case_dir), **(metadata or {})}
    with open(output_dir / "metadata.yaml", "w") as f:
        yaml.dump(meta, f, default_flow_style=False)

    logger.info("Export complete: %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Export OpenFOAM CFD results to Zarr + mast CSV"
    )
    parser.add_argument("--case-dir",   required=True,
                        help="OpenFOAM case directory")
    parser.add_argument("--towers",     required=True,
                        help="perdigao_towers.yaml")
    parser.add_argument("--site-cfg",   required=True,
                        help="perdigao.yaml")
    parser.add_argument("--case-id",    required=True,
                        help="Unique case identifier")
    parser.add_argument("--output-dir", required=True,
                        help="Output root directory (data/cfd-database/<site>/)")
    parser.add_argument("--resolution-m", type=float, default=None)
    parser.add_argument("--context-cells", type=int, default=None)
    args = parser.parse_args()

    with open(args.site_cfg) as f:
        site_cfg = yaml.safe_load(f)

    meta = {}
    if args.resolution_m:
        meta["resolution_m"] = args.resolution_m
    if args.context_cells:
        meta["context_cells"] = args.context_cells

    export_cfd(
        case_dir=args.case_dir,
        towers_yaml=args.towers,
        site_cfg=site_cfg,
        case_id=args.case_id,
        output_dir=args.output_dir,
        metadata=meta,
    )
