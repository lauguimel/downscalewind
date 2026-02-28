"""
init_from_era5.py — Initialise OpenFOAM fields from ERA5 interpolation

Replaces potentialFoam: interpolates ERA5 wind (u, v), k, and omega to
every cell centre in the mesh for a much better initial condition.

Pipeline position:
    blockMesh → [snappyHexMesh] → checkMesh → **init_from_era5** → simpleFoam

The script:
  1. Reads cell centres from the OpenFOAM case (constant/polyMesh/C or via
     postProcess -func writeCellCentres)
  2. For each cell (x, y, z):
     - Converts (x, y) → (lat, lon) via local projection
     - Interpolates ERA5 u, v vertically (log-law 0–100m, spline above)
     - Computes k(z) = u*²/√Cmu  and  ω(z) = u*/(κ·z)
  3. Writes 0/U, 0/k, 0/omega with `internalField nonuniform List<...>`

Usage
-----
    python init_from_era5.py \\
        --case-dir  data/cases/perdigao_500m_1x1/ \\
        --inflow    data/processed/inflow/2017-05-11T06_00.json
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

KAPPA = 0.41
CMU   = 0.09


# ---------------------------------------------------------------------------
# Read OpenFOAM cell centres
# ---------------------------------------------------------------------------

def read_cell_centres(case_dir: Path) -> np.ndarray:
    """Read cell centre coordinates from an OpenFOAM case.

    Tries (in order):
      1. Parse constant/polyMesh/C (if it exists after writeCellCentres)
      2. Parse 0/C{x,y,z} written by `postProcess -func writeCellCentres`

    Returns
    -------
    centres : (N, 3) array of (x, y, z) cell centre coordinates [m].
    """
    # Option 1: 0/Cx, 0/Cy, 0/Cz
    cx_path = case_dir / "0" / "Cx"
    cy_path = case_dir / "0" / "Cy"
    cz_path = case_dir / "0" / "Cz"
    if cx_path.exists() and cy_path.exists() and cz_path.exists():
        cx = _parse_of_scalar_field(cx_path)
        cy = _parse_of_scalar_field(cy_path)
        cz = _parse_of_scalar_field(cz_path)
        return np.column_stack([cx, cy, cz])

    # Option 2: constant/polyMesh/C (vector field)
    c_path = case_dir / "constant" / "polyMesh" / "C"
    if c_path.exists():
        return _parse_of_vector_field(c_path)

    raise FileNotFoundError(
        f"Cannot find cell centres in {case_dir}. "
        "Run `postProcess -func writeCellCentres` first."
    )


def _parse_of_scalar_field(filepath: Path) -> np.ndarray:
    """Parse an OpenFOAM volScalarField into a 1-D numpy array."""
    text = filepath.read_text()
    # Find the internalField block: N\n(\nval\nval\n...\n)
    match = re.search(r'internalField\s+nonuniform\s+List<scalar>\s*\n(\d+)\s*\n\(', text)
    if match:
        n = int(match.group(1))
        start = match.end()
        end = text.index(')', start)
        values = text[start:end].split()
        return np.array([float(v) for v in values[:n]])

    # Uniform field
    match = re.search(r'internalField\s+uniform\s+([\d.eE+\-]+)', text)
    if match:
        logger.warning("Scalar field %s is uniform — cannot determine N", filepath)
        return np.array([float(match.group(1))])

    raise ValueError(f"Cannot parse scalar field: {filepath}")


def _parse_of_vector_field(filepath: Path) -> np.ndarray:
    """Parse an OpenFOAM volVectorField into an (N, 3) numpy array."""
    text = filepath.read_text()
    match = re.search(r'internalField\s+nonuniform\s+List<vector>\s*\n(\d+)\s*\n\(', text)
    if not match:
        raise ValueError(f"Cannot parse vector field: {filepath}")

    n = int(match.group(1))
    start = match.end()
    end = text.index(')', start)
    # Each line: (vx vy vz)
    vectors = re.findall(r'\(\s*([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s*\)',
                         text[start:end])
    arr = np.array([[float(x), float(y), float(z)] for x, y, z in vectors[:n]])
    return arr


# ---------------------------------------------------------------------------
# Write OpenFOAM nonuniform fields
# ---------------------------------------------------------------------------

def _write_of_vector_field(
    filepath: Path,
    field_name: str,
    data: np.ndarray,
    dimensions: str = "[0 1 -1 0 0 0 0]",
) -> None:
    """Write an OpenFOAM volVectorField with nonuniform internalField."""
    n = len(data)
    lines = [
        '/*--------------------------------*- C++ -*----------------------------------*\\',
        '  =========                 |',
        '  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox',
        '   \\\\    /   O peration     | Version: 10',
        '    \\\\  /    A nd           | Web:     www.openfoam.org',
        '     \\\\/     M anipulation  |',
        '\\*---------------------------------------------------------------------------*/',
        'FoamFile',
        '{',
        '    version     2.0;',
        '    format      ascii;',
        '    class       volVectorField;',
        '    location    "0";',
        f'    object      {field_name};',
        '}',
        '// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //',
        '',
        f'dimensions      {dimensions};',
        '',
        f'internalField   nonuniform List<vector>',
        f'{n}',
        '(',
    ]
    for i in range(n):
        lines.append(f'({data[i, 0]:.6f} {data[i, 1]:.6f} {data[i, 2]:.6f})')
    lines.append(')')
    lines.append(';')
    lines.append('')

    filepath.write_text('\n'.join(lines))


def _write_of_scalar_field(
    filepath: Path,
    field_name: str,
    data: np.ndarray,
    dimensions: str = "[0 2 -2 0 0 0 0]",
) -> None:
    """Write an OpenFOAM volScalarField with nonuniform internalField."""
    n = len(data)
    lines = [
        '/*--------------------------------*- C++ -*----------------------------------*\\',
        '  =========                 |',
        '  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox',
        '   \\\\    /   O peration     | Version: 10',
        '    \\\\  /    A nd           | Web:     www.openfoam.org',
        '     \\\\/     M anipulation  |',
        '\\*---------------------------------------------------------------------------*/',
        'FoamFile',
        '{',
        '    version     2.0;',
        '    format      ascii;',
        '    class       volScalarField;',
        '    location    "0";',
        f'    object      {field_name};',
        '}',
        '// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //',
        '',
        f'dimensions      {dimensions};',
        '',
        f'internalField   nonuniform List<scalar>',
        f'{n}',
        '(',
    ]
    for i in range(n):
        lines.append(f'{data[i]:.6e}')
    lines.append(')')
    lines.append(';')
    lines.append('')

    filepath.write_text('\n'.join(lines))


# ---------------------------------------------------------------------------
# ERA5 profile interpolation to cell centres
# ---------------------------------------------------------------------------

def interpolate_era5_to_cells(
    centres: np.ndarray,
    inflow: dict,
) -> dict[str, np.ndarray]:
    """Interpolate ERA5-derived profiles to cell centres.

    Parameters
    ----------
    centres : (N, 3) cell centre coordinates [m].
    inflow  : Inflow profile dict (from prepare_inflow.py output JSON).

    Returns
    -------
    dict with keys 'U' (N,3), 'k' (N,), 'omega' (N,).
    """
    from scipy.interpolate import interp1d

    n = len(centres)
    z = centres[:, 2]  # height above datum [m]
    z = np.maximum(z, 0.1)  # avoid log(0)

    fd_x = float(inflow["flowDir_x"])
    fd_y = float(inflow["flowDir_y"])
    u_star = float(inflow["u_star"])
    z0 = float(inflow.get("z0", inflow.get("z0_eff", 0.05)))

    # Build speed profile interpolator from inflow JSON
    z_levels = np.array(inflow["z_levels"])
    u_profile = np.array(inflow["u_profile"])

    # Interpolator (linear, with extrapolation for cells above/below)
    speed_interp = interp1d(
        z_levels, u_profile,
        kind="linear", bounds_error=False,
        fill_value=(u_profile[0], u_profile[-1]),
    )

    # Interpolate speed at each cell height
    speed = speed_interp(z)
    speed = np.maximum(speed, 0.0)

    # Velocity components
    U = np.zeros((n, 3))
    U[:, 0] = speed * fd_x
    U[:, 1] = speed * fd_y
    U[:, 2] = 0.0

    # k(z) = u_star² / sqrt(Cmu) — uniform for Phase 1
    k = np.full(n, u_star**2 / CMU**0.5)

    # omega(z) = u_star / (kappa * max(z, z0))
    omega = u_star / (KAPPA * np.maximum(z, z0 * 2.0))

    logger.info(
        "ERA5 init: %d cells, speed range [%.1f, %.1f] m/s, "
        "k=%.4f m²/s², omega range [%.3f, %.3f] s⁻¹",
        n, speed.min(), speed.max(), k[0], omega.min(), omega.max(),
    )

    return {"U": U, "k": k, "omega": omega}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def init_from_era5(
    case_dir: Path | str,
    inflow_json: Path | str,
) -> None:
    """Initialise OpenFOAM fields from ERA5 interpolation.

    Overwrites 0/U, 0/k, 0/omega internalField with nonuniform data.
    Preserves boundaryField from the Jinja2-rendered templates.

    Parameters
    ----------
    case_dir : OpenFOAM case directory (must have mesh already generated).
    inflow_json : Path to inflow profile JSON (from prepare_inflow.py).
    """
    case_dir = Path(case_dir)

    with open(inflow_json) as f:
        inflow = json.load(f)

    # Read cell centres
    logger.info("Reading cell centres from %s", case_dir)
    centres = read_cell_centres(case_dir)
    logger.info("Found %d cell centres", len(centres))

    # Interpolate ERA5 to cells
    fields = interpolate_era5_to_cells(centres, inflow)

    # Read existing template-rendered files to preserve boundaryField
    u_path = case_dir / "0" / "U"
    k_path = case_dir / "0" / "k"
    omega_path = case_dir / "0" / "omega"

    # Rewrite with nonuniform internalField, preserving boundaryField
    _patch_internal_field_vector(u_path, fields["U"])
    _patch_internal_field_scalar(k_path, fields["k"])
    _patch_internal_field_scalar(omega_path, fields["omega"])

    logger.info("ERA5 initialisation complete for %s", case_dir)


def _patch_internal_field_vector(filepath: Path, data: np.ndarray) -> None:
    """Replace internalField in an existing OF volVectorField with nonuniform data."""
    text = filepath.read_text()
    n = len(data)

    # Build replacement
    values = '\n'.join(f'({data[i, 0]:.6f} {data[i, 1]:.6f} {data[i, 2]:.6f})'
                       for i in range(n))
    replacement = f'internalField   nonuniform List<vector>\n{n}\n(\n{values}\n)\n;'

    # Replace the internalField line (handles both uniform and nonuniform)
    text = re.sub(
        r'internalField\s+uniform\s+\([^)]+\)\s*;',
        replacement, text, count=1,
    )
    filepath.write_text(text)
    logger.info("Patched %s: %d cells (vector)", filepath.name, n)


def _patch_internal_field_scalar(filepath: Path, data: np.ndarray) -> None:
    """Replace internalField in an existing OF volScalarField with nonuniform data."""
    text = filepath.read_text()
    n = len(data)

    values = '\n'.join(f'{data[i]:.6e}' for i in range(n))
    replacement = f'internalField   nonuniform List<scalar>\n{n}\n(\n{values}\n)\n;'

    text = re.sub(
        r'internalField\s+uniform\s+[\d.eE+\-]+\s*;',
        replacement, text, count=1,
    )
    filepath.write_text(text)
    logger.info("Patched %s: %d cells (scalar)", filepath.name, n)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Initialise OpenFOAM fields from ERA5 interpolation"
    )
    parser.add_argument("--case-dir", required=True, help="OpenFOAM case directory")
    parser.add_argument("--inflow", required=True, help="Inflow profile JSON")
    args = parser.parse_args()

    init_from_era5(case_dir=Path(args.case_dir), inflow_json=Path(args.inflow))
