"""
convert_fields.py — Convert simpleFoam (incompressible) fields to buoyantSimpleFoam IC.

When initialising a buoyantSimpleFoam (compressible) case from a converged
simpleFoam (incompressible) solution, two conversions are needed:

  p [m²/s²]  →  p [Pa]     : p_Pa = p_kinematic * rho_ref
  T absent   →  T [K]      : uniform T_ref (or ERA5 vertical profile)

U, k, epsilon are identical in both solvers and are copied as-is.
p_rgh is not copied — buoyantSimpleFoam recomputes it from p and g·h at startup.

Usage
-----
    python convert_fields.py \\
        --source  data/cases/perdigao_250m_XXX          # simpleFoam case dir
        --target  data/cases/perdigao_100m_XXX/0        # destination 0/ dir
        --rho-ref 1.225                                  # kg/m³
        --T-ref   290.5                                  # K (ERA5 hub temperature)
        [--inflow-json data/inflow_profiles/XXX.json]   # optional: vertical T profile

The target directory must exist (it is the rendered 0/ from the buoyantSimpleFoam case).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OpenFOAM ASCII field I/O helpers
# ---------------------------------------------------------------------------

def _read_scalar_field(field_path: Path) -> tuple[str, list[float]]:
    """Read an OpenFOAM uniform or non-uniform volScalarField.

    Returns
    -------
    (header, values) where header is the raw FoamFile+dimensions+internalField
    preamble and values is the list of scalar cell values.
    """
    text = field_path.read_text()

    # Extract internalField section
    m = re.search(
        r"(internalField\s+)(nonuniform List<scalar>\s*\n\s*\d+\s*\n\()([^)]+)\)",
        text, re.DOTALL
    )
    if m:
        values_str = m.group(3).strip().split()
        values = [float(v) for v in values_str]
        return text, values

    # uniform case
    m_uniform = re.search(r"internalField\s+uniform\s+([0-9eE+\-.]+)", text)
    if m_uniform:
        return text, [float(m_uniform.group(1))]

    raise ValueError(f"Cannot parse internalField in {field_path}")


def _write_scalar_field(field_path: Path, original_text: str, new_values: list[float]) -> None:
    """Write a volScalarField with updated internalField values."""
    n = len(new_values)
    values_str = "\n".join(f"{v:.6e}" for v in new_values)
    new_internal = f"internalField   nonuniform List<scalar>\n{n}\n(\n{values_str}\n);"

    # Replace internalField block (handles both uniform and nonuniform)
    result = re.sub(
        r"internalField\s+(?:nonuniform List<scalar>\s*\n\s*\d+\s*\n\([^)]*\)|uniform\s+[0-9eE+\-.]+)\s*;",
        new_internal,
        original_text,
        count=1,
        flags=re.DOTALL,
    )
    field_path.write_text(result)


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def _find_latest_time(case_dir: Path) -> Path:
    """Return the latest time directory in a converged OF case."""
    time_dirs = []
    for d in case_dir.iterdir():
        if d.is_dir():
            try:
                time_dirs.append((float(d.name), d))
            except ValueError:
                pass
    if not time_dirs:
        raise FileNotFoundError(f"No time directories found in {case_dir}")
    time_dirs.sort(key=lambda x: x[0])
    t, d = time_dirs[-1]
    logger.info("Using latest time directory: %s (t=%.0f)", d, t)
    return d


def convert_fields(
    source_case: Path,
    target_0: Path,
    rho_ref: float = 1.225,
    T_ref: float = 300.0,
    inflow_json: Path | None = None,
) -> None:
    """Convert simpleFoam fields to buoyantSimpleFoam initial conditions.

    Parameters
    ----------
    source_case : Path
        Root of the converged simpleFoam case.
    target_0 : Path
        The ``0/`` directory of the buoyantSimpleFoam case to populate.
    rho_ref : float
        Reference air density [kg/m³].
    T_ref : float
        Reference temperature [K] for uniform T initialisation.
    inflow_json : Path, optional
        Inflow profile JSON; if present, used to get T_ref from ``T_ref`` key.
    """
    source_time = _find_latest_time(source_case)
    target_0.mkdir(parents=True, exist_ok=True)

    # Override T_ref from inflow JSON if available
    if inflow_json and inflow_json.exists():
        with open(inflow_json) as f:
            inflow = json.load(f)
        T_ref = float(inflow.get("T_ref", T_ref))
        logger.info("T_ref from inflow JSON: %.2f K", T_ref)

    # ------------------------------------------------------------------
    # 1. Copy U, k, epsilon unchanged
    # ------------------------------------------------------------------
    for field_name in ("U", "k", "epsilon"):
        src = source_time / field_name
        dst = target_0 / field_name
        if src.exists():
            shutil.copy2(src, dst)
            logger.info("Copied %s → %s", src.name, dst)
        else:
            logger.warning("Field %s not found in %s — skipping", field_name, source_time)

    # ------------------------------------------------------------------
    # 2. Convert p: [m²/s²] kinematic → [Pa] absolute
    #    p_Pa = p_kinematic * rho_ref
    # ------------------------------------------------------------------
    p_src = source_time / "p"
    p_dst = target_0 / "p"
    if p_src.exists() and p_dst.exists():
        text, values = _read_scalar_field(p_src)
        if len(values) == 1:
            # uniform: stay uniform but update dimensions comment only
            p_Pa_values = [values[0] * rho_ref]
        else:
            p_Pa_values = [v * rho_ref for v in values]

        # Read target p template (has correct dimensions [1 -1 -2 0 0 0 0])
        p_template_text = p_dst.read_text()
        _write_scalar_field(p_dst, p_template_text, p_Pa_values)
        logger.info(
            "Converted p: %.3f → %.1f Pa (×%.3f rho_ref), %d cells",
            np.mean(values), np.mean(p_Pa_values), rho_ref, len(p_Pa_values),
        )
    else:
        if not p_src.exists():
            logger.warning("p field not found in source %s", source_time)
        if not p_dst.exists():
            logger.warning("p template not found in target %s", target_0)

    # ------------------------------------------------------------------
    # 3. Create T: uniform T_ref (buoyantSimpleFoam template already exists)
    #    The template has correct BC; we only update internalField.
    # ------------------------------------------------------------------
    T_dst = target_0 / "T"
    if T_dst.exists():
        T_text = T_dst.read_text()
        # Replace internalField with uniform T_ref (no cell-by-cell data needed)
        T_text_new = re.sub(
            r"internalField\s+uniform\s+[0-9eE+\-.]+",
            f"internalField   uniform {T_ref:.2f}",
            T_text,
        )
        T_dst.write_text(T_text_new)
        logger.info("Set T internalField = %.2f K (uniform)", T_ref)
    else:
        logger.warning("T template not found in target %s — cannot set temperature", target_0)

    logger.info("Field conversion complete: %s → %s", source_case, target_0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert simpleFoam fields to buoyantSimpleFoam initial conditions."
    )
    p.add_argument("--source", required=True, type=Path,
                   help="Root of the converged simpleFoam case directory.")
    p.add_argument("--target", required=True, type=Path,
                   help="Destination 0/ directory of the buoyantSimpleFoam case.")
    p.add_argument("--rho-ref", type=float, default=1.225,
                   help="Reference air density [kg/m³] (default: 1.225).")
    p.add_argument("--T-ref", type=float, default=300.0,
                   help="Reference temperature [K] (default: 300).")
    p.add_argument("--inflow-json", type=Path, default=None,
                   help="Inflow profile JSON to read T_ref from.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )
    convert_fields(
        source_case=args.source,
        target_0=args.target,
        rho_ref=args.rho_ref,
        T_ref=args.T_ref,
        inflow_json=args.inflow_json,
    )
