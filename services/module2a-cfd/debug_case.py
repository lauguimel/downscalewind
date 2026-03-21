#!/usr/bin/env python3
"""
debug_case.py — Dump complete OpenFOAM case state to a .dbg file.

Extracts mesh, solvers, schemes, BCs (type + value + stats), constants,
fvOptions, controlDict, inflow.json, and field statistics for both
SF (precursor) and BBSF phases.

Usage:
    python debug_case.py <case_dir> [--output <path.dbg>]
    python debug_case.py data/cases/local_physics_study/case_A
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np


# ── OpenFOAM dict parser (lightweight, regex-based) ──────────────────────

def _strip_of_comments(text: str) -> str:
    """Remove C-style // and /* */ comments."""
    text = re.sub(r'//[^\n]*', '', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
    return text


def _parse_of_header(text: str) -> dict:
    """Extract FoamFile header fields."""
    m = re.search(r'FoamFile\s*\{([^}]+)\}', text, re.S)
    if not m:
        return {}
    hdr = {}
    for line in m.group(1).splitlines():
        line = line.strip().rstrip(';')
        parts = line.split(None, 1)
        if len(parts) == 2:
            hdr[parts[0]] = parts[1]
    return hdr


def _read_nonuniform_scalar(text: str) -> np.ndarray | None:
    """Extract nonuniform List<scalar> from internalField."""
    m = re.search(
        r'internalField\s+nonuniform\s+List<scalar>\s*\n(\d+)\s*\n\(',
        text,
    )
    if not m:
        return None
    n = int(m.group(1))
    start = m.end()
    end = text.index(')', start)
    return np.array([float(x) for x in text[start:end].split()[:n]])


def _read_nonuniform_vector(text: str) -> np.ndarray | None:
    """Extract nonuniform List<vector> from internalField."""
    m = re.search(
        r'internalField\s+nonuniform\s+List<vector>\s*\n(\d+)\s*\n\(',
        text,
    )
    if not m:
        return None
    n = int(m.group(1))
    start = m.end()
    vals = re.findall(r'\(([^)]+)\)', text[start:])[:n]
    return np.array([[float(x) for x in v.split()] for v in vals])


def _read_uniform_value(text: str) -> str | None:
    """Extract uniform internalField value as string."""
    m = re.search(r'internalField\s+uniform\s+(.+?)\s*;', text)
    return m.group(1).strip() if m else None


def _read_dimensions(text: str) -> str | None:
    m = re.search(r'dimensions\s+(\[[\d\s-]+\])', text)
    return m.group(1) if m else None


def _parse_boundary_field(text: str) -> dict:
    """Parse boundaryField { patch { ... } ... } into nested dict."""
    m = re.search(r'boundaryField\s*\{', text)
    if not m:
        return {}

    # Find matching closing brace
    depth = 1
    i = m.end()
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    bf_text = text[m.end():i - 1]

    # Skip entries that are clearly not patches
    _skip = {'FoamFile', 'version', 'format', 'class', 'object', 'location'}

    patches = {}
    pos = 0
    while pos < len(bf_text):
        pm = re.search(r'(\w+)\s*\{', bf_text[pos:])
        if not pm:
            break
        patch_name = pm.group(1)
        brace_start = pos + pm.end()
        depth = 1
        j = brace_start
        while j < len(bf_text) and depth > 0:
            if bf_text[j] == '{':
                depth += 1
            elif bf_text[j] == '}':
                depth -= 1
            j += 1
        block = bf_text[brace_start:j - 1]
        if patch_name not in _skip:
            patches[patch_name] = _parse_patch_block(block)
        pos = j

    return patches


def _parse_patch_block(block: str) -> dict:
    """Parse a single patch block into key-value pairs."""
    info = {}
    block = _strip_of_comments(block)

    # type
    m = re.search(r'type\s+([\w:]+)\s*;', block)
    if m:
        info['type'] = m.group(1)

    # value (uniform or nonuniform)
    m = re.search(r'\bvalue\s+uniform\s+(.+?)\s*;', block)
    if m:
        info['value'] = f"uniform {m.group(1)}"
    elif re.search(r'\bvalue\s+nonuniform', block):
        info['value'] = "nonuniform [...]"

    # inletValue
    m = re.search(r'inletValue\s+uniform\s+(.+?)\s*;', block)
    if m:
        info['inletValue'] = f"uniform {m.group(1)}"

    # gradient
    m = re.search(r'\bgradient\s+uniform\s+(.+?)\s*;', block)
    if m:
        info['gradient'] = f"uniform {m.group(1)}"

    # expression (for uniformMixed)
    m = re.search(r'expression\s+"([^"]+)"', block)
    if m:
        info['expression'] = m.group(1)

    # Prt (for wall functions)
    m = re.search(r'Prt\s+([\d.e+-]+)\s*;', block)
    if m:
        info['Prt'] = m.group(1)

    # uniformValue
    m = re.search(r'uniformValue\s+uniform\s+(.+?)\s*;', block)
    if m:
        info['uniformValue'] = f"uniform {m.group(1)}"

    # uniformGradient
    m = re.search(r'uniformGradient\s+(\w+)', block)
    if m:
        info['uniformGradient'] = m.group(1)

    return info


# ── Mesh info ─────────────────────────────────────────────────────────────

def _mesh_info(case_dir: Path) -> list[str]:
    """Extract mesh statistics from polyMesh and checkMesh log."""
    lines = ["=" * 70, "  MESH", "=" * 70, ""]

    # Boundary patches
    bnd_path = case_dir / "constant" / "polyMesh" / "boundary"
    if bnd_path.exists():
        bnd_text = bnd_path.read_text()
        patches = re.findall(r'(\w+)\n\s*\{([^}]+)\}', bnd_text)
        real_patches = [(n, b) for n, b in patches
                        if n not in ('FoamFile', 'version', 'format', 'class', 'object')]
        lines.append(f"  Patches ({len(real_patches)}):")
        for name, block in real_patches:
            ptype = ""
            nfaces = ""
            m = re.search(r'type\s+(\w+)', block)
            if m:
                ptype = m.group(1)
            m = re.search(r'nFaces\s+(\d+)', block)
            if m:
                nfaces = m.group(1)
            lines.append(f"    {name:20s}  type={ptype:12s}  nFaces={nfaces}")
        lines.append("")

    # Points / cells / faces from polyMesh files
    for name in ("points", "faces", "owner"):
        fpath = case_dir / "constant" / "polyMesh" / name
        if fpath.exists():
            text = fpath.read_text(errors='ignore')
            # Count from header note or first number
            m = re.search(r'note\s+"nPoints:\s*(\d+)', text)
            if m:
                lines.append(f"  {name}: {m.group(1)}")
                continue
            # First standalone integer after FoamFile block
            m = re.search(r'\}\s*\n(\d+)\s*\n\(', text)
            if m:
                lines.append(f"  {name}: {m.group(1)} entries")

    # checkMesh log
    log = case_dir / "log.cartesianMesh"
    if not log.exists():
        log = case_dir / "log.checkMesh"
    if log.exists():
        ck_text = log.read_text(errors='ignore')
        lines.append("")
        lines.append("  checkMesh / cartesianMesh summary:")
        for pattern, label in [
            (r'cells:\s*(\d+)', 'cells'),
            (r'points:\s*(\d+)', 'points'),
            (r'faces:\s*(\d+)', 'faces'),
            (r'internal faces:\s*(\d+)', 'internal faces'),
            (r'hexahedra:\s*(\d+)', 'hexahedra'),
            (r'prisms:\s*(\d+)', 'prisms'),
            (r'pyramids:\s*(\d+)', 'pyramids'),
            (r'polyhedra:\s*(\d+)', 'polyhedra'),
            (r'Mesh non-orthogonality.*?average:\s*([\d.]+)', 'non-orth avg'),
            (r'Mesh non-orthogonality.*?max:\s*([\d.]+)', 'non-orth max'),
            (r'Max skewness\s*=\s*([\d.]+)', 'max skewness'),
            (r'Max aspect ratio\s*=\s*([\d.]+)', 'max aspect ratio'),
            (r'Min volume\s*=\s*([\d.e+-]+)', 'min volume'),
            (r'Max volume\s*=\s*([\d.e+-]+)', 'max volume'),
            (r'Mesh OK', 'status'),
        ]:
            m = re.search(pattern, ck_text)
            if m:
                val = m.group(1) if m.lastindex else "OK"
                lines.append(f"    {label:20s}: {val}")

    # meshDict
    md_path = case_dir / "system" / "meshDict"
    if md_path.exists():
        md_text = _strip_of_comments(md_path.read_text())
        lines.append("")
        lines.append("  meshDict highlights:")
        for key in ("maxCellSize", "minCellSize", "boundaryCellSize",
                     "keepCellsIntersectingBoundary",
                     "nLayers", "thicknessRatio", "maxFirstLayerThickness"):
            m = re.search(rf'{key}\s+([\d.e+-]+)', md_text)
            if m:
                lines.append(f"    {key:35s}: {m.group(1)}")

        # Refinement zones
        for zone_m in re.finditer(r'(\w+)\s*\{[^}]*cellSize\s+([\d.e+-]+)', md_text):
            lines.append(f"    refinement {zone_m.group(1):25s}: cellSize={zone_m.group(2)}")

    lines.append("")
    return lines


# ── Solver / fvSolution ───────────────────────────────────────────────────

def _fvsolution_info(case_dir: Path) -> list[str]:
    lines = ["=" * 70, "  fvSolution (linear solvers + SIMPLE)", "=" * 70, ""]
    fpath = case_dir / "system" / "fvSolution"
    if not fpath.exists():
        lines.append("  [FILE NOT FOUND]")
        return lines

    text = _strip_of_comments(fpath.read_text())

    # Parse solver blocks
    lines.append("  Linear solvers:")
    # Find solvers { ... } top-level block
    sm = re.search(r'\bsolvers\s*\{', text)
    if sm:
        depth = 1
        si = sm.end()
        while si < len(text) and depth > 0:
            if text[si] == '{':
                depth += 1
            elif text[si] == '}':
                depth -= 1
            si += 1
        solvers_block = text[sm.end():si - 1]

        # Parse each solver entry (handle $ref and nested {})
        pos = 0
        while pos < len(solvers_block):
            em = re.search(r'(\w+)\s*\{', solvers_block[pos:])
            if not em:
                break
            name = em.group(1)
            bstart = pos + em.end()
            depth = 1
            j = bstart
            while j < len(solvers_block) and depth > 0:
                if solvers_block[j] == '{':
                    depth += 1
                elif solvers_block[j] == '}':
                    depth -= 1
                j += 1
            block = solvers_block[bstart:j - 1]
            pos = j

            # Check for $ref (e.g. "$p_rgh")
            ref_m = re.search(r'\$(\w+)', block)
            if ref_m and not re.search(r'\bsolver\s+', block):
                rtol_m = re.search(r'relTol\s+([\d.e+-]+)', block)
                rtol_val = rtol_m.group(1) if rtol_m else '?'
                lines.append(f"    {name:15s}  → ${ref_m.group(1)}; relTol={rtol_val}")
                continue

            solver = precond = tol = rtol = maxiter = '?'
            km = re.search(r'solver\s+(\w+)', block)
            if km:
                solver = km.group(1)
            km = re.search(r'preconditioner\s+(\w+)', block)
            if km:
                precond = km.group(1)
            km = re.search(r'tolerance\s+([\d.e+-]+)', block)
            if km:
                tol = km.group(1)
            km = re.search(r'relTol\s+([\d.e+-]+)', block)
            if km:
                rtol = km.group(1)
            km = re.search(r'maxIter\s+(\d+)', block)
            if km:
                maxiter = km.group(1)

            lines.append(f"    {name:15s}  solver={solver:12s}  precond={precond:6s}  tol={tol}  relTol={rtol}  maxIter={maxiter}")

    # SIMPLE block
    lines.append("")
    lines.append("  SIMPLE settings:")
    sm = re.search(r'SIMPLE\s*\{(.*?)\}', text, re.S)
    if sm:
        sblock = sm.group(1)
        for key in ("nNonOrthogonalCorrectors", "consistent", "pRefCell",
                     "pRefValue", "kMin", "epsilonMin"):
            km = re.search(rf'{key}\s+([\w.e+-]+)', sblock)
            if km:
                lines.append(f"    {key:30s}: {km.group(1)}")

        # residualControl
        rc = re.search(r'residualControl\s*\{([^}]+)\}', sblock)
        if rc:
            lines.append("    residualControl:")
            for rm in re.finditer(r'(\w+)\s+([\d.e+-]+)', rc.group(1)):
                lines.append(f"      {rm.group(1):15s}: {rm.group(2)}")

    # Relaxation
    lines.append("")
    lines.append("  Relaxation factors:")
    for section in ("fields", "equations"):
        sm = re.search(rf'{section}\s*\{{([^}}]+)\}}', text)
        if sm:
            lines.append(f"    {section}:")
            for rm in re.finditer(r'["\(]?([\w|()]+)["\)]?\s+([\d.e+-]+)', sm.group(1)):
                lines.append(f"      {rm.group(1):25s}: {rm.group(2)}")

    lines.append("")
    return lines


# ── Schemes ───────────────────────────────────────────────────────────────

def _fvschemes_info(case_dir: Path) -> list[str]:
    lines = ["=" * 70, "  fvSchemes (discretisation)", "=" * 70, ""]
    fpath = case_dir / "system" / "fvSchemes"
    if not fpath.exists():
        lines.append("  [FILE NOT FOUND]")
        return lines

    text = _strip_of_comments(fpath.read_text())

    for section in ("ddtSchemes", "gradSchemes", "divSchemes",
                     "laplacianSchemes", "interpolationSchemes",
                     "snGradSchemes"):
        m = re.search(rf'{section}\s*\{{([^}}]+)\}}', text)
        if m:
            lines.append(f"  {section}:")
            for entry in m.group(1).strip().splitlines():
                entry = entry.strip().rstrip(';')
                if entry and not entry.startswith('//'):
                    lines.append(f"    {entry}")
            lines.append("")

    return lines


# ── controlDict ───────────────────────────────────────────────────────────

def _controldict_info(case_dir: Path) -> list[str]:
    lines = ["=" * 70, "  controlDict", "=" * 70, ""]
    fpath = case_dir / "system" / "controlDict"
    if not fpath.exists():
        lines.append("  [FILE NOT FOUND]")
        return lines

    text = _strip_of_comments(fpath.read_text())

    for key in ("application", "startFrom", "startTime", "stopAt", "endTime",
                "deltaT", "writeControl", "writeInterval", "purgeWrite",
                "writeFormat", "writePrecision", "writeCompression",
                "timePrecision", "runTimeModifiable"):
        m = re.search(rf'{key}\s+(.+?)\s*;', text)
        if m:
            lines.append(f"  {key:25s}: {m.group(1).strip()}")

    # SOLVER macro
    m = re.search(r'SOLVER="([^"]+)"', text)
    if m:
        lines.append(f"  {'SOLVER (macro)':25s}: {m.group(1)}")

    # libs
    for m in re.finditer(r'libs\s+\(([^)]+)\)', text):
        lines.append(f"  {'libs':25s}: {m.group(1).strip()}")

    # Function objects
    lines.append("")
    lines.append("  Function objects:")
    for m in re.finditer(r'(\w+)\s*\{\s*type\s+(\w+)\s*;', text):
        name, ftype = m.group(1), m.group(2)
        if name in ('FoamFile',):
            continue
        lines.append(f"    {name:25s}  type={ftype}")

    lines.append("")
    return lines


# ── Physical constants ────────────────────────────────────────────────────

def _constants_info(case_dir: Path) -> list[str]:
    lines = ["=" * 70, "  PHYSICAL CONSTANTS (constant/)", "=" * 70, ""]

    # g
    g_path = case_dir / "constant" / "g"
    if g_path.exists():
        text = g_path.read_text()
        m = re.search(r'value\s+\(([^)]+)\)', text)
        if m:
            lines.append(f"  g: ({m.group(1).strip()}) m/s^2")

    # transportProperties
    tp_path = case_dir / "constant" / "transportProperties"
    if tp_path.exists():
        text = _strip_of_comments(tp_path.read_text())
        lines.append("")
        lines.append("  transportProperties:")
        for key in ("transportModel", "nu", "Pr", "Prt", "beta", "TRef",
                     "Cp", "rhoRef", "laminarPrandtl", "turbulentPrandtl"):
            m = re.search(rf'{key}\s+(?:\[[\d\s-]+\]\s+)?([\d.e+-]+|Newtonian)', text)
            if m:
                lines.append(f"    {key:25s}: {m.group(1)}")
            else:
                # Try without dimensions
                m = re.search(rf'{key}\s+([\w.e+-]+)', text)
                if m:
                    lines.append(f"    {key:25s}: {m.group(1)}")

    # physicalProperties (for SF)
    pp_path = case_dir / "constant" / "physicalProperties"
    if pp_path.exists():
        text = _strip_of_comments(pp_path.read_text())
        lines.append("")
        lines.append("  physicalProperties:")
        m = re.search(r'viscosityModel\s+(\w+)', text)
        if m:
            lines.append(f"    viscosityModel: {m.group(1)}")
        m = re.search(r'nu\s+(?:\[[\d\s-]+\]\s+)?([\d.e+-]+)', text)
        if m:
            lines.append(f"    nu: {m.group(1)}")

    # thermophysicalProperties
    thermo_path = case_dir / "constant" / "thermophysicalProperties"
    if thermo_path.exists():
        text = _strip_of_comments(thermo_path.read_text())
        lines.append("")
        lines.append("  thermophysicalProperties:")
        m = re.search(r'thermoType\s*\{([^}]+)\}', text, re.S)
        if m:
            for km in re.finditer(r'(\w+)\s+(\w+)\s*;', m.group(1)):
                lines.append(f"    thermoType.{km.group(1):15s}: {km.group(2)}")

        # equationOfState params
        for key in ("rho0", "T0", "beta", "Cp", "Hf", "molWeight"):
            m = re.search(rf'{key}\s+([\d.e+-]+)', text)
            if m:
                lines.append(f"    {key:25s}: {m.group(1)}")

    # momentumTransport
    mt_path = case_dir / "constant" / "momentumTransport"
    if mt_path.exists():
        text = _strip_of_comments(mt_path.read_text())
        lines.append("")
        lines.append("  momentumTransport:")
        for key in ("simulationType", "model", "turbulence",
                     "kMin", "epsilonMin", "nutMin", "nutMax"):
            m = re.search(rf'{key}\s+([\w.e+-]+)', text)
            if m:
                lines.append(f"    {key:25s}: {m.group(1)}")

    lines.append("")
    return lines


# ── fvOptions ─────────────────────────────────────────────────────────────

def _fvoptions_info(case_dir: Path) -> list[str]:
    lines = ["=" * 70, "  fvOptions (source terms)", "=" * 70, ""]
    fpath = case_dir / "constant" / "fvOptions"
    if not fpath.exists():
        lines.append("  [FILE NOT FOUND]")
        return lines

    text = _strip_of_comments(fpath.read_text())

    # Find all top-level entries with type
    for m in re.finditer(r'(\w+)\s*\{\s*type\s+(\w+)\s*;', text):
        name, ftype = m.group(1), m.group(2)
        if name in ('FoamFile',):
            continue
        lines.append(f"  {name}:")
        lines.append(f"    type: {ftype}")

        # Find the Coeffs block
        coeffs_name = f"{ftype}Coeffs"
        cm = re.search(rf'{coeffs_name}\s*\{{([^}}]+)\}}', text)
        if cm:
            for km in re.finditer(r'(\w+)\s+([\d.e+-]+)\s*;', cm.group(1)):
                lines.append(f"    {km.group(1):25s}: {km.group(2)}")
            sm = re.search(r'selectionMode\s+(\w+)', cm.group(1))
            if sm:
                lines.append(f"    {'selectionMode':25s}: {sm.group(1)}")
        lines.append("")

    if "fvOptions" in text and not re.search(r'type\s+\w+', text):
        lines.append("  [EMPTY — no source terms]")
        lines.append("")

    return lines


# ── Field files (0/) — BCs + statistics ───────────────────────────────────

def _field_info(case_dir: Path, time_dir: str = "0") -> list[str]:
    lines = ["=" * 70, f"  FIELDS ({time_dir}/)", "=" * 70, ""]

    tdir = case_dir / time_dir
    if not tdir.exists():
        lines.append(f"  [{time_dir}/ NOT FOUND]")
        return lines

    fields_order = ["U", "p", "p_rgh", "T", "k", "epsilon", "nut", "alphat",
                    "Phi", "phi", "LAD", "Cd"]
    all_files = sorted(f.name for f in tdir.iterdir()
                       if f.is_file() and not f.name.startswith('.'))
    # Order: known fields first, then rest
    ordered = [f for f in fields_order if f in all_files]
    ordered += [f for f in all_files if f not in ordered
                and f not in ('Cx', 'Cy', 'Cz', 'cellLevel')]

    for fname in ordered:
        fpath = tdir / fname
        try:
            text = fpath.read_text(errors='ignore')
        except Exception:
            continue

        # Skip non-OF files
        if 'FoamFile' not in text:
            continue

        hdr = _parse_of_header(text)
        dims = _read_dimensions(text)

        lines.append(f"  --- {fname} ---")
        if dims:
            lines.append(f"    dimensions: {dims}")
        if hdr.get('class'):
            lines.append(f"    class: {hdr['class']}")

        # Internal field
        uniform_val = _read_uniform_value(text)
        if uniform_val is not None:
            lines.append(f"    internalField: uniform {uniform_val}")
        else:
            # Try nonuniform
            scalar_data = _read_nonuniform_scalar(text)
            vector_data = _read_nonuniform_vector(text)
            if scalar_data is not None:
                lines.append(f"    internalField: nonuniform List<scalar> ({len(scalar_data)} values)")
                lines.append(f"      min={scalar_data.min():.6g}  max={scalar_data.max():.6g}  "
                             f"mean={scalar_data.mean():.6g}  std={scalar_data.std():.4g}")
                # Percentiles
                p5, p50, p95 = np.percentile(scalar_data, [5, 50, 95])
                lines.append(f"      p5={p5:.6g}  p50={p50:.6g}  p95={p95:.6g}")
            elif vector_data is not None:
                mag = np.linalg.norm(vector_data, axis=1)
                lines.append(f"    internalField: nonuniform List<vector> ({len(vector_data)} values)")
                lines.append(f"      |U|: min={mag.min():.4g}  max={mag.max():.4g}  mean={mag.mean():.4g}")
                for i, comp in enumerate(['x', 'y', 'z']):
                    c = vector_data[:, i]
                    lines.append(f"      U{comp}: min={c.min():.4g}  max={c.max():.4g}  mean={c.mean():.4g}")

        # Boundary conditions
        bf = _parse_boundary_field(text)
        if bf:
            lines.append("    boundaryField:")
            for pname, pinfo in bf.items():
                parts = [f"type={pinfo.get('type', '?')}"]
                for key in ('value', 'inletValue', 'uniformValue', 'gradient',
                            'uniformGradient', 'expression', 'Prt'):
                    if key in pinfo:
                        parts.append(f"{key}={pinfo[key]}")
                lines.append(f"      {pname:20s}  {', '.join(parts)}")

        lines.append("")

    return lines


# ── Inflow JSON ───────────────────────────────────────────────────────────

def _inflow_info(case_dir: Path) -> list[str]:
    lines = ["=" * 70, "  INFLOW (inflow.json)", "=" * 70, ""]

    inflow_json = case_dir / "inflow.json"
    if not inflow_json.exists():
        inflow_json = case_dir.parent / "inflow.json"
    if not inflow_json.exists():
        lines.append("  [inflow.json NOT FOUND]")
        return lines

    with open(inflow_json) as f:
        inflow = json.load(f)

    # Scalar keys
    scalar_keys = [k for k, v in inflow.items()
                   if isinstance(v, (int, float, str, bool))]
    array_keys = [k for k, v in inflow.items()
                  if isinstance(v, list)]

    for k in sorted(scalar_keys):
        v = inflow[k]
        if isinstance(v, float):
            lines.append(f"  {k:25s}: {v:.6g}")
        else:
            lines.append(f"  {k:25s}: {v}")

    lines.append("")
    for k in sorted(array_keys):
        arr = np.array(inflow[k])
        if arr.ndim == 1 and len(arr) <= 20:
            lines.append(f"  {k} ({len(arr)} values):")
            lines.append(f"    {arr}")
        elif arr.ndim == 1:
            lines.append(f"  {k} ({len(arr)} values): min={arr.min():.4g} max={arr.max():.4g}")

    # Derived quantities
    lines.append("")
    lines.append("  Derived:")
    u_hub = inflow.get('u_hub', 0)
    fdx = inflow.get('flowDir_x', 0)
    fdy = inflow.get('flowDir_y', 0)
    lines.append(f"    Ux_hub = u_hub * flowDir_x = {u_hub * fdx:.4f} m/s")
    lines.append(f"    Uy_hub = u_hub * flowDir_y = {u_hub * fdy:.4f} m/s")
    u_star = inflow.get('u_star', 0)
    k_init = u_star ** 2 / 0.09 ** 0.5
    lines.append(f"    k_init = u_star^2/Cmu^0.5 = {k_init:.6f} m^2/s^2")
    eps_init = 0.09 ** 0.75 * k_init ** 1.5 / (0.41 * 10)
    lines.append(f"    eps_init (y_wall=10m)      = {eps_init:.6e} m^2/s^3")

    if 'T_profile' in inflow and 'z_levels' in inflow:
        T = np.array(inflow['T_profile'])
        z = np.array(inflow['z_levels'])
        lines.append(f"    T_profile: {T.min():.2f} — {T.max():.2f} K over z={z.min():.0f} — {z.max():.0f} m")

    lines.append("")
    return lines


# ── Solver logs ───────────────────────────────────────────────────────────

def _solver_log_info(case_dir: Path) -> list[str]:
    lines = ["=" * 70, "  SOLVER LOGS", "=" * 70, ""]

    for log_name in sorted(case_dir.glob("log.*")):
        if log_name.stat().st_size == 0:
            continue
        try:
            text = log_name.read_text(errors='ignore')
        except Exception:
            continue

        # Last 30 lines summary
        all_lines = text.splitlines()
        n = len(all_lines)

        lines.append(f"  --- {log_name.name} ({n} lines) ---")

        # Execution time
        m = re.search(r'ExecutionTime\s*=\s*([\d.]+)\s*s', text)
        if m:
            lines.append(f"    ExecutionTime: {float(m.group(1)):.1f} s")

        # Final residuals (last occurrence of each)
        for field in ('Ux', 'Uy', 'Uz', 'p', 'p_rgh', 'k', 'epsilon', 'T'):
            matches = re.findall(
                rf'Solving for {field},.*?Final residual = ([\d.e+-]+)',
                text,
            )
            if matches:
                lines.append(f"    {field:10s} final residual: {matches[-1]}")

        # Continuity errors (last)
        cont = re.findall(r'time step continuity errors.*?global = ([\d.e+-]+)', text)
        if cont:
            lines.append(f"    continuity global (last): {cont[-1]}")

        # GAMG info
        gamg = re.findall(r'GAMG:\s*Solving for (\w+).*?Final residual = ([\d.e+-]+)', text)
        if gamg:
            last_gamg = gamg[-1]
            lines.append(f"    GAMG({last_gamg[0]}) last residual: {last_gamg[1]}")

        # Warnings / errors
        warnings = set()
        for wm in re.finditer(r'(Warning|Error|FOAM FATAL|bounding \w+)', text):
            warnings.add(wm.group(0))
        if warnings:
            lines.append(f"    WARNINGS: {', '.join(sorted(warnings))}")

        # Bounding messages (count)
        bounding = re.findall(r'bounding (\w+)', text)
        if bounding:
            from collections import Counter
            bc = Counter(bounding)
            parts = [f"{k}({v})" for k, v in bc.most_common()]
            lines.append(f"    bounding events: {', '.join(parts)}")

        lines.append("")

    return lines


# ── postProcessing ────────────────────────────────────────────────────────

def _postprocessing_info(case_dir: Path) -> list[str]:
    lines = ["=" * 70, "  POSTPROCESSING", "=" * 70, ""]

    pp = case_dir / "postProcessing"
    if not pp.exists():
        lines.append("  [postProcessing/ NOT FOUND]")
        return lines

    # fieldMinMax
    fmm_dir = pp / "fieldMinMax" / "0"
    if fmm_dir.exists():
        lines.append("  fieldMinMax (last values):")
        for dat in sorted(fmm_dir.glob("*.dat")):
            try:
                data_lines = [l for l in dat.read_text().splitlines()
                              if l.strip() and not l.startswith('#')]
                if data_lines:
                    last = data_lines[-1]
                    lines.append(f"    {dat.stem}: {last[:120]}")
            except Exception:
                pass
        lines.append("")

    # volAverages
    va_file = pp / "volAverages" / "0" / "volFieldValue.dat"
    if va_file.exists():
        lines.append("  volAverages (last 5 entries):")
        try:
            data_lines = [l for l in va_file.read_text().splitlines()
                          if l.strip() and not l.startswith('#')]
            for l in data_lines[-5:]:
                lines.append(f"    {l[:120]}")
        except Exception:
            pass
        lines.append("")

    # solverInfo
    si_dir = pp / "solverInfo" / "0"
    if si_dir.exists():
        lines.append("  solverInfo (last entries):")
        for dat in sorted(si_dir.glob("*.dat")):
            try:
                data_lines = [l for l in dat.read_text().splitlines()
                              if l.strip() and not l.startswith('#')]
                if data_lines:
                    lines.append(f"    {dat.stem}: {data_lines[-1][:120]}")
            except Exception:
                pass
        lines.append("")

    return lines


# ── Converged fields (latest time dir) ────────────────────────────────────

def _latest_time_dir(case_dir: Path) -> str | None:
    """Find latest time directory."""
    time_dirs = []
    for d in case_dir.iterdir():
        if d.is_dir() and d.name != '0':
            try:
                float(d.name)
                time_dirs.append(d)
            except ValueError:
                pass
    if not time_dirs:
        return None
    return max(time_dirs, key=lambda d: float(d.name)).name


# ── Main ──────────────────────────────────────────────────────────────────

def dump_debug(case_dir: Path, output: Path | None = None) -> Path:
    """Generate .dbg file for an OpenFOAM case."""
    if output is None:
        output = case_dir / f"{case_dir.name}.dbg"

    sections: list[str] = []

    # Header
    sections.append("=" * 70)
    sections.append(f"  DEBUG DUMP — {case_dir.name}")
    sections.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sections.append(f"  Case path: {case_dir.resolve()}")
    sections.append("=" * 70)
    sections.append("")

    # File inventory
    sections.append("  File inventory:")
    for subdir in ("0", "constant", "system"):
        sd = case_dir / subdir
        if sd.exists():
            files = sorted(f.relative_to(case_dir) for f in sd.rglob("*")
                           if f.is_file() and 'polyMesh' not in str(f)
                           and 'boundaryData' not in str(f)
                           and 'triSurface' not in str(f))
            sections.append(f"    {subdir}/ ({len(files)} files): {', '.join(f.name for f in files[:20])}")
    sections.append("")

    # All sections
    sections.extend(_mesh_info(case_dir))
    sections.extend(_controldict_info(case_dir))
    sections.extend(_fvsolution_info(case_dir))
    sections.extend(_fvschemes_info(case_dir))
    sections.extend(_constants_info(case_dir))
    sections.extend(_fvoptions_info(case_dir))
    sections.extend(_inflow_info(case_dir))
    sections.extend(_field_info(case_dir, "0"))

    # Latest time dir fields (if different from 0)
    latest = _latest_time_dir(case_dir)
    if latest and latest != "0":
        sections.append(f"  (converged fields at t={latest})")
        sections.extend(_field_info(case_dir, latest))

    sections.extend(_solver_log_info(case_dir))
    sections.extend(_postprocessing_info(case_dir))

    # Write
    output.write_text("\n".join(sections))
    print(f"Debug dump written to: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Dump complete OpenFOAM case state to .dbg file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python debug_case.py data/cases/local_physics_study/case_A
              python debug_case.py data/cases/stability/case_B --output /tmp/case_B.dbg
        """),
    )
    parser.add_argument("case_dir", type=Path, help="OpenFOAM case directory")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output .dbg file (default: <case_dir>/<case_name>.dbg)")
    args = parser.parse_args()

    if not args.case_dir.exists():
        print(f"Error: {args.case_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    dump_debug(args.case_dir, args.output)


if __name__ == "__main__":
    main()
