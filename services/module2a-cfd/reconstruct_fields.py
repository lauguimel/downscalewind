#!/usr/bin/env python3
"""
reconstruct_fields.py — Assemble parallel OpenFOAM fields without reconstructPar.

Reads internalField directly from processor*/TIME/FIELD (no BC evaluation),
assembles via cellProcAddressing, and writes assembled fields back to TIME/.

Avoids the "Object phi does not exist in objectRegistry" error caused by
uniformMixed BCs with expression "neg/pos(phi)" during reconstructPar.

Usage (from case directory):
    python3 reconstruct_fields.py [--case-dir .] [--time latest]
                                   [--fields U T k p_rgh] [--write-foam]
                                   [--zarr output.zarr]
"""

from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

import numpy as np


# ── OpenFOAM parsing ─────────────────────────────────────────────────────────

def _get_proc_dirs(case_dir: Path) -> list[Path]:
    dirs = sorted(
        case_dir.glob("processor*"),
        key=lambda p: int(re.search(r"\d+", p.name).group()),
    )
    if not dirs:
        raise RuntimeError(f"No processor* directories in {case_dir}")
    return dirs


def _find_latest_time(proc0: Path) -> str:
    times = []
    for p in proc0.iterdir():
        if p.is_dir():
            try:
                times.append(float(p.name))
            except ValueError:
                pass
    if not times:
        raise RuntimeError(f"No time directories in {proc0}")
    t = max(times)
    return str(int(t)) if t == int(t) else str(t)


def _read_addressing(proc_dir: Path) -> np.ndarray:
    """Read cellProcAddressing: local_cell -> global_cell index."""
    path = proc_dir / "constant" / "polyMesh" / "cellProcAddressing"
    text = path.read_text()
    m = re.search(r"(\d+)\s*\(", text)
    if not m:
        raise ValueError(f"Cannot parse {path}")
    n = int(m.group(1))
    start = text.index("(", m.start()) + 1
    end = text.rindex(")")
    arr = np.fromstring(text[start:end], dtype=np.int64, sep="\n")
    return arr[:n]


def _read_field(path: Path) -> tuple[np.ndarray | float, bool]:
    """
    Parse internalField from an OpenFOAM volField file.
    Returns (data, is_vector).
    - uniform scalar  → (float, False)
    - uniform vector  → (ndarray shape (3,), True)
    - nonuniform scalar → (ndarray shape (N,), False)
    - nonuniform vector → (ndarray shape (N, 3), True)
    """
    text = path.read_text()

    # Determine type from FoamFile class
    cls_m = re.search(r"class\s+(\S+?);", text)
    is_vec = bool(cls_m and "Vector" in cls_m.group(1))

    # Find internalField block (stops at boundaryField or end)
    m = re.search(r"\binternalField\s+", text)
    if not m:
        raise ValueError(f"No internalField in {path}")
    content = text[m.end():].lstrip()

    # uniform
    if content.startswith("uniform"):
        v = content.split("uniform", 1)[1].strip().split(";")[0].strip()
        if v.startswith("("):
            vals = list(map(float, v.strip("()").split()))
            return np.array(vals, dtype=np.float64), True
        return float(v), False

    # nonuniform
    n_m = re.search(r"(\d+)\s*\(", content)
    if not n_m:
        raise ValueError(f"Cannot find list size in {path}")
    n = int(n_m.group(1))

    if is_vec:
        # Extract all (x y z) tuples
        tuples = re.findall(
            r"\(\s*([-+eE\d.]+)\s+([-+eE\d.]+)\s+([-+eE\d.]+)\s*\)",
            content[n_m.start():],
        )
        arr = np.array(
            [[float(a), float(b), float(c)] for a, b, c in tuples[:n]],
            dtype=np.float64,
        )
        if len(arr) != n:
            raise ValueError(f"{path}: expected {n} vectors, parsed {len(arr)}")
        return arr, True
    else:
        # Scalars: text between outer ( and matching )
        s = content.index("(", n_m.start()) + 1
        e = content.index("\n)", s)
        arr = np.fromstring(content[s:e], dtype=np.float64, sep="\n")
        return arr[:n], False


# ── Reconstruction ────────────────────────────────────────────────────────────

def reconstruct(
    case_dir: str | Path = ".",
    time_name: str = "latest",
    fields: tuple[str, ...] = ("U", "T", "q", "k", "epsilon", "nut", "p", "p_rgh"),
) -> tuple[dict[str, np.ndarray], str]:
    """
    Reconstruct volume fields from parallel decomposition.
    Returns (field_dict, time_str).
    """
    case = Path(case_dir)
    procs = _get_proc_dirs(case)

    if time_name in ("latest", "latestTime"):
        time_name = _find_latest_time(procs[0])
    print(f"[reconstruct] {len(procs)} procs, time={time_name}", flush=True)

    addrs = [_read_addressing(p) for p in procs]
    n_cells = int(max(a.max() for a in addrs)) + 1
    print(f"[reconstruct] {n_cells} cells total", flush=True)

    result: dict[str, np.ndarray] = {}

    for fname in fields:
        path0 = procs[0] / time_name / fname
        if not path0.exists():
            print(f"  {fname}: not in processor0 — skipped", flush=True)
            continue

        data0, is_vec = _read_field(path0)

        # Uniform value in all processors → broadcast
        if isinstance(data0, (int, float, np.floating)):
            if is_vec:
                arr = np.tile(data0, (n_cells, 1))
            else:
                arr = np.full(n_cells, float(data0))
        else:
            arr = np.zeros((n_cells, 3) if is_vec else (n_cells,), dtype=np.float64)
            # Proc 0 already read
            arr[addrs[0]] = data0

            for proc, addr in zip(procs[1:], addrs[1:]):
                fp = proc / time_name / fname
                data, _ = _read_field(fp)
                if isinstance(data, (int, float, np.floating)):
                    arr[addr] = data
                else:
                    arr[addr] = data

        result[fname] = arr
        summary = f"shape={arr.shape}, |mean|={float(np.abs(arr).mean()):.4g}"
        print(f"  {fname}: {summary}", flush=True)

    return result, time_name


# ── Output ────────────────────────────────────────────────────────────────────

_DIMS = {
    "U":     "[0 1 -1 0 0 0 0]",
    "T":     "[0 0 0 1 0 0 0]",
    "k":     "[0 2 -2 0 0 0 0]",
    "epsilon": "[0 2 -3 0 0 0 0]",
    "p_rgh": "[0 2 -2 0 0 0 0]",
    "p":     "[0 2 -2 0 0 0 0]",
}


def write_foam_field(
    case_dir: str | Path,
    time_name: str,
    fname: str,
    arr: np.ndarray,
) -> None:
    """Write assembled field to TIME/FIELD in minimal OpenFOAM ASCII format."""
    is_vec = arr.ndim == 2
    cls = "volVectorField" if is_vec else "volScalarField"
    dims = _DIMS.get(fname, "[0 0 0 0 0 0 0]")
    n = len(arr)

    out_dir = Path(case_dir) / time_name
    out_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "FoamFile\n{\n",
        "    version     2.0;\n",
        "    format      ascii;\n",
        f"    class       {cls};\n",
        f"    location    \"{time_name}\";\n",
        f"    object      {fname};\n",
        "}\n",
        f"dimensions      {dims};\n\n",
    ]

    if is_vec:
        lines.append(f"internalField   nonuniform List<vector>\n{n}\n(\n")
        for row in arr:
            lines.append(f"({row[0]:.8g} {row[1]:.8g} {row[2]:.8g})\n")
    else:
        lines.append(f"internalField   nonuniform List<scalar>\n{n}\n(\n")
        for v in arr:
            lines.append(f"{v:.8g}\n")

    lines.append(")\n;\n\nboundaryField\n{\n}\n")

    (out_dir / fname).write_text("".join(lines))
    print(f"  wrote {out_dir/fname}", flush=True)


def write_zarr(
    zarr_path: str | Path,
    data: dict[str, np.ndarray],
    time_name: str,
) -> None:
    """Export assembled fields to a Zarr store."""
    import zarr  # type: ignore

    store = zarr.open(str(zarr_path), mode="w")
    grp = store.require_group(time_name)
    for fname, arr in data.items():
        grp[fname] = arr
        print(f"  zarr/{time_name}/{fname}: shape={arr.shape}", flush=True)
    print(f"[zarr] written to {zarr_path}", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--case-dir", default=".", help="OpenFOAM case directory")
    pa.add_argument("--time", default="latest",
                    help="Time directory (default: latest)")
    pa.add_argument("--fields", nargs="+",
                    default=["U", "T", "q", "k", "epsilon", "nut", "p", "p_rgh"],
                    help="Fields to reconstruct")
    pa.add_argument("--write-foam", action="store_true",
                    help="Write assembled fields back to OpenFOAM ASCII format")
    pa.add_argument("--zarr", default=None, metavar="PATH",
                    help="Also export to Zarr store at this path")
    args = pa.parse_args()

    data, time_name = reconstruct(
        case_dir=args.case_dir,
        time_name=args.time,
        fields=tuple(args.fields),
    )

    if args.write_foam:
        print(f"\n[write-foam] Writing to {args.case_dir}/{time_name}/...")
        for fname, arr in data.items():
            write_foam_field(args.case_dir, time_name, fname, arr)

    if args.zarr:
        print(f"\n[zarr] Exporting to {args.zarr}...")
        write_zarr(args.zarr, data, time_name)

    if not args.write_foam and not args.zarr:
        print("\nTip: add --write-foam to write OF fields, --zarr PATH to export Zarr")

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
