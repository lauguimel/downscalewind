#!/usr/bin/env python3
"""Build wall roughness canary variants from one prepared OpenFOAM case.

The variants keep the same mesh, inflow.json and lateral boundaryData, then
change only terrain/bottom wall roughness settings:

* z0 wall value: e.g. 0.005, 0.01, 0.03, 0.05 m
* nut wall function: atmNutkWallFunction or atmNutUWallFunction

This isolates whether near-surface wind damping is driven by excessive wall
roughness or by the k-driven rough wall closure.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path
from typing import Iterable


MODULE_DIR = Path(__file__).resolve().parents[1]


def load_inflow(case_dir: Path) -> dict:
    path = case_dir / "inflow.json"
    if not path.exists():
        raise FileNotFoundError(f"inflow.json not found in {case_dir}")
    return json.loads(path.read_text())


def finite_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def read_end_time(case_dir: Path, default: str = "500") -> str:
    path = case_dir / "system" / "controlDict"
    if not path.exists():
        return default
    match = re.search(r"\bendTime\s+([^;]+);", path.read_text(errors="replace"))
    return match.group(1).strip() if match else default


def is_time_dir(path: Path) -> bool:
    if not path.is_dir() or path.name == "0":
        return False
    try:
        float(path.name)
    except ValueError:
        return False
    return True


def copy_dir(src: Path, dst: Path, *, symlink_static: bool) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if symlink_static and item.name in {"polyMesh", "boundaryData", "triSurface"}:
                target.symlink_to(item.resolve(), target_is_directory=True)
            else:
                shutil.copytree(item, target, symlinks=True)
        else:
            shutil.copy2(item, target)


def copy_base_case(base_case: Path, dst: Path, *, symlink_static: bool, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"{dst} exists; pass --overwrite to replace it")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    for item in base_case.iterdir():
        name = item.name
        if name.startswith("processor") or name.startswith("log.") or is_time_dir(item):
            continue
        target = dst / name
        if item.is_dir():
            copy_dir(item, target, symlink_static=symlink_static)
        else:
            shutil.copy2(item, target)


def slug_float(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def patch_block(text: str, patch: str, updater) -> tuple[str, bool, bool]:
    match = re.search(rf"\b{re.escape(patch)}\s*\{{", text)
    if not match:
        return text, False, False
    depth = 0
    body_start = match.end()
    body_end = None
    for i in range(match.end() - 1, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                body_end = i
                break
    if body_end is None:
        raise ValueError(f"Unclosed patch block {patch}")

    body = text[body_start:body_end]
    new_body = updater(body)
    return text[:body_start] + new_body + text[body_end:], True, new_body != body


def replace_uniform_z0(body: str, z0: float) -> str:
    repl = f"z0              uniform {z0:.6g};"
    new, n = re.subn(r"z0\s+uniform\s+[\d.eE+-]+\s*;", repl, body)
    if n:
        return new

    # Fallback for mapped z0 blocks. This is intentionally conservative and
    # expects the mapped block used by our templates.
    new, n = re.subn(
        r"z0\s*\{\s*type\s+mappedFile;\s*mapMethod\s+\w+;\s*fieldTable\s+z0;\s*\}",
        repl,
        body,
        flags=re.S,
    )
    if n:
        return new
    raise ValueError("Could not find z0 entry in wall patch block")


def patch_wall_fields(case_dir: Path, z0_wall: float, nut_wall_function: str) -> None:
    if nut_wall_function not in {"atmNutkWallFunction", "atmNutUWallFunction"}:
        raise ValueError(f"Unsupported nut wall function: {nut_wall_function}")

    for field in ("nut", "epsilon"):
        path = case_dir / "0" / field
        text = path.read_text(errors="replace")
        found_any = False

        for patch in ("terrain", "bottom"):
            def update(body: str, field: str = field) -> str:
                out = replace_uniform_z0(body, z0_wall)
                if field == "nut":
                    out, n = re.subn(
                        r"type\s+atmNut[Uk]?WallFunction\s*;",
                        f"type            {nut_wall_function};",
                        out,
                    )
                    if n == 0:
                        raise ValueError("Could not find atmNut* wall function type")
                return out

            text, found, _changed = patch_block(text, patch, update)
            found_any = found_any or found

        path.write_text(text)
        if not found_any:
            raise ValueError(f"No terrain/bottom wall patch found in {path}")


def make_variants(
    base_case: Path,
    output_dir: Path,
    z0_values: list[float],
    wall_functions: list[str],
    *,
    overwrite: bool,
    symlink_static: bool,
) -> list[dict]:
    out: list[dict] = []
    cases_root = output_dir / "cases"
    for wall_fn in wall_functions:
        short = "nutk" if wall_fn == "atmNutkWallFunction" else "nutu"
        for z0 in z0_values:
            name = f"{short}_z0_{slug_float(z0)}"
            case_name = f"case_ts000_{name}"
            dst = cases_root / case_name
            copy_base_case(base_case, dst, symlink_static=symlink_static, overwrite=overwrite)
            patch_wall_fields(dst, z0, wall_fn)
            shutil.copy2(base_case / "inflow.json", dst / "inflow.json")
            out.append(
                {
                    "name": name,
                    "case_name": case_name,
                    "case_dir": str(dst),
                    "z0_wall_m": float(z0),
                    "nut_wall_function": wall_fn,
                    "description": f"{wall_fn}, z0_wall={z0:g} m; inflow unchanged",
                }
            )
    return out


def write_export_script(output_dir: Path, variants: list[dict], inflow: dict, time_name: str) -> None:
    era5 = inflow.get("era5_source")
    timestamp = inflow.get("timestamp")
    lat = finite_float(inflow.get("site_lat"))
    lon = finite_float(inflow.get("site_lon"))
    script = output_dir / "export_and_audit_wall_z0.sh"
    audit_csv = output_dir / "wall_z0_wind_audit.csv"
    summary_csv = output_dir / "wall_z0_wind_audit_summary.csv"
    wall_csv = output_dir / "wall_z0_audit.csv"

    if not era5 or not timestamp or not math.isfinite(lat) or not math.isfinite(lon):
        script.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
echo "Cannot export automatically: inflow.json lacks era5_source, timestamp, site_lat or site_lon." >&2
exit 2
"""
        )
        script.chmod(script.stat().st_mode | 0o755)
        return

    export_py = MODULE_DIR / "export_to_grid_zarr_v2.py"
    audit_py = MODULE_DIR / "analysis" / "audit_v2_teacher_wind.py"
    wall_audit_py = MODULE_DIR / "analysis" / "audit_wall_z0.py"
    commands: list[str] = []
    for v in variants:
        case_dir = Path(v["case_dir"]).resolve()
        commands.append(
            f'if [ -f "{case_dir / time_name / "U"}" ]; then\n'
            f'  python3 "{export_py.resolve()}" '
            f'--case-dir "{case_dir}" '
            f'--site-id "{v["name"]}" '
            f'--site-lat "{lat:.8f}" '
            f'--site-lon "{lon:.8f}" '
            f'--era5-zarr "{era5}" '
            f'--timestamp "{timestamp}" '
            f'--time "{time_name}" '
            f'--output "{case_dir / "grid.zarr"}" '
            f'--overwrite\n'
            f'  SOLVED_CASES+=("{case_dir}")\n'
            f'else\n'
            f'  echo "skip export: {case_dir} has no {time_name}/U" >&2\n'
            f'fi'
        )

    script.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail

SOLVED_CASES=()

{chr(10).join(commands)}

python3 "{audit_py.resolve()}" \\
  --data-dir "{(output_dir / "cases").resolve()}" \\
  --output "{audit_csv.resolve()}" \\
  --summary-output "{summary_csv.resolve()}" \\
  --heights 2,10,20,50,100 \\
  --crop-km 2

if [ "${{#SOLVED_CASES[@]}}" -gt 0 ]; then
  python3 "{wall_audit_py.resolve()}" \\
    "${{SOLVED_CASES[@]}}" \\
    --time "{time_name}" \\
    --output "{wall_csv.resolve()}"
else
  echo "No solved cases for wall audit" >&2
fi
"""
    )
    script.chmod(script.stat().st_mode | 0o755)


def write_readme(output_dir: Path, variants: list[dict]) -> None:
    lines = [
        "# Wall Z0 Canary",
        "",
        "Variants keep the same mesh, inflow.json and lateral boundaryData.",
        "Only terrain/bottom wall roughness settings are changed.",
        "",
        "Variants:",
    ]
    for v in variants:
        lines.append(f"- `{v['case_name']}`: {v['description']}")
    lines.extend(
        [
            "",
            "Run/audit:",
            "",
            "```bash",
            f"bash {output_dir / 'export_and_audit_wall_z0.sh'}",
            "```",
            "",
            "Decision rule:",
            "",
            "- if lowering `z0_wall` restores bulk wind, roughness was too strong;",
            "- if `atmNutUWallFunction` restores wall log-law but `atmNutkWallFunction` does not, k-driven closure is suspect;",
            "- if neither helps, the issue is likely broader ABL/turbulence maintenance or pressure/BC formulation.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-case", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--z0-values", default="0.005,0.01,0.03,0.05")
    ap.add_argument("--wall-functions", default="atmNutkWallFunction,atmNutUWallFunction")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--copy-static", action="store_true",
                    help="Copy polyMesh/boundaryData/triSurface instead of symlinking them")
    ap.add_argument("--time", default=None, help="OpenFOAM time directory for export; default: controlDict endTime")
    args = ap.parse_args(list(argv) if argv is not None else None)

    base_case = args.base_case.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inflow = load_inflow(base_case)
    z0_values = [float(x) for x in args.z0_values.split(",") if x.strip()]
    wall_functions = [x.strip() for x in args.wall_functions.split(",") if x.strip()]

    variants = make_variants(
        base_case,
        output_dir,
        z0_values,
        wall_functions,
        overwrite=args.overwrite,
        symlink_static=not args.copy_static,
    )
    time_name = args.time or read_end_time(base_case)
    manifest = {
        "base_case": str(base_case),
        "output_dir": str(output_dir),
        "time": time_name,
        "z0_values": z0_values,
        "wall_functions": wall_functions,
        "inflow_z0_eff": finite_float(inflow.get("z0_eff", inflow.get("z0"))),
        "variants": variants,
    }
    (output_dir / "wall_z0_canary_manifest.json").write_text(json.dumps(manifest, indent=2))
    write_export_script(output_dir, variants, inflow, time_name)
    write_readme(output_dir, variants)

    print(f"canary={output_dir}")
    print(f"variants={len(variants)}")
    print(f"time={time_name}")
    print(f"export_audit={output_dir / 'export_and_audit_wall_z0.sh'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
