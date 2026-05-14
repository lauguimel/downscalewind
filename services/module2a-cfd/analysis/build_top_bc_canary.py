#!/usr/bin/env python3
"""Build top-boundary-condition canary variants from one OpenFOAM case."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path
from typing import Iterable

from build_wind_conservation_canary import (
    estimate_pg_from_era5_geopotential,
    estimate_pg_from_profile_wind,
    finite_float,
    load_inflow,
    parse_csv_floats,
    read_end_time,
    render_fvoptions,
)


MODULE_DIR = Path(__file__).resolve().parents[1]


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
        if item.name.startswith("processor") or item.name.startswith("log.") or is_time_dir(item):
            continue
        target = dst / item.name
        if item.is_dir():
            copy_dir(item, target, symlink_static=symlink_static)
        else:
            shutil.copy2(item, target)


def replace_patch_block(text: str, patch: str, new_body: str) -> str:
    match = re.search(rf"\b{re.escape(patch)}\s*\{{", text)
    if not match:
        raise KeyError(f"Patch {patch!r} block not found")
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
        raise ValueError(f"Unclosed patch block {patch!r}")
    body = "\n" + new_body.strip("\n") + "\n"
    return text[:body_start] + body + text[body_end:]


def patch_top_bcs(case_dir: Path) -> None:
    replacements = {
        "U": """        type            slip;""",
        "p": """        type            fixedValue;
        value           uniform 0;""",
        "p_rgh": """        type            fixedValue;
        value           uniform 0;""",
        "k": """        type            zeroGradient;""",
        "epsilon": """        type            zeroGradient;""",
    }
    for field, body in replacements.items():
        path = case_dir / "0" / field
        if not path.exists():
            continue
        text = path.read_text(errors="replace")
        path.write_text(replace_patch_block(text, "top", body))


def make_variants(
    base_case: Path,
    output_dir: Path,
    inflow: dict,
    pg: dict,
    *,
    overwrite: bool,
    symlink_static: bool,
) -> list[dict]:
    variants = [
        {
            "name": "control",
            "description": "unchanged top BCs",
            "top_bc": "current_inletOutlet",
            "pressure_gradient": False,
        },
        {
            "name": "slip_top",
            "description": "U top slip, p/p_rgh top fixedValue 0, k/epsilon top zeroGradient",
            "top_bc": "slip_fixed_p",
            "pressure_gradient": False,
        },
        {
            "name": "slip_top_pg_geo",
            "description": "slip top plus ERA5 geostrophic pressure-gradient source",
            "top_bc": "slip_fixed_p",
            "pressure_gradient": True,
        },
    ]
    out: list[dict] = []
    cases_root = output_dir / "cases"
    for variant in variants:
        case_name = f"case_ts000_{variant['name']}"
        dst = cases_root / case_name
        copy_base_case(base_case, dst, symlink_static=symlink_static, overwrite=overwrite)
        shutil.copy2(base_case / "inflow.json", dst / "inflow.json")
        if variant["top_bc"] == "slip_fixed_p":
            patch_top_bcs(dst)
        if variant["pressure_gradient"]:
            render_fvoptions(
                dst,
                inflow,
                {
                    "coriolis": True,
                    "pressure_gradient": {
                        "enabled": True,
                        "dp_dx": pg["dp_dx"],
                        "dp_dy": pg["dp_dy"],
                    },
                },
            )
        out.append({**variant, "case_name": case_name, "case_dir": str(dst)})
    return out


def write_export_script(output_dir: Path, variants: list[dict], inflow: dict, time_name: str) -> None:
    era5 = inflow.get("era5_source")
    timestamp = inflow.get("timestamp")
    lat = finite_float(inflow.get("site_lat"))
    lon = finite_float(inflow.get("site_lon"))
    script = output_dir / "export_and_audit_top_bc.sh"
    wind_csv = output_dir / "top_bc_wind_audit.csv"
    summary_csv = output_dir / "top_bc_wind_audit_summary.csv"
    top_flux_csv = output_dir / "top_bc_flux_proxy.csv"

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
    flux_py = MODULE_DIR / "analysis" / "audit_top_flux.py"
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
  --output "{wind_csv.resolve()}" \\
  --summary-output "{summary_csv.resolve()}" \\
  --heights 2,10,20,50,100 \\
  --crop-km 2

if [ "${{#SOLVED_CASES[@]}}" -gt 0 ]; then
  python3 "{flux_py.resolve()}" \\
    "${{SOLVED_CASES[@]}}" \\
    --time "{time_name}" \\
    --output "{top_flux_csv.resolve()}"
fi
"""
    )
    script.chmod(script.stat().st_mode | 0o755)


def write_readme(output_dir: Path, pg: dict, variants: list[dict]) -> None:
    lines = [
        "# Top BC Canary",
        "",
        "Variants:",
    ]
    for v in variants:
        lines.append(f"- `{v['case_name']}`: {v['description']}")
    lines.extend(
        [
            "",
            "Pressure-gradient estimate:",
            "",
            "```json",
            json.dumps(pg, indent=2),
            "```",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-case", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--copy-static", action="store_true")
    ap.add_argument("--geo-pressure-levels", default="850,800,700")
    ap.add_argument("--geo-height-band", default=None)
    ap.add_argument("--force-profile-pg", action="store_true")
    ap.add_argument("--profile-pg-height", type=float, default=1500.0)
    ap.add_argument("--time", default=None)
    args = ap.parse_args(list(argv) if argv is not None else None)

    base_case = args.base_case.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inflow = load_inflow(base_case)

    height_band = None
    if args.geo_height_band:
        vals = parse_csv_floats(args.geo_height_band)
        if len(vals) != 2:
            raise ValueError("--geo-height-band expects two values: lo,hi")
        height_band = (vals[0], vals[1])

    if args.force_profile_pg:
        pg = estimate_pg_from_profile_wind(inflow, args.profile_pg_height)
    else:
        try:
            pg = estimate_pg_from_era5_geopotential(
                inflow,
                requested_hpa=parse_csv_floats(args.geo_pressure_levels),
                height_band_m=height_band,
            )
        except Exception as exc:
            pg = estimate_pg_from_profile_wind(inflow, args.profile_pg_height)
            pg["fallback_reason"] = str(exc)

    variants = make_variants(
        base_case,
        output_dir,
        inflow,
        pg,
        overwrite=args.overwrite,
        symlink_static=not args.copy_static,
    )
    time_name = args.time or read_end_time(base_case)
    manifest = {
        "base_case": str(base_case),
        "output_dir": str(output_dir),
        "time": time_name,
        "pressure_gradient": pg,
        "variants": variants,
    }
    (output_dir / "top_bc_canary_manifest.json").write_text(json.dumps(manifest, indent=2))
    write_export_script(output_dir, variants, inflow, time_name)
    write_readme(output_dir, pg, variants)
    print(f"canary={output_dir}")
    print(f"variants={len(variants)}")
    print(f"time={time_name}")
    print(f"export_audit={output_dir / 'export_and_audit_top_bc.sh'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
