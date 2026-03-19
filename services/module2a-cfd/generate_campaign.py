"""
generate_campaign.py — Generate OpenFOAM cases for the multi-solver campaign

Reads a campaign sweep YAML and generates complete OpenFOAM case directories
for each parameter combination. Produces a campaign.yaml for kraken-sim.

Usage
-----
    # Generate all 240 simpleFoam cases
    python generate_campaign.py configs/training/campaign_simpleFoam.yaml \
        --group sf --prefix prd_sf

    # Generate pilot subset (2 directions × 1 speed × 3 stabilities)
    python generate_campaign.py configs/training/campaign_simpleFoam.yaml \
        --group pilot_sf --prefix pilot_sf \
        --filter-directions 231 40 --filter-speeds 10
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Add module2a-cfd to path for imports
MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

PROJECT_ROOT = Path(__file__).resolve().parents[2]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_campaign")


# ---------------------------------------------------------------------------
# Geohash encoding (no external dependency)
# ---------------------------------------------------------------------------
_GEOHASH_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def _geohash_encode(lat: float, lon: float, precision: int = 8) -> str:
    """Encode lat/lon to a geohash string."""
    lat_range, lon_range = [-90.0, 90.0], [-180.0, 180.0]
    bits = [16, 8, 4, 2, 1]
    ch, bit, is_lon = 0, 0, True
    result = []
    while len(result) < precision:
        if is_lon:
            mid = (lon_range[0] + lon_range[1]) / 2
            if lon >= mid:
                ch |= bits[bit]
                lon_range[0] = mid
            else:
                lon_range[1] = mid
        else:
            mid = (lat_range[0] + lat_range[1]) / 2
            if lat >= mid:
                ch |= bits[bit]
                lat_range[0] = mid
            else:
                lat_range[1] = mid
        is_lon = not is_lon
        if bit < 4:
            bit += 1
        else:
            result.append(_GEOHASH_BASE32[ch])
            ch, bit = 0, 0
    return "".join(result)


# ---------------------------------------------------------------------------
# Stability presets
# ---------------------------------------------------------------------------
STABILITY_PRESETS = {
    "neutral": {
        "Ri_b": 0.0,
        "T_grad_Km": 0.0,
        "L_mo": None,      # inf (neutral)
    },
    "stable": {
        "Ri_b": 0.15,
        "T_grad_Km": 0.5,  # K per 100m
        "L_mo": 200.0,     # positive = stable
    },
    "unstable": {
        "Ri_b": -0.10,
        "T_grad_Km": -0.3,
        "L_mo": -200.0,    # negative = unstable
    },
    "very_stable": {
        "Ri_b": 0.30,
        "T_grad_Km": 1.0,  # K per 100m (strong inversion)
        "L_mo": 80.0,      # short Obukhov length = very stable
    },
}


# ---------------------------------------------------------------------------
# Inflow profile builder (parametric, no ERA5 dependency)
# ---------------------------------------------------------------------------

def build_parametric_inflow(
    speed_ms: float,
    direction_deg: float,
    stability: str,
    z0: float = 0.05,
    T_ref: float = 288.15,
) -> dict:
    """Build a synthetic inflow profile for parametric campaign runs.

    Unlike prepare_inflow.py which reads ERA5, this creates a clean
    log-law profile from scratch using the specified parameters.

    Parameters
    ----------
    speed_ms : float
        Reference wind speed at hub height (80 m) [m/s].
    direction_deg : float
        Wind direction FROM (meteorological convention, 0=N, 90=E) [degrees].
    stability : str
        One of 'neutral', 'stable', 'unstable'.
    z0 : float
        Surface roughness length [m].
    T_ref : float
        Reference temperature [K].

    Returns
    -------
    dict
        Inflow profile compatible with Jinja2 templates.
    """
    import numpy as np

    kappa = 0.41
    z_ref = 80.0
    stab = STABILITY_PRESETS[stability]

    # Friction velocity from log law: u(z_ref) = (u*/kappa) * ln(z_ref/z0)
    u_star = speed_ms * kappa / np.log(z_ref / z0)

    # Flow direction vector (wind blows TOWARD)
    wind_toward_deg = (direction_deg + 180.0) % 360.0
    wind_toward_rad = math.radians(wind_toward_deg)
    flow_dir_x = math.sin(wind_toward_rad)
    flow_dir_y = math.cos(wind_toward_rad)

    # Vertical profile (log law + stability correction)
    z_levels = np.concatenate([
        np.arange(2, 100, 5),       # 2-95 m, every 5 m
        np.arange(100, 500, 25),    # 100-475 m, every 25 m
        np.arange(500, 1000, 50),   # 500-950 m, every 50 m
        np.arange(1000, 3001, 100), # 1000-3000 m, every 100 m
    ])

    # Speed profile: log law with Monin-Obukhov correction
    L_mo = stab["L_mo"]
    u_profile = np.zeros_like(z_levels, dtype=float)
    for i, z in enumerate(z_levels):
        if z <= z0:
            u_profile[i] = 0.0
        else:
            log_term = np.log(z / z0) / kappa
            # Stability correction (Businger-Dyer)
            if L_mo is not None and L_mo != 0:
                zeta = z / L_mo
                if zeta > 0:  # stable
                    psi = -5.0 * min(zeta, 1.0)
                else:  # unstable
                    x = (1 - 16 * zeta) ** 0.25
                    psi = 2 * np.log((1 + x) / 2) + np.log((1 + x**2) / 2) - 2 * np.arctan(x) + np.pi / 2
            else:
                psi = 0.0
            u_profile[i] = u_star * (log_term - psi)

    # Temperature profile
    T_grad = stab["T_grad_Km"] / 100.0  # K/m
    # Dry adiabatic lapse rate
    gamma_d = 0.0098  # K/m
    T_profile = T_ref - gamma_d * z_levels + T_grad * z_levels

    # Pressure profile (ISA hydrostatic)
    P0 = 101325.0  # Pa
    RD = 287.05     # J/(kg·K)
    g  = 9.81       # m/s²
    T_mean = np.maximum(T_profile, 200.0)  # avoid division issues
    p_profile = P0 * np.exp(-g * z_levels / (RD * T_mean))

    return {
        "u_hub": float(speed_ms),
        "u_star": float(u_star),
        "z0_eff": float(z0),
        "z0": float(z0),
        "z_ref": float(z_ref),
        "kappa": float(kappa),
        "d": 0.0,
        "L_mo": stab["L_mo"],
        "T_ref": float(T_ref),
        "Ri_b": stab["Ri_b"],
        "wind_dir": float(direction_deg),
        "flowDir_x": float(flow_dir_x),
        "flowDir_y": float(flow_dir_y),
        "z_levels": z_levels.tolist(),
        "u_profile": u_profile.tolist(),
        "T_profile": T_profile.tolist(),
        "p_profile": p_profile.tolist(),
    }


# ---------------------------------------------------------------------------
# Campaign generator
# ---------------------------------------------------------------------------

@dataclass
class CampaignCase:
    """A single case in the campaign."""
    case_id: str
    solver: str
    direction_deg: float
    speed_ms: float
    stability: str
    thermal: bool
    coriolis: bool
    canopy: bool
    resolution_m: float
    domain_km: float
    context_cells: int
    boussinesq: bool = False
    fvschemes_variant: str = "robust"  # "robust" (upwind k/eps) or "accurate" (linearUpwind k/eps)
    n_iterations: int = 2000
    write_interval: int = 200
    ncpus: int = 12
    walltime: str = "04:00:00"
    mem: str = "32gb"


def load_campaign_config(yaml_path: Path) -> dict:
    """Load and parse a campaign sweep YAML."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def expand_cases(
    cfg: dict,
    prefix: str = "prd",
    filter_directions: list[float] | None = None,
    filter_speeds: list[float] | None = None,
    filter_stabilities: list[str] | None = None,
) -> list[CampaignCase]:
    """Expand parameter grid into a list of CampaignCase objects.

    Supports two YAML formats:
      - New (cfd_grid.yaml): top-level directions_deg, speeds_ms, stabilities, solver
      - Legacy: parameters.direction_deg, parameters.speed_ms, etc.
    """
    # Support both YAML layouts
    if "parameters" in cfg:
        params = cfg["parameters"]
        directions = params["direction_deg"]
        speeds = params["speed_ms"]
        stabilities = params["stability"]
        solvers = params["solver"]
    else:
        directions = cfg["directions_deg"]
        speeds = cfg["speeds_ms"]
        # stabilities can be a list of dicts [{id: "neutral", ...}] or strings
        raw_stab = cfg["stabilities"]
        stabilities = [s["id"] if isinstance(s, dict) else s for s in raw_stab]
        # solver from solver.name or default
        solver_cfg = cfg.get("solver", {})
        solver_name = solver_cfg.get("name", "simpleFoam") if isinstance(solver_cfg, dict) else solver_cfg
        solvers = [solver_name]

    physics = cfg.get("physics", {})
    mesh = cfg.get("mesh", {})

    # Apply filters
    if filter_directions:
        directions = [d for d in directions if d in filter_directions]
    if filter_speeds:
        speeds = [s for s in speeds if s in filter_speeds]
    if filter_stabilities:
        stabilities = [s for s in stabilities if s in filter_stabilities]

    cases = []
    for i, (solver, direction, speed, stability) in enumerate(
        itertools.product(solvers, directions, speeds, stabilities)
    ):
        case_id = f"{prefix}_{i:04d}"
        # ESI v2412: buoyantBoussinesqSimpleFoam is a separate solver
        actual_solver = solver
        boussinesq = physics.get("boussinesq", False)
        if solver == "buoyantBoussinesqSimpleFoam":
            boussinesq = True
        cases.append(CampaignCase(
            case_id=case_id,
            solver=actual_solver,
            direction_deg=direction,
            speed_ms=speed,
            stability=stability,
            thermal=physics.get("thermal", False),
            coriolis=physics.get("coriolis", True),
            canopy=physics.get("canopy", False),
            resolution_m=mesh.get("resolution_m", 100),
            domain_km=mesh.get("domain_km", 10),
            context_cells=mesh.get("context_cells", 1),
            boussinesq=boussinesq,
        ))

    return cases


def generate_case_dir(
    case: CampaignCase,
    output_base: Path,
    site_cfg: dict,
    srtm_tif: Path | None = None,
    mesh_cache: dict[str, Path] | None = None,
) -> Path:
    """Generate a single OpenFOAM case directory.

    Parameters
    ----------
    case : CampaignCase
        Case specification.
    output_base : Path
        Base directory for all campaign cases.
    site_cfg : dict
        Parsed perdigao.yaml content.
    srtm_tif : Path | None
        SRTM DEM for terrain. Falls back to flat terrain.
    mesh_cache : dict
        Cache of (resolution_m) -> path to a case with generated mesh.
        Used to copy polyMesh instead of regenerating.

    Returns
    -------
    Path
        The generated case directory.
    """
    from generate_mesh import generate_mesh

    case_dir = output_base / case.case_id
    if case_dir.exists():
        logger.info("Case %s already exists, skipping", case.case_id)
        return case_dir

    logger.info(
        "Generating %s: solver=%s, dir=%.1f°, speed=%.1f m/s, stab=%s",
        case.case_id, case.solver, case.direction_deg,
        case.speed_ms, case.stability,
    )

    # Build parametric inflow profile
    inflow_data = build_parametric_inflow(
        speed_ms=case.speed_ms,
        direction_deg=case.direction_deg,
        stability=case.stability,
    )

    # Write inflow JSON (temporary, consumed by generate_mesh)
    inflow_dir = output_base / "inflow_profiles"
    inflow_dir.mkdir(exist_ok=True)
    inflow_json = inflow_dir / f"inflow_{case.case_id}.json"
    with open(inflow_json, "w") as f:
        json.dump(inflow_data, f, indent=2)

    # Check mesh cache: reuse polyMesh if same resolution exists
    mesh_key = f"{case.resolution_m}m"
    reuse_mesh = mesh_cache and mesh_key in mesh_cache

    # Generate full case (mesh + templates)
    generate_mesh(
        site_cfg=site_cfg,
        resolution_m=case.resolution_m,
        context_cells=case.context_cells,
        output_dir=case_dir,
        srtm_tif=srtm_tif,
        inflow_json=inflow_json,
        domain_km=case.domain_km,
        solver_name=case.solver,
        thermal=case.thermal,
        coriolis=case.coriolis,
        canopy_enabled=case.canopy,
        boussinesq=case.boussinesq,
    )

    # Copy inflow JSON + init_from_era5.py into case dir (for Allrun on HPC)
    dst_inflow = case_dir / "inflow.json"
    shutil.copy2(inflow_json, dst_inflow)
    init_script = Path(__file__).parent / "init_from_era5.py"
    if init_script.exists():
        shutil.copy2(init_script, case_dir / "init_from_era5.py")

    # Patch fvSchemes if "accurate" variant requested
    if case.fvschemes_variant == "accurate":
        _patch_fvschemes_accurate(case_dir)

    # Override controlDict with campaign-specific settings
    # (generate_mesh defaults to 1000 iterations)
    _patch_control_dict(case_dir, case.n_iterations, case.write_interval)

    # If we have a mesh cache and this is NOT the first case at this resolution,
    # we could skip blockMesh/snappy by copying polyMesh. But for simplicity
    # and correctness (mesh depends only on resolution, not solver), we let
    # Allrun handle it. The mesh is identical across solvers at same resolution.

    # Register this case's mesh in cache for future reference
    if mesh_cache is not None and mesh_key not in mesh_cache:
        mesh_cache[mesh_key] = case_dir

    return case_dir


def _patch_control_dict(
    case_dir: Path,
    n_iter: int,
    write_interval: int,
    purge_write: int | None = None,
) -> None:
    """Patch controlDict endTime, writeInterval, and optionally purgeWrite."""
    cd_path = case_dir / "system" / "controlDict"
    if not cd_path.exists():
        return
    import re
    text = cd_path.read_text()
    text = re.sub(r"endTime\s+\d+;", f"endTime         {n_iter};", text)
    text = re.sub(r"writeInterval\s+\d+;", f"writeInterval   {write_interval};", text)
    if purge_write is not None:
        text = re.sub(r"purgeWrite\s+\d+;", f"purgeWrite      {purge_write};", text)
    cd_path.write_text(text)


def _patch_fvschemes_accurate(case_dir: Path) -> None:
    """Patch fvSchemes: switch k/epsilon from upwind to linearUpwind.

    Default (robust): bounded Gauss upwind for k and epsilon.
    Accurate variant: bounded Gauss linearUpwind for k and epsilon.
    More accurate but can be less stable on coarse/skewed meshes.
    """
    import re
    fvs_path = case_dir / "system" / "fvSchemes"
    if not fvs_path.exists():
        return
    text = fvs_path.read_text()
    text = re.sub(
        r"div\(phi,k\)\s+bounded Gauss upwind;",
        "div(phi,k)                      bounded Gauss linearUpwind grad(k);",
        text,
    )
    text = re.sub(
        r"div\(phi,epsilon\)\s+bounded Gauss upwind;",
        "div(phi,epsilon)                bounded Gauss linearUpwind grad(epsilon);",
        text,
    )
    fvs_path.write_text(text)
    logger.info("fvSchemes patched to 'accurate' (linearUpwind k/epsilon)")


def _render_run_pbs(case_dir: Path, case: CampaignCase, remote_case_dir: str) -> None:
    """Render run.pbs.j2 into the case directory."""
    from jinja2 import Environment, FileSystemLoader

    template_dir = MODULE_DIR / "templates" / "openfoam"
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template("run.pbs.j2")
    content = template.render(
        case_id=case.case_id,
        ncpus=case.ncpus,
        mem=case.mem,
        walltime=case.walltime,
        remote_case_dir=remote_case_dir,
    )
    (case_dir / "run.pbs").write_text(content)


def generate_campaign(
    config_path: Path,
    output_base: Path,
    site_config_path: Path,
    group_name: str | None = None,
    prefix: str = "prd",
    srtm_tif: Path | None = None,
    filter_directions: list[float] | None = None,
    filter_speeds: list[float] | None = None,
    filter_stabilities: list[str] | None = None,
    dry_run: bool = False,
) -> list[CampaignCase]:
    """Main entry point: generate all cases for a campaign config.

    Parameters
    ----------
    config_path : Path
        Campaign sweep YAML file.
    output_base : Path
        Directory where OpenFOAM case dirs are created.
    site_config_path : Path
        Path to sites/perdigao.yaml.
    group_name : str | None
        Group name for campaign.yaml.
    prefix : str
        Case ID prefix (e.g., 'prd_sf' for simpleFoam).
    srtm_tif : Path | None
        SRTM DEM GeoTIFF.
    filter_directions, filter_speeds, filter_stabilities : list | None
        Optional filters to generate a subset.
    dry_run : bool
        If True, only print the case list without generating.

    Returns
    -------
    list[CampaignCase]
        List of generated cases.
    """
    cfg = load_campaign_config(config_path)
    cases = expand_cases(
        cfg, prefix=prefix,
        filter_directions=filter_directions,
        filter_speeds=filter_speeds,
        filter_stabilities=filter_stabilities,
    )

    logger.info(
        "Campaign: %d cases (%s), output=%s",
        len(cases), config_path.name, output_base,
    )

    if dry_run:
        for c in cases:
            print(f"  {c.case_id}: {c.solver} dir={c.direction_deg}° "
                  f"speed={c.speed_ms}m/s stab={c.stability}")
        return cases

    # Load site config
    with open(site_config_path) as f:
        site_cfg = yaml.safe_load(f)

    output_base.mkdir(parents=True, exist_ok=True)

    # Build remote dir from site geohash (8 chars ≈ ±20 m precision)
    site_lat = site_cfg["site"]["coordinates"]["latitude"]
    site_lon = site_cfg["site"]["coordinates"]["longitude"]
    ghash = _geohash_encode(site_lat, site_lon, precision=8)
    remote_dir = f"/home/maitreje/campaigns/{ghash}/{prefix}"

    # Generate cases
    mesh_cache: dict[str, Path] = {}
    campaign_cases = []
    for case in cases:
        case_dir = generate_case_dir(
            case=case,
            output_base=output_base,
            site_cfg=site_cfg,
            srtm_tif=srtm_tif,
            mesh_cache=mesh_cache,
        )
        # Render run.pbs from template
        remote_case_dir = f"{remote_dir}/{case.case_id}"
        _render_run_pbs(case_dir, case, remote_case_dir)

    # Write kraken-sim campaign.yaml via CampaignBuilder
    from kraken_sim.schema import CampaignBuilder

    builder = CampaignBuilder(
        name=f"perdigao_{prefix}",
        hpc_host="aqua.qut.edu.au",
        hpc_username="maitreje",
        hpc_remote_dir=remote_dir,
        local_base_dir=str(output_base),
        monitoring_fields=["U", "p", "k", "epsilon"],
        auto_group=["solver", "stability"],
    )

    for case in cases:
        builder.add_case(
            case_id=case.case_id,
            case_dir=case.case_id,
            ncpus=case.ncpus,
            tags={
                "solver": case.solver,
                "direction_deg": case.direction_deg,
                "speed_ms": case.speed_ms,
                "stability": case.stability,
                "resolution_m": case.resolution_m,
                "domain_km": case.domain_km,
                "coriolis": case.coriolis,
                "canopy": case.canopy,
                "thermal": case.thermal,
                "fvschemes": case.fvschemes_variant,
            },
        )

    campaign_path = output_base / "campaign.yaml"
    builder.write_yaml(campaign_path)
    logger.info("Campaign YAML saved: %s (%d cases)", campaign_path, len(cases))

    return cases


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate OpenFOAM campaign cases from sweep YAML"
    )
    parser.add_argument(
        "config", type=Path,
        help="Campaign sweep YAML (e.g., campaign_simpleFoam.yaml)"
    )
    parser.add_argument(
        "--output", "-o", type=Path,
        default=PROJECT_ROOT / "data" / "campaign" / "cases",
        help="Output directory for case dirs (default: data/campaign/cases)",
    )
    parser.add_argument(
        "--site", type=Path,
        default=PROJECT_ROOT / "configs" / "sites" / "perdigao.yaml",
        help="Site config YAML",
    )
    parser.add_argument("--group", type=str, default=None, help="Group name for campaign.yaml")
    parser.add_argument("--prefix", type=str, default="prd", help="Case ID prefix")
    parser.add_argument("--srtm", type=Path, default=None, help="SRTM GeoTIFF path")
    parser.add_argument(
        "--filter-directions", type=float, nargs="+", default=None,
        help="Only generate these directions (e.g., 231 40)",
    )
    parser.add_argument(
        "--filter-speeds", type=float, nargs="+", default=None,
        help="Only generate these speeds (e.g., 10)",
    )
    parser.add_argument(
        "--filter-stabilities", type=str, nargs="+", default=None,
        help="Only generate these stabilities (e.g., neutral stable)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print cases without generating")

    args = parser.parse_args()

    # Auto-detect SRTM
    srtm = args.srtm
    if srtm is None:
        default_srtm = PROJECT_ROOT / "data" / "raw" / "srtm_perdigao_30m.tif"
        if default_srtm.exists():
            srtm = default_srtm

    generate_campaign(
        config_path=args.config,
        output_base=args.output,
        site_config_path=args.site,
        group_name=args.group,
        prefix=args.prefix,
        srtm_tif=srtm,
        filter_directions=args.filter_directions,
        filter_speeds=args.filter_speeds,
        filter_stabilities=args.filter_stabilities,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
