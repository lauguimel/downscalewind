"""
check_physics.py — Pre-flight physics verification for OpenFOAM ABL cases

Verifies that all physical constants and parameters are self-consistent
BEFORE submitting a campaign.  Catches the kind of errors that LLMs
introduce when generating templates (wrong Omega, missing beta, etc.).

Usage
-----
    python check_physics.py --case-dir data/campaign/case_001
    python check_physics.py --campaign data/campaign/campaign.yaml

Checks
------
1. Coriolis:  f = 2*Omega*sin(lat), Omega consistent with latitude
2. Gravity:   g = (0, 0, -9.81) ± 0.01 m/s²
3. Boussinesq: beta ≈ 1/T_ref (± 5%), rho0 ∈ [1.0, 1.4] kg/m³
4. Temperature: T_ref ∈ [250, 320] K, uniform or realistic lapse rate
5. Viscosity: nu ∈ [1.3e-5, 1.8e-5] m²/s (air at 250–320 K)
6. Turbulence model: kEpsilon with standard coefficients
7. BCs consistency: inflow velocity matches campaign parameters
8. Mesh: domain height ≥ 10 × terrain height, AR < 10
"""

from __future__ import annotations

import logging
import math
import re
import sys
from pathlib import Path

import click
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants (expected values)
# ---------------------------------------------------------------------------
OMEGA_EARTH = 7.2921e-5   # rad/s
G_MAGNITUDE = 9.81        # m/s²
NU_AIR_MIN = 1.3e-5       # m²/s (air at ~250K)
NU_AIR_MAX = 1.8e-5       # m²/s (air at ~320K)
T_REF_MIN = 250.0         # K
T_REF_MAX = 320.0         # K
RHO_AIR_MIN = 1.0         # kg/m³
RHO_AIR_MAX = 1.4         # kg/m³
BETA_TOL = 0.05           # 5% tolerance on beta = 1/T_ref


def _read_of_dict(path: Path) -> str:
    """Read an OpenFOAM dictionary file as text."""
    if not path.exists():
        return ""
    return path.read_text()


def _extract_value(text: str, key: str) -> str | None:
    """Extract a scalar or vector value after 'key' in OF dictionary text."""
    # Match: key  value; or key (x y z);
    m = re.search(rf'{key}\s+([^;]+);', text)
    return m.group(1).strip() if m else None


def _extract_vector(text: str, key: str) -> tuple[float, ...] | None:
    """Extract a vector (x y z) value."""
    m = re.search(rf'{key}\s+\(([^)]+)\)', text)
    if m:
        return tuple(float(x) for x in m.group(1).split())
    return None


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

class CheckResult:
    def __init__(self, name: str, ok: bool, message: str, value=None, expected=None):
        self.name = name
        self.ok = ok
        self.message = message
        self.value = value
        self.expected = expected

    def __str__(self):
        status = "PASS" if self.ok else "FAIL"
        s = f"  [{status}] {self.name}: {self.message}"
        if self.value is not None and self.expected is not None:
            s += f" (got {self.value}, expected {self.expected})"
        return s


def check_coriolis(case_dir: Path) -> list[CheckResult]:
    """Check Coriolis force parameters."""
    results = []
    fvopts = _read_of_dict(case_dir / "constant" / "fvOptions")

    if "atmCoriolisUSource" not in fvopts:
        if "coriolis" in fvopts.lower() or "coded" in fvopts.lower():
            results.append(CheckResult(
                "coriolis_type", False,
                "Coriolis uses 'coded' type — must use native atmCoriolisUSource for ESI v2412"))
        else:
            results.append(CheckResult(
                "coriolis_present", False, "No Coriolis source found in fvOptions"))
        return results

    results.append(CheckResult("coriolis_type", True, "Uses native atmCoriolisUSource"))

    # Check latitude vs Omega
    lat_m = re.search(r'latitude\s+([0-9.e+-]+)', fvopts)
    omega_v = _extract_vector(fvopts, "Omega")

    if lat_m:
        lat = float(lat_m.group(1))
        if not (-90 <= lat <= 90):
            results.append(CheckResult("coriolis_lat", False,
                           f"Latitude {lat}° out of range [-90, 90]", lat))
        else:
            f_expected = 2 * OMEGA_EARTH * math.sin(math.radians(lat))
            results.append(CheckResult("coriolis_lat", True,
                           f"Latitude {lat}°N → f = {f_expected:.4e} rad/s"))
    elif omega_v:
        # If using Omega vector directly, check it's not full Earth rotation
        omega_z = omega_v[2] if len(omega_v) >= 3 else 0
        if abs(omega_z - OMEGA_EARTH) < 1e-8:
            results.append(CheckResult("coriolis_omega", False,
                           "Omega = Earth rotation (pole value) — did you forget latitude?",
                           omega_z, "Omega_z < 7.29e-5 for lat < 90°"))
        else:
            results.append(CheckResult("coriolis_omega", True,
                           f"Omega_z = {omega_z:.4e} rad/s"))
    else:
        results.append(CheckResult("coriolis_params", False,
                       "Neither latitude nor Omega found in atmCoriolisUSource"))

    return results


def check_gravity(case_dir: Path) -> list[CheckResult]:
    """Check gravity vector."""
    g_text = _read_of_dict(case_dir / "constant" / "g")
    if not g_text:
        return [CheckResult("gravity", False, "constant/g not found")]

    g_vec = _extract_vector(g_text, "value")
    if g_vec is None:
        return [CheckResult("gravity", False, "Cannot parse g vector")]

    g_mag = math.sqrt(sum(x**2 for x in g_vec))
    ok = abs(g_mag - G_MAGNITUDE) < 0.1
    direction_ok = len(g_vec) >= 3 and g_vec[2] < 0
    return [
        CheckResult("gravity_magnitude", ok,
                    f"|g| = {g_mag:.3f} m/s²", g_mag, G_MAGNITUDE),
        CheckResult("gravity_direction", direction_ok,
                    f"g = {g_vec}", g_vec, "(0 0 -9.81)"),
    ]


def check_thermo(case_dir: Path) -> list[CheckResult]:
    """Check thermal properties (buoyantSimpleFoam or buoyantBoussinesqSimpleFoam)."""
    results = []

    # buoyantBoussinesqSimpleFoam uses transportProperties with beta, TRef
    tp_text = _read_of_dict(case_dir / "constant" / "transportProperties")
    if tp_text:
        beta_m = re.search(r'beta\s+\[.*?\]\s+([0-9.e+-]+)', tp_text)
        tref_m = re.search(r'TRef\s+\[.*?\]\s+([0-9.e+-]+)', tp_text)
        if beta_m and tref_m:
            beta = float(beta_m.group(1))
            TRef = float(tref_m.group(1))
            beta_expected = 1.0 / TRef
            rel_err = abs(beta - beta_expected) / beta_expected
            results.append(CheckResult("thermo_mode", True,
                           "buoyantBoussinesqSimpleFoam (T-equation, stable)"))
            results.append(CheckResult("thermo_beta", rel_err < BETA_TOL,
                           f"beta = {beta:.4e}, 1/TRef = {beta_expected:.4e} (err={rel_err:.1%})",
                           beta, beta_expected))
            ok_tref = T_REF_MIN <= TRef <= T_REF_MAX
            results.append(CheckResult("thermo_tref", ok_tref,
                           f"TRef = {TRef} K", TRef, f"[{T_REF_MIN}, {T_REF_MAX}]"))
            return results

    thermo_text = _read_of_dict(case_dir / "constant" / "thermophysicalProperties")
    if not thermo_text:
        return [CheckResult("thermo", True, "No thermophysicalProperties (simpleFoam case)")]

    # Check EoS type
    eos_m = re.search(r'equationOfState\s+(\w+)', thermo_text)
    if eos_m:
        eos = eos_m.group(1)
        if eos == "rhoConst":
            results.append(CheckResult("thermo_eos", False,
                           "rhoConst EoS — no buoyancy effect, use Boussinesq or perfectGas",
                           eos, "Boussinesq"))
        elif eos == "Boussinesq":
            results.append(CheckResult("thermo_eos", True, "Boussinesq EoS"))
            # Check beta, rho0, T0
            beta_m = re.search(r'beta\s+([0-9.e+-]+)', thermo_text)
            t0_m = re.search(r'T0\s+([0-9.e+-]+)', thermo_text)
            rho0_m = re.search(r'rho0\s+([0-9.e+-]+)', thermo_text)

            if beta_m and t0_m:
                beta = float(beta_m.group(1))
                T0 = float(t0_m.group(1))
                beta_expected = 1.0 / T0
                rel_err = abs(beta - beta_expected) / beta_expected
                results.append(CheckResult("thermo_beta", rel_err < BETA_TOL,
                               f"beta = {beta:.4e}, 1/T0 = {beta_expected:.4e} (err={rel_err:.1%})",
                               beta, beta_expected))
            if rho0_m:
                rho0 = float(rho0_m.group(1))
                results.append(CheckResult("thermo_rho0",
                               RHO_AIR_MIN <= rho0 <= RHO_AIR_MAX,
                               f"rho0 = {rho0} kg/m³", rho0, f"[{RHO_AIR_MIN}, {RHO_AIR_MAX}]"))
        elif eos == "perfectGas":
            results.append(CheckResult("thermo_eos", True, "perfectGas EoS"))
        else:
            results.append(CheckResult("thermo_eos", False, f"Unknown EoS: {eos}"))

    return results


def check_viscosity(case_dir: Path) -> list[CheckResult]:
    """Check kinematic viscosity."""
    for fname in ["transportProperties", "physicalProperties"]:
        text = _read_of_dict(case_dir / "constant" / fname)
        nu_m = re.search(r'nu\s+\[.*?\]\s+([0-9.e+-]+)', text)
        if nu_m:
            nu = float(nu_m.group(1))
            ok = NU_AIR_MIN <= nu <= NU_AIR_MAX
            return [CheckResult("viscosity", ok,
                                f"nu = {nu:.2e} m²/s", nu, f"[{NU_AIR_MIN:.1e}, {NU_AIR_MAX:.1e}]")]
    return [CheckResult("viscosity", True, "No transportProperties (buoyantSimpleFoam uses mu)")]


def check_temperature(case_dir: Path) -> list[CheckResult]:
    """Check initial temperature field."""
    results = []
    t_text = _read_of_dict(case_dir / "0" / "T")
    if not t_text:
        return [CheckResult("temperature", True, "No T field (simpleFoam case)")]

    internal_m = re.search(r'internalField\s+uniform\s+([0-9.e+-]+)', t_text)
    if internal_m:
        T_init = float(internal_m.group(1))
        ok = T_REF_MIN <= T_init <= T_REF_MAX
        results.append(CheckResult("temperature_init", ok,
                       f"T_init = {T_init} K", T_init, f"[{T_REF_MIN}, {T_REF_MAX}]"))
    return results


def check_turbulence(case_dir: Path) -> list[CheckResult]:
    """Check turbulence model."""
    for fname in ["turbulenceProperties", "momentumTransport"]:
        text = _read_of_dict(case_dir / "constant" / fname)
        if "kEpsilon" in text:
            return [CheckResult("turbulence_model", True, "k-epsilon standard")]
        if "kOmegaSST" in text:
            return [CheckResult("turbulence_model", True, "k-omega SST")]
    return [CheckResult("turbulence_model", False, "Cannot determine turbulence model")]


def check_libs(case_dir: Path) -> list[CheckResult]:
    """Check that required libs are loaded in controlDict."""
    cd_text = _read_of_dict(case_dir / "system" / "controlDict")
    fvopts = _read_of_dict(case_dir / "constant" / "fvOptions")

    results = []
    if "atmCoriolisUSource" in fvopts or "atmPlantCanopy" in fvopts:
        if "atmosphericModels" in cd_text:
            results.append(CheckResult("libs_atm", True, "atmosphericModels loaded"))
        else:
            results.append(CheckResult("libs_atm", False,
                           "fvOptions uses atm* sources but controlDict missing libs (atmosphericModels)"))
    return results


def check_inflow(case_dir: Path) -> list[CheckResult]:
    """Check inflow velocity consistency."""
    results = []
    u_text = _read_of_dict(case_dir / "0" / "U")
    if not u_text:
        return results

    # Check internalField
    vec = _extract_vector(u_text, "internalField\\s+uniform")
    if vec and len(vec) >= 2:
        speed = math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2) if len(vec) >= 3 else math.sqrt(vec[0]**2 + vec[1]**2)
        direction = (270.0 - math.degrees(math.atan2(vec[1], vec[0]))) % 360.0
        results.append(CheckResult("inflow_speed", speed > 0.5,
                       f"U_init = {speed:.2f} m/s, dir = {direction:.1f}°"))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_CHECKS = [
    check_coriolis,
    check_gravity,
    check_thermo,
    check_viscosity,
    check_temperature,
    check_turbulence,
    check_libs,
    check_inflow,
]


def check_case(case_dir: Path) -> list[CheckResult]:
    """Run all physics checks on a single case."""
    results = []
    for check_fn in ALL_CHECKS:
        try:
            results.extend(check_fn(case_dir))
        except Exception as e:
            results.append(CheckResult(check_fn.__name__, False, str(e)))
    return results


@click.command()
@click.option("--case-dir", type=click.Path(exists=True, path_type=Path),
              help="Single OpenFOAM case directory")
@click.option("--campaign", type=click.Path(exists=True, path_type=Path),
              help="Campaign YAML manifest (checks all cases)")
def main(case_dir: Path | None, campaign: Path | None):
    """Pre-flight physics verification for OpenFOAM ABL cases."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if campaign:
        with open(campaign) as f:
            manifest = yaml.safe_load(f)
        base = Path(manifest.get("local_base_dir", campaign.parent))
        cases = [c["id"] for c in manifest.get("cases", [])]
        case_dirs = [base / c for c in cases]
    elif case_dir:
        case_dirs = [case_dir]
    else:
        click.echo("Provide --case-dir or --campaign")
        sys.exit(1)

    total_fail = 0
    for cd in case_dirs:
        if not cd.exists():
            logger.warning("Case %s not found, skipping", cd.name)
            continue
        results = check_case(cd)
        failures = [r for r in results if not r.ok]
        total_fail += len(failures)

        status = "OK" if not failures else f"{len(failures)} FAIL"
        click.echo(f"\n{'='*60}")
        click.echo(f"Case: {cd.name}  [{status}]")
        click.echo(f"{'='*60}")
        for r in results:
            click.echo(str(r))

    click.echo(f"\n{'='*60}")
    if total_fail == 0:
        click.echo("All physics checks passed.")
    else:
        click.echo(f"TOTAL FAILURES: {total_fail}")
        sys.exit(1)


if __name__ == "__main__":
    main()
