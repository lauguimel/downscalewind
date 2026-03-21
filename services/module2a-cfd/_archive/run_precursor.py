"""
run_precursor.py — 1-D precursor ABL simulation (flat, cyclic)

Produces equilibrium wind profiles with realistic wind veer from Coriolis
and pressure gradient forcing.  The converged profiles replace the idealized
log-law from prepare_inflow.py for production downscaling runs.

Domain: flat box with cyclic BCs in x and y, slip top, rough wall bottom.
Forcing: geostrophic wind via pressure gradient + Coriolis.
  dp/dx = -rho * f * V_g    dp/dy = +rho * f * U_g
where (U_g, V_g) is the geostrophic wind and f the Coriolis parameter.

Pipeline:
  1. Build a flat blockMesh case (no snappy, no terrain STL)
  2. Initialise with log-law profile
  3. Run simpleFoam (or buoyantBoussinesqSimpleFoam)
  4. Extract converged vertical profiles of U, k, epsilon, T

Usage
-----
    python run_precursor.py \\
        --u-geostrophic 10.0 \\
        --wind-dir 231.0 \\
        --z0 0.05 \\
        --latitude 39.716 \\
        --n-iter 10000 \\
        --output data/precursor/perdigao_231deg/

References
----------
    Venkatraman et al. (WES 2023) — Appendix B: precursor simulation setup
    Letzgus et al. (WES 2023) — concurrent precursor for Perdigão ABL
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Physical constants
KAPPA = 0.41
CMU = 0.09
OMEGA = 7.2921e-5  # Earth angular velocity [rad/s]

# Default precursor domain
DOMAIN_X = 1000.0   # m (streamwise)
DOMAIN_Y = 1000.0   # m (crosswise)
DOMAIN_Z = 3000.0   # m (vertical)
NX = 5              # Cyclic, only need a few cells
NY = 5
NZ = 80             # Fine vertical resolution
Z_GRADING = 8.0     # Vertical grading (finer near ground)


def compute_geostrophic_components(u_geo: float, wind_dir: float) -> tuple[float, float]:
    """Convert geostrophic speed + direction to (U_g, V_g) in CFD coords.

    wind_dir is meteorological convention: degrees FROM which wind blows,
    measured clockwise from North.
    """
    dir_rad = np.radians(wind_dir)
    # CFD: x=East, y=North.  Wind FROM dir → flow TOWARDS opposite direction
    u_g = -u_geo * np.sin(dir_rad)  # East component
    v_g = -u_geo * np.cos(dir_rad)  # North component
    return float(u_g), float(v_g)


def coriolis_parameter(latitude: float) -> float:
    """Compute Coriolis parameter f = 2*Omega*sin(lat)."""
    return 2.0 * OMEGA * np.sin(np.radians(latitude))


def generate_precursor_case(
    output_dir: Path,
    u_geostrophic: float,
    wind_dir: float,
    z0: float,
    latitude: float,
    n_iter: int = 10000,
    T_ref: float = 300.0,
    solver_name: str = "simpleFoam",
    of_version: int = 10,
) -> Path:
    """Generate and set up a precursor simulation case.

    Parameters
    ----------
    output_dir : Case output directory.
    u_geostrophic : Geostrophic wind speed [m/s].
    wind_dir : Meteorological wind direction [degrees from N].
    z0 : Surface roughness length [m].
    latitude : Site latitude [degrees] for Coriolis.
    n_iter : Number of SIMPLE iterations.
    T_ref : Reference temperature [K].
    solver_name : OpenFOAM solver.
    of_version : OpenFOAM version (9 or 10).

    Returns
    -------
    Path to the generated case directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    u_g, v_g = compute_geostrophic_components(u_geostrophic, wind_dir)
    f = coriolis_parameter(latitude)

    logger.info(
        "Precursor: U_geo=%.1f m/s, dir=%.0f°, (U_g, V_g)=(%.2f, %.2f), "
        "f=%.2e, z0=%.3f m, %d iters",
        u_geostrophic, wind_dir, u_g, v_g, f, z0, n_iter,
    )

    # u_star estimate from geostrophic drag law
    u_star = KAPPA * u_geostrophic / np.log(DOMAIN_Z / z0)

    # Pressure gradient: dp/dx = -rho*f*V_g, dp/dy = rho*f*U_g
    # (rho=1 for kinematic pressure in simpleFoam)
    dp_dx = -f * v_g
    dp_dy = f * u_g

    context = {
        "domain_x": DOMAIN_X,
        "domain_y": DOMAIN_Y,
        "domain_z": DOMAIN_Z,
        "nx": NX,
        "ny": NY,
        "nz": NZ,
        "z_grading": Z_GRADING,
        "u_g": u_g,
        "v_g": v_g,
        "u_star": u_star,
        "z0": z0,
        "f_coriolis": f,
        "dp_dx": dp_dx,
        "dp_dy": dp_dy,
        "n_iter": n_iter,
        "T_ref": T_ref,
        "solver_name": solver_name,
        "of_version": of_version,
    }

    # Write case files
    _write_block_mesh_dict(output_dir / "system" / "blockMeshDict", context)
    _write_control_dict(output_dir / "system" / "controlDict", context)
    _write_fv_schemes(output_dir / "system" / "fvSchemes")
    _write_fv_solution(output_dir / "system" / "fvSolution")
    _write_decompose_par_dict(output_dir / "system" / "decomposeParDict")
    _write_fv_options(output_dir / "constant" / "fvOptions", context)
    _write_momentum_transport(output_dir / "constant", context)
    _write_transport_properties(output_dir / "constant" / "transportProperties", context)
    _write_bc_U(output_dir / "0" / "U", context)
    _write_bc_p(output_dir / "0" / "p", context)
    _write_bc_k(output_dir / "0" / "k", context)
    _write_bc_epsilon(output_dir / "0" / "epsilon", context)
    _write_bc_nut(output_dir / "0" / "nut", context)

    # Allrun script
    _write_allrun(output_dir / "Allrun", context)

    # ParaView file
    (output_dir / "case.foam").touch()

    # Save context for later extraction
    with open(output_dir / "precursor_config.json", "w") as fp:
        json.dump(context, fp, indent=2)

    logger.info("Precursor case ready: %s", output_dir)
    return output_dir


def extract_precursor_profiles(
    case_dir: Path,
    n_sample_z: int = 100,
) -> dict:
    """Extract converged vertical profiles from a completed precursor run.

    Reads the last time step and computes column-averaged profiles of
    U, V, k, epsilon.

    Parameters
    ----------
    case_dir : Completed precursor case directory.
    n_sample_z : Number of points in output vertical profile.

    Returns
    -------
    dict with keys: z_levels, u_profile, v_profile, k_profile, eps_profile,
                    u_star, z0, wind_dir_profile.
    """
    from init_from_era5 import read_cell_centres

    centres = read_cell_centres(case_dir)
    z = centres[:, 2]

    # Find latest time directory
    time_dirs = sorted(
        [d for d in case_dir.iterdir()
         if d.is_dir() and d.name.replace('.', '').isdigit() and float(d.name) > 0],
        key=lambda d: float(d.name),
    )
    if not time_dirs:
        raise FileNotFoundError(f"No time directories in {case_dir}")

    latest = time_dirs[-1]
    logger.info("Reading profiles from %s", latest)

    # Read fields
    from init_from_era5 import _parse_of_scalar_field, _parse_of_vector_field
    U = _parse_of_vector_field(latest / "U") if (latest / "U").exists() else None
    k = _parse_of_scalar_field(latest / "k") if (latest / "k").exists() else None
    eps = _parse_of_scalar_field(latest / "epsilon") if (latest / "epsilon").exists() else None

    if U is None:
        raise FileNotFoundError(f"U field not found in {latest}")

    # Bin by height
    z_edges = np.linspace(0, z.max(), n_sample_z + 1)
    z_levels = 0.5 * (z_edges[:-1] + z_edges[1:])

    u_prof = np.zeros(n_sample_z)
    v_prof = np.zeros(n_sample_z)
    k_prof = np.zeros(n_sample_z)
    eps_prof = np.zeros(n_sample_z)

    for i in range(n_sample_z):
        mask = (z >= z_edges[i]) & (z < z_edges[i + 1])
        if np.any(mask):
            u_prof[i] = U[mask, 0].mean()
            v_prof[i] = U[mask, 1].mean()
            if k is not None:
                k_prof[i] = k[mask].mean()
            if eps is not None:
                eps_prof[i] = eps[mask].mean()

    # Compute combined speed profile
    speed = np.sqrt(u_prof**2 + v_prof**2)

    # Wind direction profile (meteorological)
    wind_dir_prof = (270.0 - np.degrees(np.arctan2(v_prof, u_prof))) % 360.0

    result = {
        "z_levels": z_levels.tolist(),
        "u_profile": speed.tolist(),
        "u_x_profile": u_prof.tolist(),
        "u_y_profile": v_prof.tolist(),
        "k_profile": k_prof.tolist(),
        "eps_profile": eps_prof.tolist(),
        "wind_dir_profile": wind_dir_prof.tolist(),
        "source": "precursor",
    }

    logger.info(
        "Extracted profiles: speed range [%.1f, %.1f] m/s, "
        "direction range [%.0f, %.0f]°",
        speed.min(), speed.max(),
        wind_dir_prof[speed > 0.5].min() if np.any(speed > 0.5) else 0,
        wind_dir_prof[speed > 0.5].max() if np.any(speed > 0.5) else 0,
    )

    return result


# ---------------------------------------------------------------------------
# Case file writers (minimal, no Jinja2 — small enough to inline)
# ---------------------------------------------------------------------------

def _of_header(cls: str, obj: str, loc: str = "") -> str:
    loc_line = f'    location    "{loc}";\n' if loc else ""
    return (
        "/*--------------------------------*- C++ -*----------------------------------*\\\n"
        "  =========                 |\n"
        "  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox\n"
        "   \\\\    /   O peration     |\n"
        "    \\\\  /    A nd           |\n"
        "     \\\\/     M anipulation  |\n"
        "\\*---------------------------------------------------------------------------*/\n"
        "FoamFile\n"
        "{\n"
        "    version     2.0;\n"
        "    format      ascii;\n"
        f"    class       {cls};\n"
        f"{loc_line}"
        f"    object      {obj};\n"
        "}\n"
        "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
    )


def _mkwrite(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _write_block_mesh_dict(path: Path, ctx: dict) -> None:
    """Flat box with cyclic x/y, wall bottom, slip top."""
    dx = ctx["domain_x"]
    dy = ctx["domain_y"]
    dz = ctx["domain_z"]
    content = _of_header("dictionary", "blockMeshDict", "system") + f"""\
convertToMeters 1;

vertices
(
    (0   0   0)
    ({dx} 0   0)
    ({dx} {dy} 0)
    (0   {dy} 0)
    (0   0   {dz})
    ({dx} 0   {dz})
    ({dx} {dy} {dz})
    (0   {dy} {dz})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({ctx['nx']} {ctx['ny']} {ctx['nz']}) simpleGrading (1 1 {ctx['z_grading']})
);

boundary
(
    bottom
    {{
        type wall;
        faces
        (
            (0 3 2 1)
        );
    }}
    top
    {{
        type patch;
        faces
        (
            (4 5 6 7)
        );
    }}
    inlet_outlet_x
    {{
        type cyclic;
        neighbourPatch inlet_outlet_x_slave;
        faces
        (
            (0 4 7 3)
        );
    }}
    inlet_outlet_x_slave
    {{
        type cyclic;
        neighbourPatch inlet_outlet_x;
        faces
        (
            (1 2 6 5)
        );
    }}
    inlet_outlet_y
    {{
        type cyclic;
        neighbourPatch inlet_outlet_y_slave;
        faces
        (
            (0 1 5 4)
        );
    }}
    inlet_outlet_y_slave
    {{
        type cyclic;
        neighbourPatch inlet_outlet_y;
        faces
        (
            (3 7 6 2)
        );
    }}
);

// ************************************************************************* //
"""
    _mkwrite(path, content)


def _write_control_dict(path: Path, ctx: dict) -> None:
    content = _of_header("dictionary", "controlDict", "system") + f"""\
application     {ctx['solver_name']};

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {ctx['n_iter']};
deltaT          1;

writeControl    timeStep;
writeInterval   {ctx['n_iter']};  // Write only final time
purgeWrite      1;

writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;

runTimeModifiable yes;

// ************************************************************************* //
"""
    _mkwrite(path, content)


def _write_fv_schemes(path: Path) -> None:
    content = _of_header("dictionary", "fvSchemes", "system") + """\
ddtSchemes      { default steadyState; }
gradSchemes     { default Gauss linear; }
divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,k)      bounded Gauss upwind;
    div(phi,epsilon) bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes    { default corrected; }

// ************************************************************************* //
"""
    _mkwrite(path, content)


def _write_fv_solution(path: Path) -> None:
    content = _of_header("dictionary", "fvSolution", "system") + """\
solvers
{
    p   { solver GAMG; smoother GaussSeidel; tolerance 1e-6; relTol 0.01; }
    U   { solver PBiCGStab; preconditioner DILU; tolerance 1e-7; relTol 0.01; }
    k   { solver PBiCGStab; preconditioner DILU; tolerance 1e-7; relTol 0.01; }
    epsilon { solver PBiCGStab; preconditioner DILU; tolerance 1e-7; relTol 0.01; }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent yes;
    residualControl { p 1e-4; U 1e-4; k 1e-4; epsilon 1e-4; }
}

relaxationFactors
{
    fields    { p 0.3; }
    equations { U 0.7; k 0.5; epsilon 0.5; }
}

// ************************************************************************* //
"""
    _mkwrite(path, content)


def _write_decompose_par_dict(path: Path) -> None:
    content = _of_header("dictionary", "decomposeParDict", "system") + """\
numberOfSubdomains 4;
method scotch;

// ************************************************************************* //
"""
    _mkwrite(path, content)


def _write_fv_options(path: Path, ctx: dict) -> None:
    """Coriolis + pressure gradient forcing (geostrophic balance)."""
    content = _of_header("dictionary", "fvOptions", "constant") + f"""\
// Geostrophic pressure gradient + Coriolis
// dp/dx = -rho*f*V_g = {ctx['dp_dx']:.6e}
// dp/dy = +rho*f*U_g = {ctx['dp_dy']:.6e}
// f = {ctx['f_coriolis']:.6e} (lat={np.degrees(np.arcsin(ctx['f_coriolis'] / (2 * OMEGA))):.2f}°)

pressureGradient
{{
    type            vectorCodedSource;
    selectionMode   all;
    fields          (U);
    name            geostrophicPressureGradient;

    codeAddSup
    #{{
        const scalarField& V = mesh().V();
        vectorField& source = eqn.source();
        // Pressure gradient force = -grad(p)/rho
        // On RHS of momentum eqn, we subtract the source
        forAll(source, cellI)
        {{
            source[cellI].x() += {ctx['dp_dx']:.6e} * V[cellI];
            source[cellI].y() += {ctx['dp_dy']:.6e} * V[cellI];
        }}
    #}};
}}

coriolisForce
{{
    type            vectorCodedSource;
    selectionMode   all;
    fields          (U);
    name            coriolisSource;

    codeAddSup
    #{{
        const scalar f = {ctx['f_coriolis']:.6e};
        const volVectorField& U = mesh().lookupObject<volVectorField>("U");
        const scalarField& V = mesh().V();

        vectorField& source = eqn.source();
        forAll(U, cellI)
        {{
            source[cellI].x() -= f * U[cellI].y() * V[cellI];
            source[cellI].y() += f * U[cellI].x() * V[cellI];
        }}
    #}};
}}

// ************************************************************************* //
"""
    _mkwrite(path, content)


def _write_momentum_transport(constant_dir: Path, ctx: dict) -> None:
    obj = "turbulenceProperties" if ctx["of_version"] == 9 else "momentumTransport"
    path = constant_dir / obj
    content = _of_header("dictionary", obj) + """\
simulationType RAS;

RAS
{
    model           kEpsilon;
    turbulence      on;
    printCoeffs     on;
}

// ************************************************************************* //
"""
    _mkwrite(path, content)


def _write_transport_properties(path: Path, ctx: dict) -> None:
    T_ref = ctx["T_ref"]
    content = _of_header("dictionary", "transportProperties") + f"""\
transportModel  Newtonian;
nu              [0 2 -1 0 0 0 0]    1.5e-5;
Pr              [0 0 0 0 0 0 0]     0.71;
Prt             [0 0 0 0 0 0 0]     0.7;
beta            [0 0 0 -1 0 0 0]    {1.0/T_ref:.4e};
TRef            [0 0 0 1 0 0 0]     {T_ref:.2f};

// ************************************************************************* //
"""
    _mkwrite(path, content)


def _write_bc_U(path: Path, ctx: dict) -> None:
    """U: cyclic x/y, noSlip bottom, slip top, init with log-law."""
    u_star = ctx["u_star"]
    z0 = ctx["z0"]
    fd_x = ctx["u_g"] / max(abs(ctx["u_g"]**2 + ctx["v_g"]**2)**0.5, 0.01)
    fd_y = ctx["v_g"] / max(abs(ctx["u_g"]**2 + ctx["v_g"]**2)**0.5, 0.01)
    u_init_x = u_star / KAPPA * np.log(50.0 / z0) * fd_x
    u_init_y = u_star / KAPPA * np.log(50.0 / z0) * fd_y

    content = _of_header("volVectorField", "U", "0") + f"""\
dimensions      [0 1 -1 0 0 0 0];

internalField   uniform ({u_init_x:.4f} {u_init_y:.4f} 0);

boundaryField
{{
    bottom
    {{
        type            noSlip;
    }}
    top
    {{
        type            slip;
    }}
    "(inlet_outlet_x|inlet_outlet_x_slave|inlet_outlet_y|inlet_outlet_y_slave)"
    {{
        type            cyclic;
    }}
}}

// ************************************************************************* //
"""
    _mkwrite(path, content)


def _write_bc_p(path: Path, ctx: dict) -> None:
    content = _of_header("volScalarField", "p", "0") + """\
dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    bottom
    {
        type            zeroGradient;
    }
    top
    {
        type            fixedValue;
        value           uniform 0;
    }
    "(inlet_outlet_x|inlet_outlet_x_slave|inlet_outlet_y|inlet_outlet_y_slave)"
    {
        type            cyclic;
    }
}

// ************************************************************************* //
"""
    _mkwrite(path, content)


def _write_bc_k(path: Path, ctx: dict) -> None:
    k_init = ctx["u_star"]**2 / CMU**0.5
    content = _of_header("volScalarField", "k", "0") + f"""\
dimensions      [0 2 -2 0 0 0 0];

internalField   uniform {k_init:.6f};

boundaryField
{{
    bottom
    {{
        type            kqRWallFunction;
        value           uniform {k_init:.6f};
    }}
    top
    {{
        type            zeroGradient;
    }}
    "(inlet_outlet_x|inlet_outlet_x_slave|inlet_outlet_y|inlet_outlet_y_slave)"
    {{
        type            cyclic;
    }}
}}

// ************************************************************************* //
"""
    _mkwrite(path, content)


def _write_bc_epsilon(path: Path, ctx: dict) -> None:
    k_init = ctx["u_star"]**2 / CMU**0.5
    eps_init = CMU**0.75 * k_init**1.5 / (KAPPA * max(ctx["z0"] * 2.0, 0.1))
    content = _of_header("volScalarField", "epsilon", "0") + f"""\
dimensions      [0 2 -3 0 0 0 0];

internalField   uniform {eps_init:.6e};

boundaryField
{{
    bottom
    {{
        type            epsilonWallFunction;
        value           uniform {eps_init:.6e};
    }}
    top
    {{
        type            zeroGradient;
    }}
    "(inlet_outlet_x|inlet_outlet_x_slave|inlet_outlet_y|inlet_outlet_y_slave)"
    {{
        type            cyclic;
    }}
}}

// ************************************************************************* //
"""
    _mkwrite(path, content)


def _write_bc_nut(path: Path, ctx: dict) -> None:
    k_init = ctx["u_star"]**2 / CMU**0.5
    nut_init = CMU * k_init**2 / (CMU**0.75 * k_init**1.5 / (KAPPA * 80.0))
    content = _of_header("volScalarField", "nut", "0") + f"""\
dimensions      [0 2 -1 0 0 0 0];

internalField   uniform {nut_init:.4f};

boundaryField
{{
    bottom
    {{
        type            nutkWallFunction;
        Cmu             0.09;
        kappa           0.41;
        E               9.8;
        value           uniform {nut_init:.4f};
    }}
    top
    {{
        type            zeroGradient;
    }}
    "(inlet_outlet_x|inlet_outlet_x_slave|inlet_outlet_y|inlet_outlet_y_slave)"
    {{
        type            cyclic;
    }}
}}

// ************************************************************************* //
"""
    _mkwrite(path, content)


def _write_allrun(path: Path, ctx: dict) -> None:
    content = f"""#!/bin/sh
cd ${{0%/*}} || exit 1
. $WM_PROJECT_DIR/bin/tools/RunFunctions

runApplication blockMesh
runApplication {ctx['solver_name']}

echo "========================================"
echo "Precursor run complete."
echo "========================================"
"""
    _mkwrite(path, content)
    path.chmod(0o755)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Generate 1-D precursor ABL case")
    parser.add_argument("--u-geostrophic", type=float, required=True,
                        help="Geostrophic wind speed [m/s]")
    parser.add_argument("--wind-dir", type=float, required=True,
                        help="Meteorological wind direction [degrees from N]")
    parser.add_argument("--z0", type=float, default=0.05,
                        help="Surface roughness length [m]")
    parser.add_argument("--latitude", type=float, default=39.716,
                        help="Site latitude [degrees]")
    parser.add_argument("--n-iter", type=int, default=10000,
                        help="Number of SIMPLE iterations")
    parser.add_argument("--T-ref", type=float, default=300.0,
                        help="Reference temperature [K]")
    parser.add_argument("--solver", default="simpleFoam",
                        help="Solver name")
    parser.add_argument("--of-version", type=int, default=10, choices=[9, 10],
                        help="OpenFOAM version")
    parser.add_argument("--output", required=True,
                        help="Output case directory")
    parser.add_argument("--extract", action="store_true",
                        help="Extract profiles from completed case (don't generate)")
    args = parser.parse_args()

    if args.extract:
        profiles = extract_precursor_profiles(Path(args.output))
        out_json = Path(args.output) / "precursor_profiles.json"
        with open(out_json, "w") as fp:
            json.dump(profiles, fp, indent=2)
        print(f"Profiles extracted → {out_json}")
    else:
        generate_precursor_case(
            output_dir=Path(args.output),
            u_geostrophic=args.u_geostrophic,
            wind_dir=args.wind_dir,
            z0=args.z0,
            latitude=args.latitude,
            n_iter=args.n_iter,
            T_ref=args.T_ref,
            solver_name=args.solver,
            of_version=args.of_version,
        )
