"""
scalar_transport.py — Steady-state passive scalar transport on structured grids.

Solves the steady advection-diffusion equation for a passive scalar φ:

    U·∇φ = κ_eff ∇²φ

on a structured 3D grid (nz × ny × nx) with a fixed velocity field U.
Designed to compute T and q fields from FuXi-CFD wind output (300×300×27 @ 30m).

Discretization options:
  - Advection: upwind (1st order), QUICK (3rd order), TVD (2nd order, flux-limited)
  - Diffusion: central differences (2nd order)
  - Solver: pseudo-time-stepping Jacobi (GPU-friendly)

Boundary conditions:
  - Lateral: Dirichlet from ERA5 interpolated profile
  - Top: Dirichlet (free atmosphere = ERA5)
  - Bottom (terrain): zero-flux (∂φ/∂n = 0) or specified gradient (lapse rate)

Reference:
    Ferziger, Perić (2002). Computational Methods for Fluid Dynamics.
    Springer. Chapter 3 (Finite Volume) + Chapter 5 (Scalar Transport).
"""
from __future__ import annotations

from enum import Enum
from typing import NamedTuple

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class AdvectionScheme(Enum):
    UPWIND = "upwind"       # 1st order, very diffusive but unconditionally stable
    QUICK = "quick"         # 3rd order (Leonard 1979), good balance accuracy/stability
    TVD_VANLEER = "vanleer" # 2nd order TVD, van Leer limiter — no spurious oscillations


# ── FuXi-CFD grid specification ─────────────────────────────────────────────

# 27 vertical levels (non-uniform spacing, AGL in meters)
FUXI_Z_LEVELS = np.array([
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
    55, 60, 65, 70, 80, 90, 100, 110, 120, 130,
    140, 150, 160, 170, 180, 195, 214,
], dtype=np.float64)

FUXI_NX, FUXI_NY, FUXI_NZ = 300, 300, 27
FUXI_DX, FUXI_DY = 30.0, 30.0  # meters


def compute_dz(z_levels: np.ndarray) -> np.ndarray:
    """Compute cell heights from level centers.

    Uses midpoint rule: dz[k] = distance between midpoints of cells k-1/k and k/k+1.
    """
    n = len(z_levels)
    dz = np.zeros(n)

    # First cell: from ground (z=0) to midpoint between level 0 and 1
    dz[0] = (z_levels[0] + z_levels[1]) / 2.0

    # Interior cells
    for k in range(1, n - 1):
        dz[k] = (z_levels[k + 1] - z_levels[k - 1]) / 2.0

    # Last cell: symmetric about last level
    dz[-1] = z_levels[-1] - (z_levels[-2] + z_levels[-1]) / 2.0

    return dz


# ── Advection flux functions ────────────────────────────────────────────────

def _flux_upwind(phi_L: np.ndarray, phi_R: np.ndarray,
                 vel: np.ndarray) -> np.ndarray:
    """First-order upwind flux at face between L and R cells."""
    return np.where(vel > 0, vel * phi_L, vel * phi_R)


def _van_leer_limiter(r: np.ndarray) -> np.ndarray:
    """Van Leer flux limiter: ψ(r) = (r + |r|) / (1 + |r|)."""
    return (r + np.abs(r)) / (1.0 + np.abs(r) + 1e-30)


def _advection_x(phi: np.ndarray, u: np.ndarray, dx: float,
                 scheme: AdvectionScheme) -> np.ndarray:
    """Compute -∂(u·φ)/∂x using the specified scheme."""
    if scheme == AdvectionScheme.UPWIND:
        flux = _flux_upwind(phi, np.roll(phi, -1, axis=2), u)
        flux_m = _flux_upwind(np.roll(phi, 1, axis=2), phi, np.roll(u, 1, axis=2))
        return -(flux - flux_m) / dx

    elif scheme == AdvectionScheme.TVD_VANLEER:
        # Face values with TVD limiter
        # Forward differences
        dphi_fwd = np.roll(phi, -1, axis=2) - phi
        dphi_bwd = phi - np.roll(phi, 1, axis=2)
        dphi_2fwd = np.roll(phi, -2, axis=2) - np.roll(phi, -1, axis=2)

        # Ratio for limiter (upwind-ratio)
        r_pos = dphi_bwd / (dphi_fwd + 1e-30)  # when u > 0
        r_neg = dphi_2fwd / (dphi_fwd + 1e-30)  # when u < 0 (reverse direction)

        psi_pos = _van_leer_limiter(r_pos)
        psi_neg = _van_leer_limiter(r_neg)

        # Face value (TVD reconstruction)
        phi_face_pos = phi + 0.5 * psi_pos * dphi_fwd           # u > 0
        phi_face_neg = np.roll(phi, -1, axis=2) - 0.5 * psi_neg * dphi_fwd  # u < 0

        phi_face = np.where(u > 0, phi_face_pos, phi_face_neg)
        flux = u * phi_face

        # Same for left face
        dphi_fwd_m = phi - np.roll(phi, 1, axis=2)
        dphi_bwd_m = np.roll(phi, 1, axis=2) - np.roll(phi, 2, axis=2)
        dphi_2fwd_m = np.roll(phi, -1, axis=2) - phi

        r_pos_m = dphi_bwd_m / (dphi_fwd_m + 1e-30)
        r_neg_m = dphi_2fwd_m / (dphi_fwd_m + 1e-30)

        u_m = np.roll(u, 1, axis=2)
        phi_face_pos_m = np.roll(phi, 1, axis=2) + 0.5 * _van_leer_limiter(r_pos_m) * dphi_fwd_m
        phi_face_neg_m = phi - 0.5 * _van_leer_limiter(r_neg_m) * dphi_fwd_m
        phi_face_m = np.where(u_m > 0, phi_face_pos_m, phi_face_neg_m)
        flux_m = u_m * phi_face_m

        return -(flux - flux_m) / dx

    elif scheme == AdvectionScheme.QUICK:
        # QUICK: 3-point upstream-weighted (Leonard 1979)
        # For u > 0: φ_face = 3/8 φ_R + 6/8 φ_C - 1/8 φ_L
        phi_C = phi
        phi_R = np.roll(phi, -1, axis=2)
        phi_L = np.roll(phi, 1, axis=2)
        phi_LL = np.roll(phi, 2, axis=2)
        phi_RR = np.roll(phi, -2, axis=2)

        face_pos = (3 * phi_R + 6 * phi_C - phi_L) / 8.0
        face_neg = (3 * phi_C + 6 * phi_R - phi_RR) / 8.0
        phi_face = np.where(u > 0, face_pos, face_neg)
        flux = u * phi_face

        u_m = np.roll(u, 1, axis=2)
        face_pos_m = (3 * phi_C + 6 * phi_L - phi_LL) / 8.0
        face_neg_m = (3 * phi_L + 6 * phi_C - phi_R) / 8.0
        phi_face_m = np.where(u_m > 0, face_pos_m, face_neg_m)
        flux_m = u_m * phi_face_m

        return -(flux - flux_m) / dx


def _advection_y(phi: np.ndarray, v: np.ndarray, dy: float,
                 scheme: AdvectionScheme) -> np.ndarray:
    """Compute -∂(v·φ)/∂y — transpose of x logic."""
    # Swap axes 1<->2 to reuse _advection_x logic
    phi_t = np.swapaxes(phi, 1, 2)
    v_t = np.swapaxes(v, 1, 2)
    result = _advection_x(phi_t, v_t, dy, scheme)
    return np.swapaxes(result, 1, 2)


def _advection_z(phi: np.ndarray, w: np.ndarray, dz: np.ndarray,
                 scheme: AdvectionScheme) -> np.ndarray:
    """Compute -∂(w·φ)/∂z with non-uniform spacing.

    Uses upwind with explicit z-indexing (no np.roll wrap-around).
    """
    nz = phi.shape[0]
    result = np.zeros_like(phi)

    for k in range(1, nz - 1):
        # Upwind flux at top face of cell k
        flux_top = np.where(w[k] > 0, w[k] * phi[k], w[k] * phi[k + 1])
        # Upwind flux at bottom face of cell k
        flux_bot = np.where(w[k - 1] > 0, w[k - 1] * phi[k - 1], w[k - 1] * phi[k])
        result[k] = -(flux_top - flux_bot) / dz[k]

    # Bottom (k=0): only top face
    flux_top_0 = np.where(w[0] > 0, w[0] * phi[0], w[0] * phi[1])
    result[0] = -flux_top_0 / dz[0]

    # Top (k=nz-1): only bottom face
    flux_bot_top = np.where(w[-2] > 0, w[-2] * phi[-2], w[-2] * phi[-1])
    result[-1] = flux_bot_top / dz[-1]

    return result


# ── Diffusion ────────────────────────────────────────────────────────────────

def _diffusion(phi: np.ndarray, dx: float, dy: float, dz: np.ndarray,
               kappa: float) -> np.ndarray:
    """Second-order central-difference Laplacian: κ∇²φ.

    Uses zero-gradient (Neumann) at z-boundaries to avoid wrap-around artifacts.
    """
    nz = phi.shape[0]
    dz3 = dz[:, None, None]

    lap_x = (np.roll(phi, -1, axis=2) - 2 * phi + np.roll(phi, 1, axis=2)) / dx**2
    lap_y = (np.roll(phi, -1, axis=1) - 2 * phi + np.roll(phi, 1, axis=1)) / dy**2

    # z-direction: explicit indexing to avoid wrap
    lap_z = np.zeros_like(phi)
    for k in range(1, nz - 1):
        lap_z[k] = (phi[k + 1] - 2 * phi[k] + phi[k - 1]) / dz[k]**2
    # Boundaries: zero-gradient (∂²φ/∂z² = 0)
    lap_z[0] = (phi[1] - phi[0]) / dz[0]**2
    lap_z[-1] = (phi[-2] - phi[-1]) / dz[-1]**2

    return kappa * (lap_x + lap_y + lap_z)


# ── Boundary conditions ─────────────────────────────────────────────────────

def _apply_bcs(phi: np.ndarray, bc_profile: np.ndarray,
               terrain_mask: np.ndarray | None = None,
               lapse_rate: float = 0.0) -> None:
    """Apply boundary conditions in-place.

    Lateral + top: Dirichlet from ERA5 profile.
    Bottom: zero-gradient (if lapse_rate=0) or imposed gradient.
    Terrain: zero-flux (copy from cell above).
    """
    nz = phi.shape[0]

    # Lateral Dirichlet (all 4 faces)
    phi[:, :, 0] = bc_profile[:, None]
    phi[:, :, -1] = bc_profile[:, None]
    phi[:, 0, :] = bc_profile[:, None]
    phi[:, -1, :] = bc_profile[:, None]

    # Top: Dirichlet
    phi[-1, :, :] = bc_profile[-1]

    # Bottom: zero-gradient or lapse rate
    if abs(lapse_rate) < 1e-10:
        phi[0, :, :] = phi[1, :, :]
    else:
        dz0 = FUXI_Z_LEVELS[1] - FUXI_Z_LEVELS[0]
        phi[0, :, :] = phi[1, :, :] + lapse_rate * dz0

    # Terrain mask: copy from cell directly above
    if terrain_mask is not None:
        for k in range(1, nz):
            phi[k][terrain_mask[k]] = phi[min(k + 1, nz - 1)][terrain_mask[k]]
        phi[0][terrain_mask[0]] = phi[1][terrain_mask[0]]


# ── Main solver ──────────────────────────────────────────────────────────────

def solve_scalar_transport(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    bc_profile: np.ndarray,
    dx: float = FUXI_DX,
    dy: float = FUXI_DY,
    dz: np.ndarray | None = None,
    kappa: float = 5.0,
    terrain_mask: np.ndarray | None = None,
    lapse_rate: float = 0.0,
    scheme: str = "vanleer",
    max_iter: int = 500,
    tol: float = 1e-5,
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """Solve steady-state passive scalar transport.

    Args:
        u, v, w: velocity field (nz, ny, nx) from FuXi-CFD or CFD [m/s]
        bc_profile: boundary profile (nz,) — T(z) in K or q(z) in kg/kg
        dx, dy: horizontal grid spacing [m] (default 30m)
        dz: vertical spacing (nz,), computed from FUXI_Z_LEVELS if None
        kappa: turbulent diffusivity [m²/s] (~5-10 for ABL thermal)
        terrain_mask: (nz, ny, nx) or (ny, nx) bool — True inside terrain.
            If 2D, expanded to 3D assuming terrain at lowest levels.
        lapse_rate: dT/dz at ground [K/m], 0 = adiabatic (default)
        scheme: "upwind", "vanleer", "quick"
        max_iter: maximum pseudo-time iterations
        tol: convergence tolerance on max(|δφ|)
        verbose: print iteration info

    Returns:
        (phi, info): scalar field (nz, ny, nx) and convergence info dict
    """
    nz, ny, nx = u.shape
    adv_scheme = AdvectionScheme(scheme)

    if dz is None:
        dz = compute_dz(FUXI_Z_LEVELS[:nz])

    # Handle 2D terrain mask → 3D
    if terrain_mask is not None and terrain_mask.ndim == 2:
        # Assume terrain mask means "ground is present" at (y, x)
        # No 3D mask available without DEM — skip interior masking
        terrain_3d = None
    elif terrain_mask is not None and terrain_mask.ndim == 3:
        terrain_3d = terrain_mask
    else:
        terrain_3d = None

    # Initialize: inflow profile everywhere
    phi = np.broadcast_to(bc_profile[:, None, None], (nz, ny, nx)).copy().astype(np.float64)

    # Estimate stable pseudo-timestep
    u_max = max(np.abs(u).max(), np.abs(v).max(), np.abs(w).max(), 0.01)
    cfl_adv = min(dx, dy, float(dz.min())) / u_max
    cfl_diff = min(dx**2, dy**2, float(dz.min())**2) / (2 * kappa + 1e-30)
    dt = 0.25 * min(cfl_adv, cfl_diff)

    history = []

    for it in range(max_iter):
        # Advection
        rhs = _advection_x(phi, u, dx, adv_scheme)
        rhs += _advection_y(phi, v, dy, adv_scheme)
        rhs += _advection_z(phi, w, dz, adv_scheme)

        # Diffusion
        rhs += _diffusion(phi, dx, dy, dz, kappa)

        # Update
        phi_new = phi + dt * rhs

        # BCs
        _apply_bcs(phi_new, bc_profile, terrain_3d, lapse_rate)

        # Convergence
        change = np.abs(phi_new - phi).max()
        history.append(change)
        phi = phi_new

        if verbose and (it % 100 == 0 or change < tol):
            print(f"  iter {it:4d}: Δmax={change:.2e}, "
                  f"range=[{phi.min():.4f}, {phi.max():.4f}]")

        if change < tol:
            if verbose:
                print(f"  Converged at iteration {it}")
            break

    info = {
        "iterations": it + 1,
        "converged": change < tol,
        "final_residual": float(change),
        "history": history,
        "scheme": scheme,
        "kappa": kappa,
        "dt": dt,
    }

    return phi, info


# ── Convenience: process a FuXi-CFD case ─────────────────────────────────────

def transport_T_q_on_wind_field(
    u: np.ndarray,           # (27, 300, 300)
    v: np.ndarray,
    w: np.ndarray,
    T_profile: np.ndarray,   # (27,) ERA5 temperature [K]
    q_profile: np.ndarray,   # (27,) ERA5 specific humidity [kg/kg]
    terrain_mask: np.ndarray | None = None,
    kappa_T: float = 5.0,
    kappa_q: float = 5.0,
    lapse_rate_T: float = -0.0065,  # dry adiabatic lapse rate [K/m]
    scheme: str = "vanleer",
    max_iter: int = 500,
    tol: float = 1e-5,
    verbose: bool = False,
) -> dict:
    """Compute T and q fields by passive transport on a wind field.

    Args:
        u, v, w: wind components (nz, ny, nx) [m/s]
        T_profile: ERA5 temperature profile (nz,) [K]
        q_profile: ERA5 specific humidity profile (nz,) [kg/kg]
        terrain_mask: (ny, nx) or (nz, ny, nx) bool
        kappa_T: thermal diffusivity [m²/s] (5-10 typical for ABL)
        kappa_q: moisture diffusivity [m²/s] (≈ kappa_T)
        lapse_rate_T: surface temperature gradient [K/m] (-0.0065 = dry adiabatic)
        scheme: advection scheme ("upwind", "vanleer", "quick")

    Returns:
        dict with T (nz, ny, nx) [K], q (nz, ny, nx) [kg/kg], info_T, info_q.
    """
    if verbose:
        print("Solving T transport...")
    T, info_T = solve_scalar_transport(
        u, v, w, T_profile,
        kappa=kappa_T, terrain_mask=terrain_mask,
        lapse_rate=lapse_rate_T, scheme=scheme,
        max_iter=max_iter, tol=tol, verbose=verbose,
    )

    if verbose:
        print(f"  T: [{T.min():.1f}, {T.max():.1f}] K, {info_T['iterations']} iter")
        print("Solving q transport...")

    q, info_q = solve_scalar_transport(
        u, v, w, q_profile,
        kappa=kappa_q, terrain_mask=terrain_mask,
        lapse_rate=0.0, scheme=scheme,
        max_iter=max_iter, tol=tol, verbose=verbose,
    )
    q = np.clip(q, 0, None)

    if verbose:
        print(f"  q: [{q.min():.6f}, {q.max():.6f}] kg/kg, {info_q['iterations']} iter")

    return {"T": T, "q": q, "info_T": info_T, "info_q": info_q}
