"""
ScalarTransport — GPU-accelerated steady-state passive scalar transport on structured grids.

Solves the steady advection-diffusion equation for a passive scalar phi:
    U . nabla(phi) = kappa_eff * laplacian(phi)

Designed for FuXi-CFD wind fields (300x300x27 @ 30m).
All operations use CUDA.jl CuArrays for GPU execution.
No scalar indexing — all operations are vectorized broadcast or sliced views.

Reference:
    Ferziger, Peric (2002). Computational Methods for Fluid Dynamics.
    Chapter 3 (Finite Volume) + Chapter 5 (Scalar Transport).
"""
module ScalarTransport

using CUDA

export solve_transport, transport_T_q, build_terrain_mask
export HF_Z_LEVELS, NX, NY, NZ, DX, DY, compute_dz

# ── FuXi-CFD HuggingFace dataset grid specification ──────────────────────────

const HF_Z_LEVELS = Float64[
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
    55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
    106.5, 114.95, 125.94, 140.22, 158.78, 182.91, 214.29,
]

const NX = 300
const NY = 300
const NZ = 27
const DX = 30.0  # meters
const DY = 30.0  # meters


"""
    compute_dz(z_levels) -> Vector{Float64}

Cell heights from level centers using midpoint rule.
"""
function compute_dz(z_levels::AbstractVector{Float64})
    n = length(z_levels)
    dz = zeros(Float64, n)
    dz[1] = (z_levels[1] + z_levels[2]) / 2.0
    for k in 2:n-1
        dz[k] = (z_levels[k+1] - z_levels[k-1]) / 2.0
    end
    dz[n] = z_levels[n] - (z_levels[n-1] + z_levels[n]) / 2.0
    return dz
end


# ── Advection (upwind, first order) ─────────────────────────────────────────

"""
    advection_x!(rhs, phi, u, dx)

Compute -d(u*phi)/dx using first-order upwind. In-place on GPU arrays.
Arrays are (nz, ny, nx).
"""
function advection_x!(rhs::CuArray{Float64,3}, phi::CuArray{Float64,3},
                      u::CuArray{Float64,3}, dx::Float64)
    nz, ny, nx = size(phi)

    # Face flux at i+1/2 for i=1..nx-1
    face = CUDA.zeros(Float64, nz, ny, nx - 1)
    @views @. face = ifelse(u[:,:,1:end-1] > 0,
                            u[:,:,1:end-1] * phi[:,:,1:end-1],
                            u[:,:,1:end-1] * phi[:,:,2:end])

    # rhs for interior cells (i=2..nx-1)
    @views @. rhs[:,:,2:end-1] += -(face[:,:,2:end] - face[:,:,1:end-1]) / dx

    return nothing
end

"""
    advection_y!(rhs, phi, v, dy)

Compute -d(v*phi)/dy using first-order upwind.
"""
function advection_y!(rhs::CuArray{Float64,3}, phi::CuArray{Float64,3},
                      v::CuArray{Float64,3}, dy::Float64)
    nz, ny, nx = size(phi)

    face = CUDA.zeros(Float64, nz, ny - 1, nx)
    @views @. face = ifelse(v[:,1:end-1,:] > 0,
                            v[:,1:end-1,:] * phi[:,1:end-1,:],
                            v[:,1:end-1,:] * phi[:,2:end,:])

    @views @. rhs[:,2:end-1,:] += -(face[:,2:end,:] - face[:,1:end-1,:]) / dy
    return nothing
end

"""
    advection_z!(rhs, phi, w, dz)

Compute -d(w*phi)/dz using first-order upwind with non-uniform spacing.
"""
function advection_z!(rhs::CuArray{Float64,3}, phi::CuArray{Float64,3},
                      w::CuArray{Float64,3}, dz_gpu::CuArray{Float64,1})
    nz, ny, nx = size(phi)
    dz3 = reshape(dz_gpu, nz, 1, 1)  # (nz, 1, 1) for broadcasting

    # Face flux at k+1/2 for k=1..nz-1
    face = CUDA.zeros(Float64, nz - 1, ny, nx)
    @views @. face = ifelse(w[1:end-1,:,:] > 0,
                            w[1:end-1,:,:] * phi[1:end-1,:,:],
                            w[1:end-1,:,:] * phi[2:end,:,:])

    # Interior cells (k=2..nz-1)
    @views @. rhs[2:end-1,:,:] += -(face[2:end,:,:] - face[1:end-1,:,:]) / dz3[2:end-1,:,:]

    # Bottom (k=1): only top face
    @views @. rhs[1:1,:,:] += -face[1:1,:,:] / dz3[1:1,:,:]

    # Top (k=nz): only bottom face
    @views @. rhs[end:end,:,:] += face[end:end,:,:] / dz3[end:end,:,:]

    return nothing
end


# ── Diffusion ────────────────────────────────────────────────────────────────

"""
    diffusion!(rhs, phi, dx, dy, dz_gpu, kappa)

Second-order central-difference Laplacian: kappa * nabla^2(phi).
Zero-gradient at z-boundaries.
"""
function diffusion!(rhs::CuArray{Float64,3}, phi::CuArray{Float64,3},
                    dx::Float64, dy::Float64,
                    dz_gpu::CuArray{Float64,1}, kappa::Float64)
    nz, ny, nx = size(phi)
    dz3 = reshape(dz_gpu, nz, 1, 1)

    # x-direction (interior only)
    @views @. rhs[:,:,2:end-1] += kappa * (phi[:,:,3:end] - 2*phi[:,:,2:end-1] + phi[:,:,1:end-2]) / dx^2

    # y-direction
    @views @. rhs[:,2:end-1,:] += kappa * (phi[:,3:end,:] - 2*phi[:,2:end-1,:] + phi[:,1:end-2,:]) / dy^2

    # z-direction (interior)
    @views @. rhs[2:end-1,:,:] += kappa * (phi[3:end,:,:] - 2*phi[2:end-1,:,:] + phi[1:end-2,:,:]) / dz3[2:end-1,:,:]^2

    # z-boundaries: zero-gradient Laplacian (use 1:1 range to avoid scalar indexing)
    @views @. rhs[1:1,:,:] += kappa * (phi[2:2,:,:] - phi[1:1,:,:]) / dz3[1:1,:,:]^2
    @views @. rhs[end:end,:,:] += kappa * (phi[end-1:end-1,:,:] - phi[end:end,:,:]) / dz3[end:end,:,:]^2

    return nothing
end


# ── Boundary conditions ──────────────────────────────────────────────────────

"""
    apply_bcs!(phi, bc3, bc_top, u, v, w, dz0, lapse_rate, terrain_mask)

Phi-based inlet/outlet BCs on all boundaries. All args are CuArrays.
bc3: (nz, 1, 1) profile for broadcasting.
bc_top: (1, 1) scalar broadcast for top BC (avoids scalar indexing on bc_profile[end]).
"""
function apply_bcs!(phi::CuArray{Float64,3},
                    bc3::CuArray{Float64,3},
                    bc_top::CuArray{Float64,2},
                    u::CuArray{Float64,3},
                    v::CuArray{Float64,3},
                    w::CuArray{Float64,3},
                    dz0::Float64,
                    lapse_rate::Float64,
                    terrain_mask::Union{Nothing,CuArray{Bool,3}})
    nz, ny, nx = size(phi)

    # West face (x=1): inflow if u > 0
    @views @. phi[:,:,1] = ifelse(u[:,:,1] > 0, bc3[:,:,1], phi[:,:,2])

    # East face (x=end): inflow if u < 0
    @views @. phi[:,:,end] = ifelse(u[:,:,end] < 0, bc3[:,:,1], phi[:,:,end-1])

    # South face (y=1): inflow if v > 0
    @views @. phi[:,1,:] = ifelse(v[:,1,:] > 0, bc3[:,1,:], phi[:,2,:])

    # North face (y=end): inflow if v < 0
    @views @. phi[:,end,:] = ifelse(v[:,end,:] < 0, bc3[:,1,:], phi[:,end-1,:])

    # Top: Dirichlet for subsidence/calm, Neumann for updraft
    # bc_top is (1,1) CuArray to avoid scalar indexing
    @views @. phi[end:end,:,:] = ifelse(w[end:end,:,:] > 0.01,
                                         phi[end-1:end-1,:,:],
                                         bc_top)

    # Bottom: zero-gradient or lapse rate
    if abs(lapse_rate) < 1e-10
        @views @. phi[1:1,:,:] = phi[2:2,:,:]
    else
        @views @. phi[1:1,:,:] = phi[2:2,:,:] + lapse_rate * dz0
    end

    # Terrain mask: vectorized — for each masked cell, copy from cell above.
    # We process bottom-up so each level sees the updated level above.
    # Use shifted mask+phi pairs (no scalar indexing, each iteration is a broadcast).
    if terrain_mask !== nothing
        for k in 1:nz-1
            # This is a broadcast over (ny, nx) — no scalar indexing
            kp = min(k + 1, nz)
            @views @. phi[k:k,:,:] = ifelse(terrain_mask[k:k,:,:],
                                             phi[kp:kp,:,:],
                                             phi[k:k,:,:])
        end
    end

    return nothing
end


# ── Terrain mask ─────────────────────────────────────────────────────────────

"""
    build_terrain_mask(dem, z_levels) -> CuArray{Bool,3}

Build 3D terrain mask: true where z_level < DEM elevation.
dem: (ny, nx) CPU array in meters ASL. Built on CPU, transferred to GPU.
"""
function build_terrain_mask(dem::AbstractMatrix{<:Real},
                            z_levels::AbstractVector{Float64})
    ny, nx = size(dem)
    nz = length(z_levels)
    # Vectorized on CPU: reshape z to (nz,1,1), broadcast against dem (1,ny,nx)
    z3 = reshape(z_levels, nz, 1, 1)
    dem3 = reshape(Float64.(dem), 1, ny, nx)
    mask = z3 .< dem3  # (nz, ny, nx) BitArray
    return CuArray(mask)
end


# ── Main solver ──────────────────────────────────────────────────────────────

"""
    solve_transport(u, v, w, bc_profile; kwargs...) -> (phi, info)

Solve steady-state passive scalar transport via pseudo-time Jacobi iteration.

All velocity arrays (u, v, w) should be CuArray{Float64,3} of shape (nz, ny, nx).
bc_profile: CuArray{Float64,1} of shape (nz,).

Returns phi (CuArray) and info Dict with convergence data.
"""
function solve_transport(u::CuArray{Float64,3},
                         v::CuArray{Float64,3},
                         w::CuArray{Float64,3},
                         bc_profile::CuArray{Float64,1};
                         dx::Float64=DX,
                         dy::Float64=DY,
                         dz::Vector{Float64}=compute_dz(HF_Z_LEVELS),
                         kappa::Float64=5.0,
                         max_iter::Int=500,
                         tol::Float64=1e-5,
                         lapse_rate::Float64=0.0,
                         terrain_mask::Union{Nothing,CuArray{Bool,3}}=nothing,
                         verbose::Bool=false)
    nz, ny, nx = size(u)
    dz_gpu = CuArray(dz)

    # Precompute BC arrays to avoid scalar indexing in the loop
    bc3 = reshape(bc_profile, nz, 1, 1)  # (nz, 1, 1) for broadcasting
    bc_top = reshape(bc_profile[end:end], 1, 1)  # (1, 1) — no scalar index
    dz0 = HF_Z_LEVELS[2] - HF_Z_LEVELS[1]  # CPU scalar, fine

    # Initialize: broadcast profile everywhere
    phi = repeat(reshape(bc_profile, nz, 1, 1), 1, ny, nx)

    # Stable pseudo-timestep (reduce on GPU, collect to CPU)
    u_max = max(CUDA.@allowscalar(maximum(abs.(u))),
                CUDA.@allowscalar(maximum(abs.(v))),
                CUDA.@allowscalar(maximum(abs.(w))),
                0.01)
    cfl_adv = min(dx, dy, minimum(dz)) / u_max
    cfl_diff = min(dx^2, dy^2, minimum(dz)^2) / (2 * kappa + 1e-30)
    dt = 0.25 * min(cfl_adv, cfl_diff)

    history = Float64[]
    rhs = CUDA.zeros(Float64, nz, ny, nx)

    for it in 1:max_iter
        fill!(rhs, 0.0)

        # Advection + diffusion
        advection_x!(rhs, phi, u, dx)
        advection_y!(rhs, phi, v, dy)
        advection_z!(rhs, phi, w, dz_gpu)
        diffusion!(rhs, phi, dx, dy, dz_gpu, kappa)

        # Update
        phi_new = phi .+ dt .* rhs

        # BCs
        apply_bcs!(phi_new, bc3, bc_top, u, v, w, dz0, lapse_rate, terrain_mask)

        # Convergence check — CUDA.@allowscalar for the single reduction
        change = CUDA.@allowscalar maximum(abs.(phi_new .- phi))
        push!(history, change)
        phi = phi_new

        if verbose && (it % 100 == 0 || change < tol)
            @info "iter $(lpad(it,4)): dmax=$(round(change; sigdigits=3))"
        end

        if change < tol
            verbose && @info "Converged at iteration $it"
            break
        end
    end

    info = Dict(
        "iterations" => length(history),
        "converged" => history[end] < tol,
        "final_residual" => history[end],
        "kappa" => kappa,
        "dt" => dt,
    )

    return phi, info
end


# ── Convenience: T and q transport ───────────────────────────────────────────

"""
    transport_T_q(u, v, w, dem; kwargs...) -> (T, q, info_T, info_q)

Compute T and q fields by passive transport on a FuXi-CFD wind field.
u, v, w: CuArray{Float64,3} (nz, ny, nx).
dem: Matrix{Float64} (ny, nx) — terrain elevation in meters.

Uses ISA standard atmosphere for T and exponential decay for q.
"""
function transport_T_q(u::CuArray{Float64,3},
                       v::CuArray{Float64,3},
                       w::CuArray{Float64,3},
                       dem::AbstractMatrix{<:Real};
                       kappa_T::Float64=5.0,
                       kappa_q::Float64=5.0,
                       max_iter::Int=500,
                       tol::Float64=1e-5,
                       verbose::Bool=false)
    # ISA standard atmosphere temperature profile
    T_profile = CuArray(288.15 .- 0.0065 .* HF_Z_LEVELS)

    # Exponential humidity profile
    q_profile = CuArray(0.008 .* exp.(-HF_Z_LEVELS ./ 2500.0))

    # Terrain mask
    terrain_mask = build_terrain_mask(dem, HF_Z_LEVELS)

    # Solve T
    verbose && @info "Solving T transport..."
    T, info_T = solve_transport(u, v, w, T_profile;
        kappa=kappa_T, lapse_rate=-0.0065,
        terrain_mask=terrain_mask, max_iter=max_iter, tol=tol, verbose=verbose)

    # Solve q
    verbose && @info "Solving q transport..."
    q, info_q = solve_transport(u, v, w, q_profile;
        kappa=kappa_q, lapse_rate=0.0,
        terrain_mask=terrain_mask, max_iter=max_iter, tol=tol, verbose=verbose)
    clamp!(q, 0.0, Inf)

    return T, q, info_T, info_q
end

end # module
