#!/usr/bin/env julia
"""
batch_transport.jl — Batch passive scalar transport on FuXi-CFD dataset.

Processes all (or a range of) FuXi-CFD cases, solving for T and q fields
using the GPU-accelerated transport solver, and saves results as npz.

Usage:
    julia --project=transport batch_transport.jl \\
        --data-dir /data/maitreje/fuxicfd/extracted \\
        [--start-idx 0] [--end-idx 12000] \\
        [--max-iter 500] [--tol 1e-5] \\
        [--kappa-T 5.0] [--kappa-q 5.0] \\
        [--verbose]
"""

using ArgParse
using CUDA
using NPZ
using JSON3
using ProgressMeter

# Activate project and load solver
push!(LOAD_PATH, joinpath(@__DIR__, "src"))
using ScalarTransport


function parse_args()
    s = ArgParseSettings(description="Batch T/q transport on FuXi-CFD wind fields")

    @add_arg_table! s begin
        "--data-dir"
            help = "Root directory containing case_NNNNNN/ folders"
            required = true
        "--start-idx"
            help = "First case index to process (0-based)"
            arg_type = Int
            default = 0
        "--end-idx"
            help = "Last case index (exclusive, 0-based). -1 = all"
            arg_type = Int
            default = -1
        "--max-iter"
            help = "Max pseudo-time iterations for solver"
            arg_type = Int
            default = 500
        "--tol"
            help = "Convergence tolerance"
            arg_type = Float64
            default = 1e-5
        "--kappa-T"
            help = "Thermal diffusivity [m^2/s]"
            arg_type = Float64
            default = 5.0
        "--kappa-q"
            help = "Moisture diffusivity [m^2/s]"
            arg_type = Float64
            default = 5.0
        "--verbose"
            help = "Print per-case solver details"
            action = :store_true
    end

    return ArgParse.parse_args(s)
end


"""
    discover_cases(data_dir) -> Vector{String}

Find all case directories (case_NNNNNN/) sorted by name.
"""
function discover_cases(data_dir::String)
    entries = readdir(data_dir; join=true)
    cases = filter(entries) do p
        isdir(p) && startswith(basename(p), "case_") &&
        isfile(joinpath(p, "outputs.npz")) &&
        isfile(joinpath(p, "inputs.npz"))
    end
    sort!(cases)
    return cases
end


"""
    process_case(case_dir; kwargs...) -> Dict

Load a FuXi-CFD case, solve T/q transport, save outputs_Tq.npz.
Returns stats dict with timing and convergence info.
"""
function process_case(case_dir::String;
                      kappa_T::Float64, kappa_q::Float64,
                      max_iter::Int, tol::Float64, verbose::Bool)
    t0 = time()

    # Load wind field (outputs.npz: u,v,w,k each 27x300x300)
    outputs = npzread(joinpath(case_dir, "outputs.npz"))
    u_cpu = Float64.(outputs["u"])  # (27, 300, 300)
    v_cpu = Float64.(outputs["v"])
    w_cpu = Float64.(outputs["w"])

    # Load terrain (inputs.npz: dem 300x300)
    inputs = npzread(joinpath(case_dir, "inputs.npz"))
    dem = Float64.(inputs["dem"])  # (300, 300)

    # Transfer to GPU
    u = CuArray(u_cpu)
    v = CuArray(v_cpu)
    w = CuArray(w_cpu)

    # Solve
    T, q, info_T, info_q = transport_T_q(u, v, w, dem;
        kappa_T=kappa_T, kappa_q=kappa_q,
        max_iter=max_iter, tol=tol, verbose=verbose)

    # Transfer back to CPU and convert to Float32 for storage
    T_cpu = Float32.(Array(T))
    q_cpu = Float32.(Array(q))

    # Save
    out_path = joinpath(case_dir, "outputs_Tq.npz")
    npzwrite(out_path, Dict(
        "T" => T_cpu,
        "q" => q_cpu,
    ))

    elapsed = time() - t0

    return Dict(
        "case" => basename(case_dir),
        "elapsed_s" => round(elapsed; digits=2),
        "T_iterations" => info_T["iterations"],
        "T_converged" => info_T["converged"],
        "T_residual" => info_T["final_residual"],
        "q_iterations" => info_q["iterations"],
        "q_converged" => info_q["converged"],
        "q_residual" => info_q["final_residual"],
    )
end


function main()
    args = parse_args()
    data_dir = args["data-dir"]
    verbose = args["verbose"]

    @info "Discovering cases in $data_dir..."
    all_cases = discover_cases(data_dir)
    @info "Found $(length(all_cases)) cases"

    # Select range
    start_idx = args["start-idx"] + 1  # Julia 1-based
    end_idx = args["end-idx"] == -1 ? length(all_cases) : args["end-idx"]
    cases = all_cases[start_idx:min(end_idx, length(all_cases))]
    @info "Processing cases $start_idx to $(start_idx + length(cases) - 1)"

    # Check GPU
    if CUDA.functional()
        @info "CUDA device: $(CUDA.name(CUDA.device()))"
    else
        error("CUDA not available. This script requires a GPU.")
    end

    # Process
    stats = Dict[]
    failed = String[]
    p = Progress(length(cases); desc="Transport solver", showspeed=true)

    for case_dir in cases
        try
            s = process_case(case_dir;
                kappa_T=args["kappa-T"], kappa_q=args["kappa-q"],
                max_iter=args["max-iter"], tol=args["tol"],
                verbose=verbose)
            push!(stats, s)

            if !s["T_converged"] || !s["q_converged"]
                push!(failed, s["case"])
            end
        catch e
            @warn "Failed on $(basename(case_dir)): $e"
            push!(failed, basename(case_dir))
            push!(stats, Dict(
                "case" => basename(case_dir),
                "error" => string(e),
            ))
        end
        next!(p)
    end

    # Summary
    n_processed = length(stats)
    n_converged = count(s -> get(s, "T_converged", false) && get(s, "q_converged", false), stats)
    elapsed_total = sum(get(s, "elapsed_s", 0.0) for s in stats)
    avg_time = n_processed > 0 ? elapsed_total / n_processed : 0.0

    @info """
    === Batch Transport Summary ===
    Processed: $n_processed / $(length(cases))
    Converged: $n_converged / $n_processed
    Failed:    $(length(failed))
    Total time: $(round(elapsed_total; digits=1))s
    Avg time/case: $(round(avg_time; digits=2))s
    """

    # Save summary
    summary_path = joinpath(data_dir, "transport_summary.json")
    open(summary_path, "w") do io
        JSON3.pretty(io, Dict(
            "n_processed" => n_processed,
            "n_converged" => n_converged,
            "n_failed" => length(failed),
            "failed_cases" => failed,
            "total_time_s" => round(elapsed_total; digits=1),
            "avg_time_per_case_s" => round(avg_time; digits=3),
            "settings" => Dict(
                "max_iter" => args["max-iter"],
                "tol" => args["tol"],
                "kappa_T" => args["kappa-T"],
                "kappa_q" => args["kappa-q"],
            ),
            "per_case" => stats,
        ))
    end
    @info "Summary saved to $summary_path"

    # Exit code
    if length(failed) > 0
        @warn "$(length(failed)) cases failed or did not converge"
    end
end

main()
