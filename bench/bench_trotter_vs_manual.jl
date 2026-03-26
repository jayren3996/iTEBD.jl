using BenchmarkTools
using LinearAlgebra
using Printf
using Random
using iTEBD

const SMOKE_MODE = "--smoke" in ARGS
const BENCH_SAMPLES = SMOKE_MODE ? 1 : 10
const BENCH_SECONDS = SMOKE_MODE ? 0.02 : 1.0
const AKLT_STEPS = SMOKE_MODE ? 10 : 1000
const AKLT_MAXDIM = 50
const REALTIME_CONFIGS = SMOKE_MODE ? [(0.2, 4, "budget"), (0.1, 8, "half_dt")] :
                                      [(0.2, 20, "budget"), (0.1, 40, "half_dt")]

Base.@kwdef struct BenchRow
    case::String
    variant::String
    median_ms::Float64
    allocs::Int
    bytes::Int
    prep_ms::Union{Nothing, Float64}=nothing
    agreement::Union{Nothing, Float64}=nothing
    error::Union{Nothing, Float64}=nothing
end

function bench_estimate(f::F) where {F}
    f()
    trial = @benchmark $f() samples=BENCH_SAMPLES evals=1 seconds=BENCH_SECONDS
    estimate = median(trial)
    return (
        time_ms=estimate.time / 1e6,
        allocs=estimate.allocs,
        bytes=estimate.memory,
    )
end

function format_metric(x::Union{Nothing, Float64}; digits::Int=6)
    isnothing(x) && return "-"
    return @sprintf("%.*f", digits, x)
end

function format_bytes(bytes::Integer)
    units = ("B", "KiB", "MiB", "GiB")
    value = Float64(bytes)
    idx = 1
    while value >= 1024 && idx < length(units)
        value /= 1024
        idx += 1
    end
    return @sprintf("%.2f %s", value, units[idx])
end

function gate_product(gates)
    dim = size(first(gates)[1], 1)
    U = Matrix{ComplexF64}(I, dim, dim)
    for (G, _, _) in gates
        U = Matrix{ComplexF64}(G) * U
    end
    return U
end

function normalized_overlap(ψ1, ψ2)
    num = abs(iTEBD.inner_product(ψ1, ψ2))
    den = sqrt(abs(iTEBD.inner_product(ψ1, ψ1)) * abs(iTEBD.inner_product(ψ2, ψ2)))
    return num / den
end

function aklt_setup()
    sx = sqrt(2) / 2 * [0 1 0; 1 0 1; 0 1 0]
    sy = sqrt(2) / 2 * 1im * [0 -1 0; 1 0 -1; 0 1 0]
    sz = [1 0 0; 0 0 0; 0 0 -1]
    ss = kron(sx, sx) + kron(sy, sy) + kron(sz, sz)
    h = ss + 1 / 3 * ss^2 + 2 / 3 * I(9)
    dt = 0.1
    gate = exp(-dt * h)
    layers = [[(h, 1, 2)], [(h, 2, 1)]]
    Random.seed!(1234)
    template = iTEBD.rand_iMPS(2, 3, AKLT_MAXDIM)
    return (; dt, h, gate, layers, template)
end

function manual_aklt!(ψ, gate, steps; maxdim::Integer)
    for _ in 1:steps
        applygate!(ψ, gate, 1, 2; maxdim)
        applygate!(ψ, gate, 2, 1; maxdim)
    end
    return ψ
end

function layered_aklt!(ψ, layers, dt, steps; maxdim::Integer)
    evolve!(ψ, layers, dt, steps; trotter=:second, evolution=:imaginary, maxdim=maxdim)
    return ψ
end

function precomputed_aklt!(ψ, gates, steps; maxdim::Integer)
    evolve!(ψ, gates, steps; maxdim=maxdim)
    return ψ
end

function aklt_rows()
    setup = aklt_setup()
    prep = bench_estimate(() -> trotter_gates(setup.layers, setup.dt; trotter=:second, evolution=:imaginary))
    gates = trotter_gates(setup.layers, setup.dt; trotter=:second, evolution=:imaginary)

    manual_ref = manual_aklt!(deepcopy(setup.template), setup.gate, AKLT_STEPS; maxdim=AKLT_MAXDIM)
    layered_ref = layered_aklt!(deepcopy(setup.template), setup.layers, setup.dt, AKLT_STEPS; maxdim=AKLT_MAXDIM)
    precomputed_ref = precomputed_aklt!(deepcopy(setup.template), gates, AKLT_STEPS; maxdim=AKLT_MAXDIM)

    manual_metrics = bench_estimate(() -> manual_aklt!(deepcopy(setup.template), setup.gate, AKLT_STEPS; maxdim=AKLT_MAXDIM))
    layered_metrics = bench_estimate(() -> layered_aklt!(deepcopy(setup.template), setup.layers, setup.dt, AKLT_STEPS; maxdim=AKLT_MAXDIM))
    precomputed_metrics = bench_estimate(() -> precomputed_aklt!(deepcopy(setup.template), gates, AKLT_STEPS; maxdim=AKLT_MAXDIM))

    return BenchRow[
        BenchRow(
            case="manual_applygate_second_order",
            variant="aklt",
            median_ms=manual_metrics.time_ms,
            allocs=manual_metrics.allocs,
            bytes=manual_metrics.bytes,
            agreement=1.0,
        ),
        BenchRow(
            case="layered_evolve_second_order",
            variant="aklt",
            median_ms=layered_metrics.time_ms,
            allocs=layered_metrics.allocs,
            bytes=layered_metrics.bytes,
            prep_ms=prep.time_ms,
            agreement=normalized_overlap(layered_ref, manual_ref),
        ),
        BenchRow(
            case="precomputed_gate_list_second_order",
            variant="aklt",
            median_ms=precomputed_metrics.time_ms,
            allocs=precomputed_metrics.allocs,
            bytes=precomputed_metrics.bytes,
            prep_ms=prep.time_ms,
            agreement=normalized_overlap(precomputed_ref, manual_ref),
        ),
    ]
end

function realtime_setup()
    x = ComplexF64[0 1; 1 0]
    z = ComplexF64[1 0; 0 -1]
    a = kron(x, I(2))
    b = kron(z, x)
    layers = [[(a, 1, 2)], [(b, 1, 2)]]
    template = product_iMPS(ComplexF64, [[1, 0], [1, 0]])
    return (; a, b, layers, template)
end

function realtime_exact_error(setup, dt::Real, steps::Integer, order::Symbol)
    total_time = dt * steps
    exact = exp(-1im * total_time * (setup.a + setup.b))
    stages = iTEBD._trotter_stage_schedule(length(setup.layers), order, steps)
    gates = iTEBD._materialize_trotter_gates(setup.layers, dt, stages; evolution=:real)
    approx = gate_product(gates)
    return opnorm(approx - exact)
end

function realtime_rows()
    setup = realtime_setup()
    rows = BenchRow[]

    for (dt, steps, config_name) in REALTIME_CONFIGS
        for order in (:second, :fourth, :fourth_opt)
            evolve_once!() = begin
                ψ = deepcopy(setup.template)
                evolve!(ψ, setup.layers, dt, steps; trotter=order, evolution=:real, maxdim=4)
                return ψ
            end
            metrics = bench_estimate(evolve_once!)
            push!(
                rows,
                BenchRow(
                    case="layered_real_time_orders",
                    variant="$(order)@$(config_name)",
                    median_ms=metrics.time_ms,
                    allocs=metrics.allocs,
                    bytes=metrics.bytes,
                    error=realtime_exact_error(setup, dt, steps, order),
                ),
            )
        end
    end

    return rows
end

function print_rows(rows)
    println("iTEBD Trotter Benchmark")
    println("mode = $(SMOKE_MODE ? "smoke" : "full")")
    println()
    @printf(
        "%-36s %-18s %12s %12s %12s %12s %12s %12s\n",
        "case",
        "variant",
        "median_ms",
        "allocs",
        "bytes",
        "prep_ms",
        "agreement",
        "exact_err",
    )
    println(repeat("-", 132))
    for row in rows
        @printf(
            "%-36s %-18s %12.3f %12d %12s %12s %12s %12s\n",
            row.case,
            row.variant,
            row.median_ms,
            row.allocs,
            format_bytes(row.bytes),
            format_metric(row.prep_ms; digits=3),
            format_metric(row.agreement),
            format_metric(row.error),
        )
    end
end

function main()
    rows = vcat(aklt_rows(), realtime_rows())
    print_rows(rows)
end

main()
