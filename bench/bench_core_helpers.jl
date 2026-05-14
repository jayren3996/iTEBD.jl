using BenchmarkTools
using Printf
using Random
using iTEBD

const SMOKE_MODE = "--smoke" in ARGS
const BENCH_SAMPLES = SMOKE_MODE ? 1 : 20
const BENCH_SECONDS = SMOKE_MODE ? 0.02 : 1.0

function bench_estimate(f::F) where {F}
    f()
    trial = @benchmark $f() samples=BENCH_SAMPLES evals=1 seconds=BENCH_SECONDS
    estimate = median(trial)
    return (
        time_ns=estimate.time,
        allocs=estimate.allocs,
        bytes=estimate.memory,
    )
end

function main()
    Random.seed!(1234)
    spectrum = collect(range(1.0, 0.001; length=1000))
    f() = iTEBD._discarded_weight_choice(
        spectrum;
        mindim=4,
        maxdim=128,
        truncerr=1e-8,
        svd_min=1e-12,
    )

    result = f()
    metrics = bench_estimate(f)

    println("iTEBD Core Helper Benchmark")
    println("mode = $(SMOKE_MODE ? "smoke" : "full")")
    @printf("%-34s %12s %12s %12s %12s\n", "case", "median_us", "allocs", "bytes", "chi_keep")
    @printf(
        "%-34s %12.3f %12d %12d %12d\n",
        "discarded_weight_choice",
        metrics.time_ns / 1e3,
        metrics.allocs,
        metrics.bytes,
        result.chi_keep,
    )
    SMOKE_MODE && println("core helper smoke validation passed")
end

main()
