using Test

@testset "BENCH_SMOKE" begin
    root = dirname(@__DIR__)
    bench_project = joinpath(root, "bench")
    bench_script = joinpath(bench_project, "bench_trotter_vs_manual.jl")
    output = read(
        `$(Base.julia_cmd()) --project=$bench_project -e "using Pkg; Pkg.resolve(); Pkg.instantiate(); include(raw\"$bench_script\")" -- --smoke`,
        String,
    )

    @test occursin("manual_applygate_second_order", output)
    @test occursin("layered_evolve_second_order", output)
    @test occursin("precomputed_gate_list_second_order", output)
    @test occursin("layered_real_time_orders", output)
    @test occursin("smoke validation passed", output)

    core_bench_script = joinpath(bench_project, "bench_core_helpers.jl")
    core_output = read(
        `$(Base.julia_cmd()) --project=$bench_project -e "using Pkg; Pkg.resolve(); Pkg.instantiate(); include(raw\"$core_bench_script\")" -- --smoke`,
        String,
    )

    @test occursin("discarded_weight_choice", core_output)
    @test occursin("core helper smoke validation passed", core_output)
end
