using Test

@testset "BENCH_SMOKE" begin
    bench_project = joinpath(pwd(), "bench")
    bench_script = joinpath(bench_project, "bench_trotter_vs_manual.jl")
    output = read(
        `$(Base.julia_cmd()) --project=$bench_project -e "using Pkg; Pkg.resolve(); Pkg.instantiate(); include(raw\"$bench_script\")" -- --smoke`,
        String,
    )

    @test occursin("manual_applygate_second_order", output)
    @test occursin("layered_evolve_second_order", output)
    @test occursin("precomputed_gate_list_second_order", output)
    @test occursin("layered_real_time_orders", output)
end
