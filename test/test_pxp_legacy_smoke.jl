using Test

@testset "PXP_LEGACY_SMOKE" begin
    root = dirname(@__DIR__)
    scripts = ["PXP_TEST_1.jl", "PXP_TEST_2.jl"]

    for script in scripts
        path = joinpath(@__DIR__, script)
        output = read(`$(Base.julia_cmd()) --project=$root --compile=min --startup-file=no $path --smoke`, String)
        @test occursin("error", lowercase(output))
    end
end
