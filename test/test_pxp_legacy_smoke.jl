using Test

@testset "PXP_LEGACY_SMOKE" begin
    root = dirname(@__DIR__)
    scripts = ["PXP_TEST_1.jl", "PXP_TEST_2.jl"]

    for script in scripts
        path = joinpath(@__DIR__, script)
        output = read(`$(Base.julia_cmd()) --project=$root --compile=min --startup-file=no $path --smoke`, String)
        matches = collect(eachmatch(r"(?m)(?:Z1 error|Z2 error|Error)(?: =|:) ([0-9.eE+-]+)", output))
        @test !isempty(matches)
        @test all(m -> parse(Float64, m.captures[1]) < 0.1, matches)
    end
end
