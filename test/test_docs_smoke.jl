using Test

@testset "DOCS_SCAFFOLD" begin
    root = dirname(@__DIR__)
    @test isfile(joinpath(root, "docs", "make.jl"))
    @test isfile(joinpath(root, "docs", "src", "index.md"))
    @test isfile(joinpath(root, "docs", "Project.toml"))
end
