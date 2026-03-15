using Test

@testset "DOCS_SCAFFOLD" begin
    @test isfile("docs/make.jl")
    @test isfile("docs/src/index.md")
    @test isfile("docs/Project.toml")
end
