using Test
using iTEBD

@testset "DOCS_SCAFFOLD" begin
    root = dirname(@__DIR__)
    @test isfile(joinpath(root, "docs", "make.jl"))
    @test isfile(joinpath(root, "docs", "src", "index.md"))
    @test isfile(joinpath(root, "docs", "Project.toml"))
end

@testset "DOCS_BINDINGS" begin
    @test haskey(Docs.meta(iTEBD), Docs.Binding(iTEBD, :canonical!))
end

@testset "DOCS_BUILD_SMOKE" begin
    root = dirname(@__DIR__)
    docs_dir = joinpath(root, "docs")
    make_jl = joinpath(docs_dir, "make.jl")
    command = addenv(
        `$(Base.julia_cmd()) --project=$docs_dir --compile=min --startup-file=no $make_jl`,
        "ITEBD_DOCS_SMOKE" => "true",
        "JULIA_NUM_THREADS" => "1",
    )
    process = run(ignorestatus(command))
    @test process.exitcode == 0
end
