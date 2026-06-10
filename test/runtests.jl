using Test
using iTEBD
using InfiniteTEBD

@testset "iTEBD shell wrapper" begin
    @test iTEBD.InfiniteTEBD === InfiniteTEBD
    @test !iTEBD._reexports_name(Symbol("#eval"))

    for name in names(InfiniteTEBD; all=false, imported=false)
        @test name in names(iTEBD; all=false, imported=false)
        @test getproperty(iTEBD, name) === getproperty(InfiniteTEBD, name)
    end

    for name in (:tensor_decomp!, :schmidt_canonical, :imps2mps, :SORTTOL, :ZEROTOL)
        @test isdefined(iTEBD, name)
        @test getproperty(iTEBD, name) === getproperty(InfiniteTEBD, name)
    end

    psi = iTEBD.product_iMPS(ComplexF64, [[1, 0], [0, 1]])
    @test psi isa InfiniteTEBD.DenseIMPS
    @test iTEBD.inner_product(psi) ≈ 1.0 atol=1e-12

    package_root = dirname(dirname(pathof(iTEBD)))
    duplicated_source_files = [
        "Contractions.jl",
        "Gate.jl",
        "ITensorsInterop.jl",
        "Krylov.jl",
        "Miscellaneous.jl",
        "ScarFinder.jl",
        "Schmidt.jl",
        "SymmetricStubs.jl",
        "TensorAlgebra.jl",
        "iMPS.jl",
    ]

    @test all(!isfile(joinpath(package_root, "src", file)) for file in duplicated_source_files)
    @test !isdir(joinpath(package_root, "ext"))
end
