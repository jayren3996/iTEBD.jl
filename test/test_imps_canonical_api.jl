@isdefined(iTEBD) || include("../src/iTEBD.jl")
using .iTEBD: iTEBD, rand_iMPS, canonical!, iMPS
using Test
using LinearAlgebra

@testset "IMPS_CANONICAL_CONVENTION" begin
    psi = rand_iMPS(ComplexF64, 1, 2, 4)
    @test canonical!(psi) === psi
    @test length(psi.Γ) == 1
    @test length(psi.λ) == 1

    G = psi.Γ[1]
    Dl, d, Dr = size(G)
    right_overlap = zeros(ComplexF64, Dl, Dl)
    for s in 1:d
        Bs = reshape(G[:, s, :], Dl, Dr)
        right_overlap .+= Bs * Bs'
    end
    @test right_overlap ≈ Matrix{ComplexF64}(I, Dl, Dl) atol=1e-10
end

@testset "NO_BLOCK_CANONICAL_EXPORT" begin
    @test !isdefined(iTEBD, :block_canonical)
end
