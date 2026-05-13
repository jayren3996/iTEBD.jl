@isdefined(iTEBD) || include("../src/iTEBD.jl")
using .iTEBD: iTEBD, product_iMPS, applygate!, _gate_indices, tensor_umul, tensor_umul!
using Test
using LinearAlgebra

@testset "GATE_INDICES_NON_WRAPPING" begin
    psi = product_iMPS(ComplexF64, [[1,0], [0,1], [1,0]])
    inds = _gate_indices(psi, 1, 2)
    @test inds == 1:2
    @test inds isa UnitRange
end

@testset "GATE_INDICES_WRAPPING" begin
    psi = product_iMPS(ComplexF64, [[1,0], [0,1], [1,0]])
    inds = _gate_indices(psi, 2, 1)
    @test collect(inds) == [2, 3, 1]
end

@testset "GATE_INDICES_SINGLE_SITE" begin
    psi = product_iMPS(ComplexF64, [[1,0], [0,1]])
    inds = _gate_indices(psi, 1, 1)
    # For i==j, _gate_indices returns the full unit cell wrap-around by design.
    @test collect(inds) == [1, 2, 1]
end

@testset "ONE_SITE_GATE_CORRECTNESS" begin
    X = [0.0 1.0; 1.0 0.0]
    psi = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
    applygate!(psi, X, 1, 1)
    @test psi.Γ[1][1,:,1] ≈ [0, 1] atol=1e-12
    @test psi.Γ[2][1,:,1] ≈ [0, 1] atol=1e-12
end

@testset "TENSOR_UMUL_INPLACE_CORRECTNESS" begin
    G = ComplexF64[0 1; 1 0]
    Γ = rand(ComplexF64, 3, 2, 3)
    Γ_orig = copy(Γ)
    Γ_new = tensor_umul(G, Γ)
    tensor_umul!(G, Γ)
    @test Γ ≈ Γ_new atol=1e-12
    @test Γ ≈ tensor_umul(G, Γ_orig) atol=1e-12
end
