using Test
using LinearAlgebra
using Random
using ITensors
using ITensorMPS

Random.seed!(20260525)

@isdefined(iTEBD) || include("../src/iTEBD.jl")
using .iTEBD: iTEBD, iMPS, rand_iMPS, product_iMPS, canonical!, expect, getindex
using .iTEBD: MAXDIM, SVDTOL, SORTTOL, ZEROTOL

#-----------------------------------------------------------------------
# Issue 1: imps2mps converts product iMPS to finite MPS
#-----------------------------------------------------------------------
@testset "IMPS2MPS_PRODUCT_STATE" begin
    ψ = product_iMPS(ComplexF64, [[1, 0]])
    sites = [Index(2, "site=$i") for i in 1:4]
    mps = iTEBD.imps2mps(ψ, sites)

    @test mps isa ITensorMPS.AbstractMPS
    @test length(mps) == length(sites)
end

@testset "IMPS2MPS_REPEATS_UNIT_CELL_TO_REQUESTED_LENGTH" begin
    ψ = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
    sites = [Index(2, "site=$i") for i in 1:5]
    mps = iTEBD.imps2mps(ψ, sites; L=length(sites))

    @test mps isa ITensorMPS.AbstractMPS
    @test length(mps) == length(sites)
end

@testset "IMPS2MPS_L1_DISTINCT_BOUNDARY_INDICES" begin
    # For L == 1 the wrap-around convention collapses left == right onto the
    # same Index, producing a self-contracted ITensor. The fix allocates a
    # distinct boundary Index, so the resulting tensor has three independent
    # legs.
    ψ = product_iMPS(ComplexF64, [[1, 0]])
    sites = [Index(2, "site=1")]
    mps = iTEBD.imps2mps(ψ, sites; L=1)

    @test mps isa ITensorMPS.AbstractMPS
    @test length(mps) == 1
    inds_t = inds(mps[1])
    @test length(inds_t) == 3
    @test length(unique(inds_t)) == 3
end

#-----------------------------------------------------------------------
# Issue 2: Global constants have correct types
#-----------------------------------------------------------------------
@testset "GLOBAL_CONSTANT_TYPES" begin
    @test typeof(MAXDIM) === Int
    @test typeof(SVDTOL) === Float64
    @test typeof(SORTTOL) === Float64
    @test typeof(ZEROTOL) === Float64
end

#-----------------------------------------------------------------------
# Issue 3: iMPS struct uses Int not Int64
#-----------------------------------------------------------------------
@testset "IMPS_STRUCT_INT_PLATFORM" begin
    ψ = product_iMPS(ComplexF64, [[1, 0]])
    @test ψ.n isa Int
    @test typeof(ψ.n) === Int
end

#-----------------------------------------------------------------------
# Issue 4: Constructors use _schmidt_value_type
#-----------------------------------------------------------------------
@testset "SCHMIDT_VALUE_TYPE_RESPECTED" begin
    ψ = rand_iMPS(ComplexF32, 2, 2, 4)
    @test eltype(ψ.Γ[1]) === ComplexF32
    @test eltype(ψ.λ[1]) === Float32

    ψ2 = product_iMPS(ComplexF32, [[1.0f0, 0.0f0], [0.0f0, 1.0f0]])
    @test eltype(ψ2.Γ[1]) === ComplexF32
    @test eltype(ψ2.λ[1]) === Float32
end

#-----------------------------------------------------------------------
# Issue 5: getindex type stability with Float32 Schmidt values
#-----------------------------------------------------------------------
@testset "GETINDEX_FLOAT32_TOLERANCE" begin
    ψ = rand_iMPS(ComplexF32, 2, 2, 4)
    Γ, λ = ψ[1]
    @test eltype(λ) === Float32
    @test eltype(Γ) === ComplexF32
end

#-----------------------------------------------------------------------
# Issue 6: expect with wrap-around and non-wrap-around
#-----------------------------------------------------------------------
@testset "EXPECT_WRAPAROUND_TYPES" begin
    ψ = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
    Z = [1.0 0.0; 0.0 -1.0]

    # Non-wrap-around j >= i
    val1 = expect(ψ, kron(Z, Z), 1, 2)
    @test val1 ≈ -1.0 atol=1e-12

    # Wrap-around j < i
    val2 = expect(ψ, kron(Z, Z), 2, 1)
    @test val2 ≈ -1.0 atol=1e-12
end

#-----------------------------------------------------------------------
# Issue 7: iMPS inner constructor invariants
#
# The inner constructor enforces (a) length(Γ) == length(λ) == n > 0,
# (b) size(Γ[i], 3) == length(λ[i]) and matches size(Γ[i+1], 1) at the
# wraparound seam, and (c) λ are finite and non-negative.
#-----------------------------------------------------------------------
@testset "IMPS_INNER_CONSTRUCTOR_VALID_INPUTS_OK" begin
    Γ = [randn(ComplexF64, 2, 2, 3), randn(ComplexF64, 3, 2, 2)]
    λ = [ones(3), ones(2)]
    ψ = iMPS(Γ, λ, 2)
    @test ψ.n == 2
    @test length(ψ.Γ) == 2
    @test length(ψ.λ) == 2
end

@testset "IMPS_INNER_CONSTRUCTOR_REJECTS_LENGTH_MISMATCH" begin
    Γ = [randn(ComplexF64, 2, 2, 2)]
    λ = [ones(2), ones(2)]
    @test_throws ArgumentError iMPS(Γ, λ, 2)  # length(Γ) != n
    @test_throws ArgumentError iMPS(Γ, λ, 1)  # length(λ) != n
end

@testset "IMPS_INNER_CONSTRUCTOR_REJECTS_NONPOSITIVE_N" begin
    @test_throws ArgumentError iMPS(Array{ComplexF64, 3}[], Vector{Float64}[], 0)
    @test_throws ArgumentError iMPS(Array{ComplexF64, 3}[], Vector{Float64}[], -1)
end

@testset "IMPS_INNER_CONSTRUCTOR_REJECTS_BOND_DIM_MISMATCH" begin
    # size(Γ[1], 3) = 3 but size(Γ[2], 1) = 5 — fails at wraparound seam.
    Γ = [randn(ComplexF64, 2, 2, 3), randn(ComplexF64, 5, 2, 2)]
    λ = [ones(3), ones(2)]
    @test_throws DimensionMismatch iMPS(Γ, λ, 2)

    # length(λ[1]) = 4 but size(Γ[1], 3) = 3 — fails λ vs Γ check.
    Γ2 = [randn(ComplexF64, 2, 2, 3), randn(ComplexF64, 3, 2, 2)]
    λ2 = [ones(4), ones(2)]
    @test_throws DimensionMismatch iMPS(Γ2, λ2, 2)
end

@testset "IMPS_INNER_CONSTRUCTOR_REJECTS_BAD_SCHMIDT_VALUES" begin
    Γ = [randn(ComplexF64, 2, 2, 2), randn(ComplexF64, 2, 2, 2)]
    @test_throws ArgumentError iMPS(Γ, [Float64[1.0, -0.5], Float64[1.0, 1.0]], 2)
    @test_throws ArgumentError iMPS(Γ, [Float64[1.0, NaN], Float64[1.0, 1.0]], 2)
    @test_throws ArgumentError iMPS(Γ, [Float64[1.0, Inf], Float64[1.0, 1.0]], 2)
end

@testset "IMPS_INNER_CONSTRUCTOR_REJECTS_RAW_BAD_TENSOR_LIST" begin
    # The outer iMPS(T, Γs; renormalize=false) constructor must also surface
    # the inner-constructor's invariants — previously this path produced a
    # silently malformed state.
    Γs = [randn(ComplexF64, 2, 2, 3), randn(ComplexF64, 5, 2, 2)]
    @test_throws DimensionMismatch iMPS(ComplexF64, Γs; renormalize=false)
end
