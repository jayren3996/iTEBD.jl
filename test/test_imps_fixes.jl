using Test
using LinearAlgebra
using ITensors
using ITensorMPS

@isdefined(iTEBD) || include("../src/iTEBD.jl")
using .iTEBD: iTEBD, iMPS, rand_iMPS, product_iMPS, canonical!, expect, getindex
using .iTEBD: MAXDIM, SVDTOL, SORTTOL, ZEROTOL, KRLOV_POWER

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

#-----------------------------------------------------------------------
# Issue 2: Global constants have correct types
#-----------------------------------------------------------------------
@testset "GLOBAL_CONSTANT_TYPES" begin
    @test typeof(MAXDIM) === Int
    @test typeof(SVDTOL) === Float64
    @test typeof(SORTTOL) === Float64
    @test typeof(ZEROTOL) === Float64
    @test typeof(KRLOV_POWER) === Int
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
