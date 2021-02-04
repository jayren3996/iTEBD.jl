include("../src/iTEBD.jl")
using Test
using LinearAlgebra
using TensorOperations
using .iTEBD
import .iTEBD: block_canonical, spin
#---------------------------------------------------------------------------------------------------
# Constant Objects
#---------------------------------------------------------------------------------------------------
const AKLT = begin
    tensor = zeros(2,3,2)
    tensor[1,1,2] = +sqrt(2/3)
    tensor[1,2,1] = -sqrt(1/3)
    tensor[2,2,2] = +sqrt(1/3)
    tensor[2,3,1] = -sqrt(2/3)
    tensor
end

const AKLT_MPS = iMPS([AKLT, AKLT])
const AKLT_MPS_3 = iMPS([AKLT, AKLT, AKLT])

const GHZ = begin
    tensor = zeros(2,2,2)
    tensor[1,1,1] = 1
    tensor[2,2,2] = 1
    tensor
end

#---------------------------------------------------------------------------------------------------
# Test imaginary-time iTEBD 
#
# 1. Imaginary-time evolving under AKLT Hamiltonian.
# 2. The ourcome is compared to AKLT MPS.
#---------------------------------------------------------------------------------------------------
@testset "AKLT_iTEBD" begin
    dt = 0.1
    rdim = 50

    GA, GB = begin
        ss = spin("xx",3) + spin("yy",3) + spin("zz",3)
        H = ss + 1/3*ss^2 + 2/3*I(9)
        expH = exp(- dt * H)
        gate(expH, [1,2], bound=rdim), gate(expH, [2,1], bound=rdim)
    end
    
    mps = rand_iMPS(2, 3, rdim)
    
    println("First-time run:")
    @time for i=1:1000
        applygate!(mps, GA)
        applygate!(mps, GB)
    end

    @test inner_product(mps, AKLT_MPS) ≈ 1.0 atol=1e-5
    if size(mps.Γ[1]) != (2, 3, 2)
        mps = canonical(mps, trim=true, bound=rdim)
    end
    @test size(mps.Γ[1]) == (2, 3, 2)
    @test size(mps.Γ[2]) == (2, 3, 2)
    @test mps.λ[1] ≈ [1/sqrt(2), 1/sqrt(2)] atol=1e-5
    @test mps.λ[2] ≈ [1/sqrt(2), 1/sqrt(2)] atol=1e-5

    # Bench-Mark
    # Best: 0.638020 seconds (228.75 k allocations: 529.783 MiB, 6.13% gc time)
    println("Second-time run:")
    mps = rand_iMPS(2, 3, rdim)
    @time for i=1:1000
        applygate!(mps, GA)
        applygate!(mps, GB)
    end
end
#---------------------------------------------------------------------------------------------------
@testset "AKLT_iTEBD_3" begin
    dt = 0.1
    rdim = 50

    GA, GB, GC = begin
        ss = spin("xx",3) + spin("yy",3) + spin("zz",3)
        H = ss + 1/3*ss^2 + 2/3*I(9)
        expH = exp(- dt * H)
        gate(expH, [1,2], bound=rdim), gate(expH, [2,3], bound=rdim), gate(expH, [3,1], bound=rdim)
    end

    mps = rand_iMPS(3, 3, rdim)

    @time for i=1:1000
        applygate!(mps, GA)
        applygate!(mps, GB)
        applygate!(mps, GC)
    end
    mps = canonical(mps, trim=true, bound=rdim)

    @test size(mps.Γ[1]) == (2, 3, 2)
    @test size(mps.Γ[2]) == (2, 3, 2)
    @test size(mps.Γ[3]) == (2, 3, 2)
    @test mps.λ[1] ≈ [1/sqrt(2), 1/sqrt(2)] atol=1e-5
    @test mps.λ[2] ≈ [1/sqrt(2), 1/sqrt(2)] atol=1e-5
    @test mps.λ[3] ≈ [1/sqrt(2), 1/sqrt(2)] atol=1e-5
    @test inner_product(mps, AKLT_MPS_3) ≈ 1.0 atol=1e-5
end

#---------------------------------------------------------------------------------------------------
# Test Block-canonical
#---------------------------------------------------------------------------------------------------
@testset "GHZ State" begin
    target_1 = zeros(1,2,1)
    target_1[1,1,1] = 1
    target_2 = zeros(1,2,1)
    target_2[1,2,1] = 1
    function checkres(res)
        b1 = isapprox(inner_product(res, target_1), 1.0, atol=1e-5)
        b2 = isapprox(inner_product(res, target_2), 1.0, atol=1e-5)
        return [b1, b2]
    end
    for i = 1:100
        # GHZ under random unitary rotation
        rand_U = exp( -1im * Hermitian( rand(2, 2) ) )
        @tensor GHZ_RU[:] := rand_U[-1,1] * GHZ[1,-2,2] * rand_U'[2,-3]
        res = block_canonical(GHZ_RU)
        @test length(res) == 2

        test1 = checkres(res[1])
        test2 = checkres(res[2])
        @test any(test1)
        @test any(test2)
        @test test1 .+ test2 == [1, 1]

        # GHZ under random positive non-unitary rotation
        rand_V = rand(2, 2) + I(2)
        rand_Vi = inv(rand_V)
        @tensor GHZ_RV[:] := rand_V[-1,1] * GHZ[1,-2,2] * rand_Vi[2,-3]
        res = block_canonical(GHZ_RV)

        @test length(res) == 2

        test1 = checkres(res[1])
        test2 = checkres(res[2])
        @test any(test1)
        @test any(test2)
        @test test1 .+ test2 == [1, 1]
    end
end
#---------------------------------------------------------------------------------------------------
@testset "Double AKLT State" begin
    double_aklt = zeros(4,3,4)
    double_aklt[1:2, :, 1:2] .= AKLT
    double_aklt[3:4, :, 3:4] .= AKLT
    for i=1:100
        # Double AKLT under random unitary rotation
        rand_U = exp( -1im * Hermitian( rand(4, 4) ) )
        @tensor double_aklt_RU[:] := rand_U[-1,1] * double_aklt[1,-2,2] * rand_U'[2,-3]
        res = block_canonical(double_aklt)

        @test length(res) == 2

        @test inner_product(AKLT, res[1]) ≈ 1.0 atol=1e-5
        @test inner_product(AKLT, res[2]) ≈ 1.0 atol=1e-5

        # Double AKLT under random non-unitary rotation
        rand_V = rand(4, 4)
        rand_Vi = inv(rand_V)
        @tensor double_aklt_RV[:] := rand_V[-1,1] * double_aklt[1,-2,2] * rand_Vi[2,-3]
        res = block_canonical(double_aklt)

        @test length(res) == 2

        @test inner_product(AKLT, res[1]) ≈ 1.0 atol=1e-5
        @test inner_product(AKLT, res[2]) ≈ 1.0 atol=1e-5
    end
end
