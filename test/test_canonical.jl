include("../src/iTEBD.jl")
using Test
using LinearAlgebra
using TensorOperations
import .iTEBD: trm
import .iTEBD: fixed_point, canonical, block_canonical, inner_product
import .iTEBD: right_cannonical, schmidt_canonical
#---------------------------------------------------------------------------------------------------
# test basis quantum circuits
#---------------------------------------------------------------------------------------------------
const AKLT = begin
    tensor = zeros(2,3,2)
    tensor[1,1,2] = +sqrt(2/3)
    tensor[1,2,1] = -sqrt(1/3)
    tensor[2,2,2] = +sqrt(1/3)
    tensor[2,3,1] = -sqrt(2/3)
    tensor
end

const GHZ = begin
    tensor = zeros(2,2,2)
    tensor[1,1,1] = 1
    tensor[2,2,2] = 1
    tensor
end

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
        nres, tres = block_canonical(GHZ_RU)
        @test length(nres) == 2
        @test length(tres) == 2
        @test nres[1] ≈ 1.0 atol = 1e-5
        @test nres[2] ≈ 1.0 atol = 1e-5
        test1 = checkres(tres[1])
        test2 = checkres(tres[2])
        @test any(test1)
        @test any(test2)
        @test test1 .+ test2 == [1, 1]

        # GHZ under random positive non-unitary rotation
        rand_V = rand(2, 2) + I(2)
        rand_Vi = inv(rand_V)
        @tensor GHZ_RV[:] := rand_V[-1,1] * GHZ[1,-2,2] * rand_Vi[2,-3]
        nres, tres = block_canonical(GHZ_RV)
        @test length(nres) == 2
        @test length(tres) == 2
        @test nres[1] ≈ 1.0 atol = 1e-5
        @test nres[2] ≈ 1.0 atol = 1e-5
        test1 = checkres(tres[1])
        test2 = checkres(tres[2])
        @test any(test1)
        @test any(test2)
        @test test1 .+ test2 == [1, 1]
    end
end

@testset "Double AKLT State" begin
    double_aklt = zeros(4,3,4)
    double_aklt[1:2, :, 1:2] .= AKLT
    double_aklt[3:4, :, 3:4] .= AKLT
    for i=1:100
        # Double AKLT under random unitary rotation
        rand_U = exp( -1im * Hermitian( rand(4, 4) ) )
        @tensor double_aklt_RU[:] := rand_U[-1,1] * double_aklt[1,-2,2] * rand_U'[2,-3]
        nres, tres = block_canonical(double_aklt)
        @test length(nres) == 2
        @test length(tres) == 2
        @test nres[1] ≈ 1.0 atol = 1e-5
        @test nres[2] ≈ 1.0 atol = 1e-5
        @test inner_product(AKLT, tres[1]) ≈ 1.0 atol=1e-5
        @test inner_product(AKLT, tres[2]) ≈ 1.0 atol=1e-5

        # Double AKLT under random non-unitary rotation
        rand_V = rand(4, 4)
        rand_Vi = inv(rand_V)
        @tensor double_aklt_RV[:] := rand_V[-1,1] * double_aklt[1,-2,2] * rand_Vi[2,-3]
        nres, tres = block_canonical(double_aklt)
        @test length(nres) == 2
        @test length(tres) == 2
        @test nres[1] ≈ 1.0 atol = 1e-5
        @test nres[2] ≈ 1.0 atol = 1e-5
        @test inner_product(AKLT, tres[1]) ≈ 1.0 atol=1e-5
        @test inner_product(AKLT, tres[2]) ≈ 1.0 atol=1e-5
    end
end


