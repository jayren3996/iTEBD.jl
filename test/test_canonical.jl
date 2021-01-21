include("../src/iTEBD.jl")
using Test
using LinearAlgebra
using TensorOperations
import .iTEBD: fixed_point, canonical, block_canonical, inner_product
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
    for i = 1:10
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
        if !(any(test1)) 
            println("Vres[$i, 1]: $(inner_product(res[1], target_1)), $(inner_product(res[1], target_2))")
            println("Vres[$i, 2]: $(inner_product(res[2], target_1)), $(inner_product(res[2], target_2))")
            println("eig(V) = $(eigvals(rand_V))")
            println(res[1][:,1,:])
            println(res[1][:,2,:])
            println(res[2][:,1,:])
            println(res[2][:,2,:])
        end
        @test any(test1)
        @test any(test2)
        @test test1 .+ test2 == [1, 1]
    end
end

@testset "Double AKLT State" begin
    double_aklt = zeros(4,3,4)
    double_aklt[1:2, :, 1:2] .= AKLT
    double_aklt[3:4, :, 3:4] .= AKLT
    for i=1:10
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
"""
@testset "Complex Stacks" begin
    # Stacks of AKLT, GHZ, 2×2 random block, and zero-block
    rand_tensor = rand(2,3,2)
    stack = zeros(8,3,8)
    stack[1:2, :, 1:2] .= AKLT 
    stack[3, 1, 3] = 1
    stack[4, 3, 4] = 1
    stack[5:6, :, 5:6] .= rand_tensor
    # GHZ target
    target_1 = zeros(1,3,1)
    target_1[1,1,1] = 1
    target_2 = zeros(1,3,1)
    target_2[1,3,1] = 1
    # Under non-unitary rotation
    rand_V = rand(8,8)
    rand_Vi = inv(rand_V)
    @tensor stack_RV[:] := rand_V[-1,1] * stack[1,-2,2] * rand_Vi[2,-3]
    # results
    res = block_canonical(stack_RV)
    @test length(res) == 6
    function test_res(res)
        out = zeros(Bool, 5)
        out[1] = inner_product(res, AKLT) ≈ 1.0
        out[2] = inner_product(res, target_1) ≈ 1.0
        out[3] = inner_product(res, target_2) ≈ 1.0
        out[4] = inner_product(res, rand_tensor) ≈ 1.0
        out[5] = inner_product(res, res) ≈ 1.0
    end
    test_list = test_res.(res)
    @test any(test_list[1])
    @test any(test_list[2])
    @test any(test_list[3])
    @test any(test_list[4])
    @test any(test_list[5])
    @test any(test_list[6])
    @test sum(test_list) == [1,1,1,1,2]
end
"""

