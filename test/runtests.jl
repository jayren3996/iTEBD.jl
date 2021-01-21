include("../src/iTEBD.jl")
using Test
using LinearAlgebra
using TensorOperations
#---------------------------------------------------------------------------------------------------
# test basis quantum circuits
#---------------------------------------------------------------------------------------------------
import .iTEBD: tensor_lmul!, tensor_rmul!, tensor_umul, tensor_group, tensor_decomp, applygate!
@testset "Basic Multiplication" begin
    # generate random tensor and values
    rand_tensor = rand(7, 5, 7)
    rand_values = rand(7)
    rand_val_diag = Diagonal(rand_values)
    rand_mat = rand(5,5)
    # calculate target values
    @tensor lmul_target[:] := rand_val_diag[-1,1] * rand_tensor[1,-2,-3]
    @tensor rmul_target[:] := rand_tensor[-1,-2,1] * rand_val_diag[1,-3]
    @tensor umul_target[:] := rand_mat[-2,1] * rand_tensor[-1,1,-3]
    # test result
    tensor_lmul!(rand_values, rand_tensor)
    @test rand_tensor ≈ lmul_target
    tensor_rmul!(rand_tensor, rand_values)
    @test rand_tensor ≈ rmul_target
    rand_tensor = tensor_umul(rand_mat, rand_tensor)
    @test rand_tensor ≈ umul_target
end

import .iTEBD: spinmat, itebd, rand_iMPS
@testset "iTEBD" begin
    dt = 0.1
    rdim = 50
    H = begin
        ss = spinmat("xx",3) + spinmat("yy",3) + spinmat("zz",3)
        h2 = ss + 1/3*ss^2
    end
    sys = itebd(H, dt, mode="i", bound=rdim)
    mps = rand_iMPS(2, 3, rdim)
    # Best: 0.85s
    @time for i=1:1000
        mps = sys(mps)
    end
    @test inner_product(mps, aklt) ≈ 1.0 atol=1e-5
end




