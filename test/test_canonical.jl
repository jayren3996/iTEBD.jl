include("../src/iTEBD.jl")
using .iTEBD
using Test
using LinearAlgebra
using TensorOperations
import .iTEBD: spin, trm, right_canonical, block_decomp

#---------------------------------------------------------------------------------------------------
# Random tensors: (1,3,1) + (2,3,2) + (3,3,3) + (4,3,4) 
#---------------------------------------------------------------------------------------------------
@testset "RANDOM_TENSOR" begin
    TALLY = 0
    for i=1:1000
        ten1 = rand(1, 3, 1)
        ten2 = rand(2, 3, 2)
        ten3 = rand(3, 3, 3)
        ten4 = rand(4, 3, 4)
        ten1 /= sqrt(inner_product(ten1))
        ten2 /= sqrt(inner_product(ten2))
        ten3 /= sqrt(inner_product(ten3))
        ten4 /= sqrt(inner_product(ten4))
        tens = [ten1, ten2, ten3, ten4]

        # Block
        tenc = zeros(10, 3, 10)
        tenc[1:1, :, 1:1] .= ten1
        tenc[2:3, :, 2:3] .= ten2
        tenc[4:6, :, 4:6] .= ten3
        tenc[7:10, :, 7:10] .= ten4
        rand_V = rand(10, 10)
        @tensor tenr[:] := rand_V[-1,1] * tenc[1,-2,2] * inv(rand_V)[2,-3]

        tenr = right_canonical(tenr, krylov_power=10000)
        res = block_decomp(tenr, krylov_power=10000)

        if length(res)==4
            TALLY += 1
            for j=1:4
                dim = size(res[j], 1)
                @test inner_product(res[j], tens[dim]) ≈ 1.0 atol=1e-4
            end
        end
    end
    println("Pass Rate: $(TALLY/10) %")
end


