include("../src/iTEBD.jl")
using .iTEBD
using Test
using LinearAlgebra
using TensorOperations
import .iTEBD: spinmat, trm, fixed_point, right_canonical, block_decomp, fixed_mat_2, vals_group

#---------------------------------------------------------------------------------------------------
# Random tensors: (1,3,1) + (2,3,2) + (3,3,3) + (4,3,4) 
#---------------------------------------------------------------------------------------------------
TENSOR = zeros(10, 3, 10)
FIXMAT = zeros(ComplexF64, 10, 10)

@testset "RANDOM_TENSOR" begin
    for i=1:100
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

        for j=1:2
            tenr = right_canonical(tenr)
        end

        res = block_decomp(tenr)

        @test length(res)==4
        if length(res)==4
            for j=1:4
                dim = size(res[j], 1)
                @test inner_product(res[j], tens[dim]) ≈ 1.0 atol=1e-4
            end
        else
            TENSOR .= tenr
            break
        end
    end
end

TENSOR_RC = right_canonical(TENSOR)
for j=1:2
    TENSOR_RC = right_canonical(TENSOR_RC)
end
VALS = eigvals(fixed_mat_2(trm(TENSOR_RC), 10))
println(
    group = vals_group(VALS)
)
