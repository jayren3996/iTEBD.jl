include("../src/iTEBD.jl")
using Test
using LinearAlgebra
using TensorOperations
using .iTEBD
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

const S1X = sqrt(2) / 2 * [0 1 0; 1 0 1; 0 1 0]
const S1Y = sqrt(2) / 2 * 1im * [0 -1 0; 1 0 -1; 0 1 0]
const S1Z = [1 0 0; 0 0 0; 0 0 -1]

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
        ss = kron(S1X, S1X) + kron(S1Y, S1Y) + kron(S1Z, S1Z)
        H = ss + 1/3*ss^2 + 2/3*I(9)
        expH = exp(- dt * H)
        expH, expH
    end
    
    mps = rand_iMPS(2, 3, rdim)
    
    println("First-time run:")
    @time for i=1:1000
        applygate!(mps, GA, 1, 2; maxdim=rdim)
        applygate!(mps, GB, 2, 1; maxdim=rdim)
    end

    @test inner_product(mps, AKLT_MPS) ≈ 1.0 atol=1e-5
    if size(mps.Γ[1]) != (2, 3, 2)
        canonical!(mps; maxdim=rdim)
    end
    @test size(mps.Γ[1]) == (2, 3, 2)
    @test size(mps.Γ[2]) == (2, 3, 2)
    @test mps.λ[1] ≈ [1/sqrt(2), 1/sqrt(2)] atol=1e-5
    @test mps.λ[2] ≈ [1/sqrt(2), 1/sqrt(2)] atol=1e-5

    # Bench-Mark
    # Best: 0.419608 seconds (221.96 k allocations: 271.772 MiB, 5.56% gc time)
    println("Second-time run:")
    mps = rand_iMPS(2, 3, rdim)
    @time for i=1:1000
        applygate!(mps, GA, 1, 2; maxdim=rdim)
        applygate!(mps, GB, 2, 1; maxdim=rdim)
    end
end
#---------------------------------------------------------------------------------------------------
@testset "AKLT_iTEBD_3" begin
    dt = 0.1
    rdim = 50

    GA, GB, GC = begin
        ss = kron(S1X, S1X) + kron(S1Y, S1Y) + kron(S1Z, S1Z)
        H = ss + 1/3*ss^2 + 2/3*I(9)
        expH = exp(- dt * H)
        expH, expH, expH
    end

    mps = rand_iMPS(3, 3, rdim)

    @time for i=1:1000
        applygate!(mps, GA, 1, 2; maxdim=rdim)
        applygate!(mps, GB, 2, 3; maxdim=rdim)
        applygate!(mps, GC, 3, 1; maxdim=rdim)
    end
    canonical!(mps; maxdim=rdim)

    @test size(mps.Γ[1]) == (2, 3, 2)
    @test size(mps.Γ[2]) == (2, 3, 2)
    @test size(mps.Γ[3]) == (2, 3, 2)
    @test mps.λ[1] ≈ [1/sqrt(2), 1/sqrt(2)] atol=1e-5
    @test mps.λ[2] ≈ [1/sqrt(2), 1/sqrt(2)] atol=1e-5
    @test mps.λ[3] ≈ [1/sqrt(2), 1/sqrt(2)] atol=1e-5
    @test inner_product(mps, AKLT_MPS_3) ≈ 1.0 atol=1e-5
end


include("test_imps_canonical_api.jl")
include("test_adaptive_bonddim.jl")
include("test_evolve_api.jl")
include("test_scarfinder_nstep.jl")
include("test_docs_smoke.jl")
include("test_bench_smoke.jl")
