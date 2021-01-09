include("../src/iTEBD.jl")
using .iTEBD
#using iTEBD
using Test
#--- Test canonical
const aklt = begin
    aklt_tensor = zeros(2,3,2)
    aklt_tensor[1,1,2] = +sqrt(2/3)
    aklt_tensor[1,2,1] = -sqrt(1/3)
    aklt_tensor[2,2,2] = +sqrt(1/3)
    aklt_tensor[2,3,1] = -sqrt(2/3)
    aklt_tensor
    iMPS([aklt_tensor, aklt_tensor])
end

@testset "CanonicalForm" begin
    aklt_canonical = canonical(aklt)
    ov = inner_product(aklt, aklt_canonical)
    @test ov ≈ 1.0
    @test aklt_canonical.λ[1][1] ≈ 1/sqrt(2)
    @test aklt_canonical.λ[2][2] ≈ 1/sqrt(2)
    
end

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




