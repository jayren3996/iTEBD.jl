include("../src/iTEBD.jl")
using .iTEBD
using Test
#--- Test canonical
@testset "CanonicalForm" begin
    aklt = zeros(2,3,2)
    aklt[1,1,2] = +sqrt(2/3)
    aklt[1,2,1] = -sqrt(1/3)
    aklt[2,2,2] = +sqrt(1/3)
    aklt[2,3,1] = -sqrt(2/3)

    model = zeros(4,3,4)
    model[1,1,2] = +sqrt(2/3)
    model[1,2,1] = -sqrt(1/3)
    model[2,2,2] = +sqrt(1/3)
    model[2,3,1] = -sqrt(2/3)
    model[3,1,4] = +sqrt(2/3)
    model[3,2,3] = -sqrt(1/3)
    model[4,2,4] = +sqrt(1/3)
    model[4,3,3] = -sqrt(2/3)

    T,V = canonical(model,model)
    ov = inner(T...,aklt,aklt)
    @test V[1][1] ≈ V[1][2]
    @test V[2][1] ≈ V[2][2]
    @test ov ≈ 1.0
end

@testset "ITEBD" begin
    aklt = zeros(2,3,2)
    aklt[1,1,2] = +sqrt(2/3)
    aklt[1,2,1] = -sqrt(1/3)
    aklt[2,2,2] = +sqrt(1/3)
    aklt[2,3,1] = -sqrt(2/3)
    dt = 0.1
    rdim = 50
    H = begin
        ss = spinop("xx",1) + spinop("yy",1) + spinop("zz",1)
        h2 = ss + 1/3*ss^2
    end
    tebd = TEBD(H,dt, mode="i",bound=rdim)

    Ts = [rand(rdim,3,rdim), rand(rdim,3,rdim)]
    λs = [rand(rdim), rand(rdim)]
    # Best: 1.07s
    @time Ts,λs = tebd(Ts,λs,1000)
    @test inner(Ts...,aklt,aklt) ≈ 1.0 atol=1e-5
end
