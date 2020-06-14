include("../src/Canonical.jl")
include("../src/iTEBD.jl")
using .Canonical: canonical, overlap, applygate
using .iTEBD
using LinearAlgebra
using Test
#--- Test canonical
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

res = canonical(model,model)
ov = overlap(res[1],res[2],aklt,aklt)
@test res[3][1] ≈ res[3][2]
@test res[4][1] ≈ res[4][2]
@test ov ≈ 1.0
#--- Test iTEBD
dt = 0.1
rdim = 50
H = begin
    ss = spinop("xx",1) + spinop("yy",1) + spinop("zz",1)
    h2 = ss + 1/3*ss^2
end
tebd = TEBD(H,dt, mode="i",bound=rdim)

mps = begin
    A = rand(rdim,3,rdim)
    la = Diagonal(rand(rdim))
    B = rand(rdim,3,rdim)
    lb = Diagonal(rand(rdim))
    (A,B,la,lb)
end


for i = 1:1000
    global mps
    mps = tebd(mps)
end

@test overlap(mps[1],mps[2],aklt,aklt) ≈ 1.0 atol=1e-5
