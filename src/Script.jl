using ITensors
using LinearAlgebra

# AKLT
N = 10
s = siteinds("S=1", N)
ψ0 = randomMPS(s, 1)

function ITensors.op(
    ::OpName"AKLT", 
    ::SiteType"S=1",
    s1::Index, 
    s2::Index; 
    τ::Number
)
    O = 0.5 * op("S+", s1) * op("S-", s2) + 0.5 * op("S-", s1) * op("S+", s2) + op("Sz", s1) * op("Sz", s2)
    mO = reshape(array(O), 9,9)
    itensor(mO + 1/3 * mO^2, s1', s2', s1, s2)
end

τ = -0.1
os = [("AKLT", (i, i+1), (τ = τ,)) for i=1:N-1]
expτH = ops(os, s)
ψτ = ψ0
for j=1:500
    ψτ = MPS(ψτ ./ norm(ψτ)^(1/N))
    ψτ = apply(expτH, ψτ, maxdim=30, cutoff=1e-5)
end