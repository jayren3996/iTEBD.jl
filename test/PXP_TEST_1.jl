include("../src/iTEBD.jl")

using .iTEBD
using DelimitedFiles
using LinearAlgebra
using Printf
using Test

const SMOKE_MODE = "--smoke" in ARGS
const DT = 0.1
const SUB_DIV = SMOKE_MODE ? 1 : 5
const BOND = SMOKE_MODE ? 16 : 400

function _reference_data()
    z1_path = joinpath(@__DIR__, "Ent_Z1.dat")
    z2_path = joinpath(dirname(@__DIR__), "examples", "Ent_Z2.dat")
    z1 = readdlm(z1_path)
    z2 = readdlm(z2_path)
    if SMOKE_MODE
        z1 = z1[1:min(4, size(z1, 1)), :]
        z2 = z2[1:min(4, size(z2, 1)), :]
    end
    return z1, z2
end

function _two_site_gate()
    V0 = 100
    sx = [0.0 0.5; 0.5 0.0]
    pn = [1.0 0.0; 0.0 0.0]
    h_pxp = kron(sx, I(2)) + kron(I(2), sx) + V0 * kron(pn, pn)
    return exp(-1im * (DT / SUB_DIV) * h_pxp)
end

function _run_series!(psi::iMPS, gate::AbstractMatrix, len::Integer)
    entropy = zeros(Float64, len)
    for i in 2:len
        for _ in 1:SUB_DIV
            applygate!(psi, gate, 1, 2; maxdim=BOND)
            applygate!(psi, gate, 2, 1; maxdim=BOND)
        end
        @printf("Finish: %.2f %%.\n", 100 * (i - 1) / max(len - 1, 1))
        entropy[i] = iTEBD.ent_S(psi, 1)
    end
    return entropy
end

z1_data, z2_data = _reference_data()
z1_entr = z1_data[:, 2]
z2_entr = z2_data[:, 2]
gate = _two_site_gate()

z1_mps = product_iMPS(ComplexF64, [[0, 1], [0, 1]])
z2_mps = product_iMPS(ComplexF64, [[0, 1], [1, 0]])

z1_entropy = _run_series!(z1_mps, gate, length(z1_entr))
z2_entropy = _run_series!(z2_mps, gate, length(z2_entr))

z1_error = norm(z1_entropy - z1_entr)
z2_error = norm(z2_entropy - z2_entr)

@test isfinite(z1_error)
@test isfinite(z2_error)

println("Z1 error: $(z1_error)")
println("Z2 error: $(z2_error)")
