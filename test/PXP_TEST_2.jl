using DelimitedFiles
using LinearAlgebra
using Printf
using Test
using iTEBD

const SMOKE_MODE = "--smoke" in ARGS
const BOND = SMOKE_MODE ? 16 : 400
const DT = 0.1 / (SMOKE_MODE ? 1 : 5)
const SUB_DIV = SMOKE_MODE ? 1 : 5

function _reference_data()
    path = joinpath(dirname(@__DIR__), "examples", "Ent_Z2.dat")
    data = readdlm(path)
    if SMOKE_MODE
        data = data[1:min(4, size(data, 1)), :]
    end
    return data[:, 1], data[:, 2]
end

function _pxp_gate()
    p0 = [0.0 0.0; 0.0 1.0]
    x = [0.0 1.0; 1.0 0.0]
    h_pxp = kron(p0, x, p0)
    return exp(-1im * DT * h_pxp)
end

function _pxp_step!(psi::iMPS, gate::AbstractMatrix)
    applygate!(psi, gate, 1, 3; maxdim=BOND)
    applygate!(psi, gate, 2, 4; maxdim=BOND)
    applygate!(psi, gate, 3, 1; maxdim=BOND)
    applygate!(psi, gate, 4, 2; maxdim=BOND)
    return psi
end

t, reference_entropy = _reference_data()
entropy = zeros(Float64, length(reference_entropy))
gate = _pxp_gate()

z2 = product_iMPS(ComplexF64, [[0, 1], [1, 0], [0, 1], [1, 0]])

for i in 2:length(reference_entropy)
    for _ in 1:SUB_DIV
        _pxp_step!(z2, gate)
    end
    entropy[i] = iTEBD.ent_S(z2, 1)
    @printf("t = %.2f, S = %.5f.\n", t[i], entropy[i])
end

error = norm(entropy - reference_entropy)
@test isfinite(error)
println("Error = $(error)")
