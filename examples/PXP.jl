#using iTEBD
include("../src/iTEBD.jl")
using .iTEBD
using LinearAlgebra
using Plots

const time_step = 0.001
const iteration = 1000
const measure_point = time_step * (1:iteration)
const Ω = 2π * 4.2
const V₀ = 2π * 51
const Δ₀ = 0.55 * Ω
const Δₘ = 0.55 * Ω
const ωₘ = 1.2 * Ω

const Nmat = [1.0 0.0; 0.0 0.0]
const Nmat_2 = kron(Nmat, I(2)) + kron(I(2), Nmat)

const Xmat = [0.0 1.0; 1.0 0.0]
const Xmat_2 = kron(Xmat, I(2)) + kron(I(2), Xmat)

const Hpxp = Ω/2 * Xmat_2 + V₀ * kron(Nmat, Nmat)

function td_system(t::Float64)
    Δₜ = Δ₀ - Δₘ * cos(ωₘ * t)
    H = Hpxp - Δₜ * Nmat_2
    itebd(H, time_step, bound=400)
end

const pxp_evolving = itebd(Hpxp, time_step, bound=400)

const Z2_MPS = begin
    A = zeros(1, 2, 1)
    B = zeros(1, 2, 1)
    A[1, 1, 1] = 1
    B[1, 2, 1] = 1
    iMPS([A, B])
end

const FM_MPS = begin
    A = zeros(1, 2, 1)
    A[1, 2, 1] = 1
    iMPS([A, A])
end


z2_mps = deepcopy(Z2_MPS)
#fm_mps = deepcopy(FM_MPS)

z2_driven_mps = deepcopy(Z2_MPS)
#fm_driven_mps = deepcopy(FM_MPS)

const z2_entropy = zeros(iteration)
#const fm_entropy = zeros(iteration)
const z2_driven_entropy = zeros(iteration)
#const fm_driven_entropy = zeros(iteration)

for i=1:iteration
    global pxp_evolving, z2_mps, fm_mps, z2_driven_mps, fm_driven_mps

    expht = td_system(measure_point[i])

    z2_mps = pxp_evolving(z2_mps)
    #fm_mps = pxp_evolving(fm_mps)

    z2_driven_mps = expht(z2_driven_mps)
    #fm_driven_mps = expht(fm_driven_mps)

    z2_entropy[i] = entropy(z2_mps, 1)
    #fm_entropy[i] = entropy(fm_mps, 1)

    z2_driven_entropy[i] = entropy(z2_driven_mps, 1)
    #fm_driven_entropy[i] = entropy(fm_driven_mps, 1)
end

data = [z2_entropy z2_driven_entropy]
labels = ["Z₂" "driven"]

plot(measure_point, data, label = labels)
