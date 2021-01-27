#using iTEBD
include("../src/iTEBD.jl")
using .iTEBD
using LinearAlgebra
using Plots

time_step = 0.005
iteration = 125
measure_point = time_step * (1:iteration)
Ω  =   2π * 4.2
V₀ =   2π * 51
Δ₀ = 0.55 * Ω
Δₘ = 0.55 * Ω
ωₘ = 1.225 * Ω

Nmat = [1.0 0.0; 0.0 0.0]
Nmat_2 = ( kron(Nmat, I(2)) + kron(I(2), Nmat) ) / 2

Xmat = [0.0 1.0; 1.0 0.0]
Xmat_2 = ( kron(Xmat, I(2)) + kron(I(2), Xmat) ) / 2

Hpxp = Ω/2 * Xmat_2 + V₀ * kron(Nmat, Nmat)

function td_system(t::Float64)
    Δₜ = Δ₀ + Δₘ * cos(ωₘ * t)
    H = Hpxp - 0.5 * Δₜ * Nmat_2
    itebd(H, time_step, bound=400)
end

pxp_evolving = itebd(Hpxp, time_step, bound=400)

Z2_MPS = begin
    A = zeros(1, 2, 1)
    B = zeros(1, 2, 1)
    A[1, 1, 1] = 1
    B[1, 2, 1] = 1
    iMPS([A, B])
end

z2_entropy = zeros(iteration)
z2_driven_entropy = zeros(iteration)

z2_mps = deepcopy(Z2_MPS)
z2_driven_mps = deepcopy(Z2_MPS)
for i=1:iteration
    expht = td_system(measure_point[i])

    z2_mps = pxp_evolving(z2_mps)
    z2_driven_mps = expht(z2_driven_mps)

    z2_entropy[i] = entropy(z2_mps, 1)
    z2_driven_entropy[i] = entropy(z2_driven_mps, 1)
end

data = [z2_entropy z2_driven_entropy]
labels = ["AFM" "AFM driven"]

plot(measure_point, data, label=labels)
