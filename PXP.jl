using LinearAlgebra
using Plots
using iTEBD
import iTEBD: spinmat


# Pure PXP dynamics

const time_step = 0.01
const iteration = 3000
const measure_point = time_step * (1:iteration)

const HPXP = begin
    pmat = [1.0 0.0; 0.0 0.0]
    xmat = [0.0 1.0; 1.0 0.0]
    kron(pmat, xmat, pmat)
end

const pxp_evolving = itebd(HPXP, time_step)
const entropy_data = zeros(iteration)

Z2MPS = begin
    A = zeros(1, 2, 1)
    B = zeros(1, 2, 1)
    C = zeros(1, 2, 1)
    A[1, 1, 1] = 1
    B[1, 2, 1] = 1
    C[1, 1, 1] = 1
    iMPS([A, B, C])
end

for i=1:iteration
    global pxp_evolving, Z2MPS
    Z2MPS = pxp_evolving(Z2MPS)
    entropy_data[i] = entanglement(Z2MPS, 1)
end

plot(measure_point, entropy_data)