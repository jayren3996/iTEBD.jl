include("../src/iTEBD.jl")
using .iTEBD
using LinearAlgebra
using PyCall
np = pyimport("numpy")
#---------------------------------------------------------------------------------------------------
# LOAD DATA
#---------------------------------------------------------------------------------------------------
Z1_DATA = np.loadtxt("Ent_Z1.dat")
Z2_DATA = np.loadtxt("Ent_Z2.dat")
Z1_TIME = Z1_DATA[:, 1]
Z2_TIME = Z2_DATA[:, 1]
Z1_ENTR = Z1_DATA[:, 2]
Z2_ENTR = Z2_DATA[:, 2]

#---------------------------------------------------------------------------------------------------
# HAMILTONIAN
#---------------------------------------------------------------------------------------------------
HPXP = begin
    V0 = 100
    σx = [0.0 1.0; 1.0 0.0]
    pn = [1.0 0.0; 0.0 0.0]
    Xmat = (kron(σx, I(2)) + kron(I(2), σx)) / 2
    Nmat = kron(pn, pn)
    Xmat + V0 * Nmat
end

pxp_evolving = itebd(HPXP, 0.1, bound=400)

Z1_MPS = begin
    A = zeros(1, 2, 1)
    A[1, 2, 1] = 1
    iMPS([A, A])
end

Z2_MPS = begin
    A = zeros(1, 2, 1)
    B = zeros(1, 2, 1)
    A[1, 1, 1] = 1
    B[1, 2, 1] = 1
    iMPS([A, B])
end

Z1_EE = zeros(length(Z1_TIME))
Z2_EE = zeros(length(Z2_TIME))

for i=2:length(Z1_TIME)
    global Z1_MPS, Z1_EE
    Z1_MPS = pxp_evolving(Z1_MPS)
    Z1_EE[i] = entropy(Z1_MPS, 1)
end

for i=2:length(Z2_TIME)
    global Z2_MPS, Z2_EE
    Z2_MPS = pxp_evolving(Z2_MPS)
    Z2_EE[i] = entropy(Z2_MPS, 1)
end

#---------------------------------------------------------------------------------------------------
# ERROR RESULTS
#---------------------------------------------------------------------------------------------------
Z1_ERROR = norm(Z1_EE - Z1_ENTR)
Z2_ERROR = norm(Z2_EE - Z2_ENTR)

println("Z1 error: $(Z1_ERROR)")
println("Z2 error: $(Z2_ERROR)")
