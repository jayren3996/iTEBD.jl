include("../src/iTEBD.jl")
using .iTEBD
using LinearAlgebra
using PyCall
using Plots
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
Z1_LEN = length(Z1_TIME)
Z2_LEN = length(Z2_TIME)

#---------------------------------------------------------------------------------------------------
# HAMILTONIAN
#---------------------------------------------------------------------------------------------------
const DT = 0.1
const SUB_DIV = 5
const BOND = 400

const GA, GB = begin
    V0 = 100
    σx = [0.0 0.5; 0.5 0.0]
    pn = [1.0 0.0; 0.0 0.0]
    Xmat = kron(σx, I(2)) + kron(I(2), σx)
    Nmat = kron(pn, pn)
    HPXP = Xmat + V0 * Nmat
    expH = exp(-1im * (DT/SUB_DIV) *HPXP)
    gate(expH, [1,2], bound=BOND), gate(expH, [2,1], bound=BOND)
end

Z1_MPS = product_iMPS(ComplexF64, [[0,1],[0,1]])
Z2_MPS = product_iMPS(ComplexF64, [[0,1],[1,0]])

const Z1_EE = zeros(Z1_LEN)
const Z2_EE = zeros(Z2_LEN)

for i=2:Z1_LEN
    global Z1_MPS, GA, GB, Z1_EE
    for j=1:SUB_DIV
        applygate!(Z1_MPS, GA)
        applygate!(Z1_MPS, GB)
    end
    println("Finish: $((i-1)/(Z1_LEN+Z2_LEN)*50) %.")
    Z1_EE[i] = entropy(Z1_MPS, 1)
end

for i=2:Z2_LEN
    global Z2_MPS, GA, GB, Z2_EE
    for j=1:SUB_DIV
        applygate!(Z2_MPS, GA)
        applygate!(Z2_MPS, GB)
    end
    println("Finish: $(50 + (i-1)/(Z1_LEN+Z2_LEN)*50) %.")
    Z2_EE[i] = entropy(Z2_MPS, 1)
end

#---------------------------------------------------------------------------------------------------
# ERROR RESULTS
#---------------------------------------------------------------------------------------------------
Z1_ERROR = norm(Z1_EE - Z1_ENTR)
Z2_ERROR = norm(Z2_EE - Z2_ENTR)

println("Z1 error: $(Z1_ERROR)")
println("Z2 error: $(Z2_ERROR)")

plot(Z1_TIME, [Z1_EE, Z1_ENTR])
plot(Z2_TIME, [Z2_EE, Z2_ENTR])
