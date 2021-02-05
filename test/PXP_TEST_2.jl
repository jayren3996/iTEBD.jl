
using LinearAlgebra
using Printf
using Plots
using iTEBD
using iTEBD: spin
using PyCall
np = pyimport("numpy")
data = np.loadtxt("Ent_Z2.dat")
t = data[:, 1]
s0 = data[:, 2]

const BOND = 400
const DT = 0.1 / 5
const steps = 350


# Z2 state
z2 = begin
    v = [[0, 1], [1,0], [0, 1], [1,0]]
    product_iMPS(ComplexF64, v)
end

const G1, G2, G3, G4 = begin
    sx = spin((2, "1x1"))
    p2 = diagm([0,1,1,1])
    p3 = kron(p2, I(2)) * kron(I(2), p2)
    H = p3 * sx * p3
    expH = exp(-1im * DT * H)
    gate(expH, [1,2,3], bound=BOND), 
    gate(expH, [2,3,4], bound=BOND), 
    gate(expH, [3,4,1], bound=BOND),
    gate(expH, [4,1,2], bound=BOND)
end

const S = zeros(steps)

for i=2:steps
    global DT, z2, G1, G2, G3, G4, S
    for j=1:5
        applygate!(z2, G1)
        applygate!(z2, G2)
        applygate!(z2, G3)
        applygate!(z2, G4)
    end
    EE = entropy(z2, 1)
    @printf("t = %.2f, S = %.5f.\n", 0.1*(i-1), EE)
    S[i] = EE
end


plot(t, [s0, S])
println("Error = $(norm(S-s0))")
