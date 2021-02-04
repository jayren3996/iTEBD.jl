using LinearAlgebra
using iTEBD
using iTEBD: spin

const BOND = 100
const DT = 0.1
const steps = 150
# Z2 state
z2 = begin
    v = [[1,0], [0, 1]]
    product_iMPS(ComplexF64, v)
end

const GA, GB = begin
    sx = spin((2, "x"))
    expH = exp(-1im * DT * sx)
    gate(expH, [1]), gate(expH, [2])
end

const PA, PB = begin
    p = diagm([0, 1, 1, 1])
    gate(p, [1,2], bound=BOND), gate(p, [2,1], bound=BOND)
end

const S = zeros(steps)

for i=1:10
    global DT, z2, GA, GB, PA, PB, S
    applygate!(z2, GA)
    applygate!(z2, GB)
    applygate!(z2, PA)
    applygate!(z2, PB)
    #z2 = canonical(z2)
    EE = entropy(z2, 1)
    println("t = $(DT*i), S = $EE.")
    S[i] = EE
end