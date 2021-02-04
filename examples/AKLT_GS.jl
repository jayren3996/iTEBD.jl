using iTEBD
using iTEBD: spin

# Create random iMPS
BOND = 50
imps = rand_iMPS(ComplexF64, 2, 3, BOND)

# Create AKLT Hamiltonian and iTEBD engine
GA, GB = begin
    dt = 0.01
    ss = spin((1,"xx"), (1,"yy"), (1,"zz"), D=3)
    H = ss + 1/3*ss^2
    expH = exp(- dt * H)
    gate(expH, [1,2], bound=BOND), gate(expH, [2,1], bound=BOND)
end

# Exact AKLT ground state
aklt = begin
    aklt_tensor = zeros(2,3,2)
    aklt_tensor[1,1,2] = +sqrt(2/3)
    aklt_tensor[1,2,1] = -sqrt(1/3)
    aklt_tensor[2,2,2] = +sqrt(1/3)
    aklt_tensor[2,3,1] = -sqrt(2/3)
    aklt_tensor
    iMPS([aklt_tensor, aklt_tensor])
end

# Setup TEBD
for i=1:2000
    global imps, aklt, GA, GB
    applygate!(imps, GA)
    applygate!(imps, GB)
    if mod(i, 100) == 0
        println("Overlap: ", inner_product(aklt, imps))
    end
end
