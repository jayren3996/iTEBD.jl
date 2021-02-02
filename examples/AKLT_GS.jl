using iTEBD
using LinearAlgebra
using iTEBD: spinmat

# Create random iMPS
imps = begin
    dim_num = 50
    rand_iMPS(2, 3, dim_num)
end

# Create AKLT Hamiltonian and iTEBD engine
hamiltonian = begin
    SS = spinmat("xx", 3) + spinmat("yy", 3) + spinmat("zz", 3)
    SS + 1/3 * SS^2 + 2/3 * I(9)
end

engine = begin
    time_step = 0.01
    itebd(hamiltonian, time_step, mode="i")
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
    global imps, aklt, engine
    imps = engine(imps)
    if mod(i, 100) == 0
        println("Overlap: ", inner_product(aklt, imps))
    end
end
