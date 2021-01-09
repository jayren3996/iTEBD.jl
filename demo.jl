using iTEBD

# Create random iMPS
imps = begin
    dim_num = 50
    rand_iMPS(2, 3, dim_num)
end

# Create AKLT Hamiltonian and iTEBD engine
hamiltonian = begin
    SS = spinop("xx",1) + spinop("yy",1) + spinop("zz",1)
    SS + 1/3 * SS^2
end 
engine = begin
    time_steps = 0.01
    itebd(hamiltonian, time_steps, mode="i")
end

# Setup TEBD
for i=1:2000
    global imps
    imps = eigen(imps)
    println("Energy per site: ", expectation(imps, hamiltonian))
end


