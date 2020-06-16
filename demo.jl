using iTEBD
# Create random iMPS
dim_num = 50
random_tensor = rand(dim_num, 3, dim_num)
tensors = [random_tensor, random_tensor]
values = [rand(dim_num), rand(dim_num)]
# Create AKLT Hamiltonian
SS = spinop("xx",1) + spinop("yy",1) + spinop("zz",1)
hamiltonian = SS + 1/3 * SS^2
# Setup TEBD
time_steps = 0.01
tebd_system = TEBD(hamiltonian, time_steps, mode="i")
for i=1:2000
    global tensors, values
    tensors, values = tebd_system(tensors, values)
    println("T = ", tebd_system.N * tebd_system.dt)
end
println("Energy per site: ", energy(hamiltonian, tensors...))
