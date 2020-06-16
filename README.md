# iTEBD.jl
Julia package for simple iTEBD calculation.

## Introduction

This julia module is for infinite time-evolution block-decimation (iTEBD) algorithm based on infinite matrix-product-state (iMPS).

The iTEBD algorithm relies on a Trotter-Suzuki and subsequent approximation of the time-evolution operator. It provides an extremely efficient method to study both the time evolution and the ground state (using imaginary time evolution) of 1-D gapped system.

## Installation
```julia
pkg> add https://github.com/jayren3996/iTEBD.jl
```

## Code Examples

### iMPS objects

The 2-site iMPS object are represented by a tensor list and a value list:

```julia
tensors = Array{Array{T,3},1}
values = Array{Array{T,1},1}
```

When brought to canonical form, each ```tensors[i]``` is right-canonical, and each ```values[i]``` is the Schmidt values between neighboring sites. We can constructed an iMPS by simply pack two 3-D array and two vector. For example, a random state can be constructed by:

```julia
dim_num = 50
tensors = [rand(dim_num, 3, dim_num), rand(dim_num, 3, dim_num)]
values = [rand(dim_num), rand(dim_num)]
```

### Hamiltonian

An Hamiltonian is just an  ```Array{T,2}```. There is also a helper function ```spinop``` for constructing spin Hamiltonian. For example, AKLT Hamiltoniancan be constructed by:

```julia
SS = spinop("xx",1) + spinop("yy",1) + spinop("zz",1)
hamiltonian = SS + 1/3 * SS^2
```

### Setup and run iTEBD

After obtaining iMPS and Hamiltonian matrix, we can then use ```TEBD``` function to construct iTEBD system:

```julia
time_steps = 0.01
tebd_system = TEBD(hamiltonian, time_steps, site=2, mode="i", bound=50, tol=1e-7)
```

Note that there are 4 optional input the the constructor, where ```site``` determines the number of sites in a unit-cell. ```mode ``` determine whether the evolution is in real-time (```mode="r"```) or imaginary time (```mode="i"```). ```bound``` controls the SVD truncation, and ```tol``` control the threshold for Schmidt values.

For reference, the ```TEBD``` object is defined as:

```julia
mutable struct TEBD{T}
    site::Int64
    gate::Array{T,2}
    dt::Float64
    N::Int64
    bound::Int64
    tol::Float64
end
```

We can now apply the ```TEBD``` object to mps to update the system:

```julia
for i=1:1000
    global tensors, values
    tensors, values = tebd_system(tensors, values)
    println("T = ", tebd_system.N * tebd_system.dt)
end
println("Energy per site: ", energy(hamiltonian, tensors...))
```

The ground state energy ```-0.6666666``` would be printed.

The procedure for real-time evolution is the same, but just set ```mode="r"```.

### Canonical form

In many cases, it is much simpler to work on the canonical form of MPS. Here, the canonical form is the left-canonical form. However, we keep track of the Schmidt values (singular values) so that it can easily transformed to Schmidt canonical form, introduced by G. Vidal (G. Vidal, Phys. Rev. B **78**, 155117 􏱋2008􏱌). 

There are 2 method for the function ```canonical``` :

```julia
canonical(A::Array{T,3}; check=true)
canonical(A::Array{T,3},B::Array{T,3}; check=true)
```

The canonical algorithm given by (Vidal, 2008) requires the dominent eigenvalue of the transfer matrix non-degenerate. While in ```check=true``` mode, it will choose a boundary state (default to be equally-superpositional state) to make sure the algorithm works. However, no guarantee that the canonicalized MPS is of the simpleast form (the dimension of MPS may be further reduced). 