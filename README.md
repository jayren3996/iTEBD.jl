# iTEBD.jl
Julia package for simple iTEBD calculation.

## Introduction

This julia module is for 2-site infinite time-evolution block-decimation (iTEBD) algorithm based on 2-site infinite matrix-product-state (iMPS).

The iTEBD algorithm relies on a Trotter-Suzuki and subsequent approximation of the time-evolution operator. It provides an extremely efficient method to study both the time evolution and the ground state (using imaginary time evolution) of 1-D gapped system.

Currently, this package support only 2-site gates. i.e. only nearest-neighbor Hamiltonians are supported.

## Installation
```julia
pkg> add https://github.com/jayren3996/iTEBD.jl
```

## Code Examples

### iMPS objects

The 2-site iMPS object are represented by:

```julia
Tuple{Array{T,3},Array{T,3},Array{T,1},T,Array{T,1}}
```

i.e. ```imps = (A,B,λ1,λ2)```. When brought to canonical form, ```A```, ```B``` are right-canonical, and ```λ1```, ```λ2``` are the Schmidt values between neighboring sites. We can constructed an iMPS by simply pack two 3-D array and two vector. For example, a random state can be constructed by:

```julia
dim_num = 50
random_tensor = rand(dim_num, 3, dim_num)
mps = (random_tensor, random_tensor, ones(dim_num), ones(dim_num))
```

### Hamiltonian

An Hamiltonian is just an  ```Array{T,2}```. There is also a helper function ```spinop``` for constructing spin Hamiltonian. For example, AKLT Hamiltoniancan be constructed by:

```julia
SS = spinop("xx",S=1) + spinop("yy",S=1) + spinop("zz",S=1)
hamiltonian = SS + 1/3 * SS^2
```

### Setup and run iTEBD

After obtaining iMPS and Hamiltonian matrix, we can then use ```TEBD``` function to construct iTEBD system:

```julia
time_steps = 0.1
tebd_system = TEBD(hamiltonian, time_steps, mode="i", bound=50, tol=1e-7)
```

Note that there are 3 optional input the the constructor, where ```mode ``` determine whether the evolution is in real-time (```mode="r"```) or imaginary time (```mode="i"```). ```bound``` controls the SVD truncation, and ```tol``` control the threshold for Schmidt values.

For reference, the ```TEBD``` object is defined as:

```julia
mutable struct TEBD{T}
    gate::Array{T,4}
    dt::Float64
    N::Int64
    bound::Int64
    tol::Float64
end
```

We can now apply the ```TEBD``` object to mps to update the system:

```julia
for i=1:1000
    mps = tebd_system(mps)
    println("T = ", tebd_system.N * tebd_system.dt)
end
```

The procedure for real-time evolution is the same, but just set ```mode="r"```.

### Canonical form

In many cases, it is much simpler to work on the canonical form of MPS. Here, the canonical form is the left-canonical form. However, we keep track of the Schmidt values (singular values) so that it can easily transformed to Schmidt canonical form, introduced by G. Vidal (G. Vidal, Phys. Rev. B **78**, 155117 􏱋2008􏱌). 

There are 2 method for the function ```canonical``` :

```julia
canonical(A::Array{T,3}; check=true)
canonical(A::Array{T,3},B::Array{T,3}; check=true)
```

when ```check=true``` , it will check whether the dominent eigenvalue of the transfer matrix is degenerate. If so, it means the iMPS is in a superposional "cat state", i.e. the MPS state can be reduced to smaller dimension by specifying boundary states. 

What  a checked canonical function does is that it will choose a boundary state (default to be equally superpositional state) to reduced the MPS dimension. And if the system is gapped, different choices of boundary should not influence the bulk state. 