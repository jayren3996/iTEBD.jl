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
Tuple{Array{T,3},Array{T,3},Diagonal{T,Array{T,1}},Diagonal{T,Array{T,1}}}
```

i.e. ```imps = (A,B,λ1,λ2)```. We can constructed an iMPS by simply pack two 3-D array and two Diagonal matrix. For example, a random state can be constructed by:

```julia
dim_num = 50
random_tensor = rand(dim_num, 3, dim_num)
mps = (random_tensor, random_tensor, Diagonal(ones(dim_num)), Diagonal(ones(dim_num)))
```

### Hamiltonian

An Hamiltonian is just an  ```Array{T,2}```. While for ground state searching problems, we will multiply the matrices by an imaginary part ```-1im```. For example, AKLT Hamiltonian (for imaginary time evolution) can be constructed by:

```julia
Sx = [0 1 0;1 0 1;0 1 0]/sqrt(2)
Sy = [0 1im 0;-1im 0 1im;0 -1im 0]/sqrt(2)
Sz = Diagonal([-1,0,1])
SS = kron(Sx, Sx) + kron(Sy, Sy) + kron(Sz, Sz)
H2 = SS + 1/3 * SS^2
hamiltonian = -1im * H2
```

### Setup and run iTEBD

After obtaining iMPS and Hamiltonian matrix, we can then use ```tebd``` function to construct iTEBD system:

```julia
tebd_system = tebd(mps,hamiltonian,bound=50,tol=1e-7,N=100)
```

Note that there are 3 optional input the the constructor, where ```bound``` controls the SVD truncation, ```tol``` control the threshold for Schmidt, and ```N``` controls the steps after which to canonicalize the iMPS.

For reference, the ```TEBD``` object is defined as:

```julia
mutable struct TEBD
    mps     # iMPS state
    gate    # Quantum gate that generate time revolution
    dt      # Time steps
    n       # Number of times
    N       # Period of canonicalize
    bound   # Truncation
    tol     # Schmidt value threshold
end
```

We can now run the simulation by iteratively appling ```run!``` to the system:

```julia
time_steps
for i=1:1000
		run!(tebd_system)
		println("T = ", tebd_system.n * tebd_system.dt)
end
```

The result is stored in ```tebd_system.mps``` . The procedure for real-time evolution is similar.

### Canonical form

In many cases, it is much simpler to work on the canonical form of MPS. Here, the canonical form is for the Schmidt canonical form, introduced by G. Vidal (G. Vidal, Phys. Rev. B **78**, 155117 􏱋2008􏱌).

There are 3 method for the function ```canonical``` :

```julia
canonical(A::Array{T,3}; check=true)
canonical(A::Array{T,3},B::Array{T,3}; check=true)
canonical(A::Array{T,3},B::Array{T,3},λ1::Diagonal{T,Array{T,1}},λ2::Diagonal{T,Array{T,1}}; check=true)
```

when ```check=true``` , it will check whether the dominent eigenvalue of the transfer matrix is degenerate. If so, it means the iMPS is in a superposional "cat state", i.e. the boundary can determine the quantum state. What  a checked canonical function does is that it will choose a random boundary state to get rid of of superposition. And if the system is gapped, different choices of boundary should not influence the bulk state. 