module iTEBD
#--- Import
using LinearAlgebra
using TensorOperations
include("Canonical.jl")
using .Canonical: transfermat, tensorsplit, canonical
#--- Export
export transfermat, tensorsplit, canonical
export applygate
export tebd, run!
#--- CONSTANT
const BOUND = 50
const TOL = 1e-7
#--- Apply Gates
function applygate(G,A,B,λ1,λ2; bound=BOUND, tol=TOL)
    i1,i2 = size(A)[1:2]
    i3,i4 = size(B)[2:3]
    block = Array{promote_type(eltype(A),eltype(G))}(undef,i1,i2,i3,i4)
    @tensor block[α,β,γ,τ] = λ2[α,1]*A[1,2,3]*λ1[3,4]*B[4,5,6]*λ2[6,τ]*G[β,γ,2,5]
    U,λ1p,V = tensorsplit(block,bound=bound,tol=tol,renormalize=true)
    len = size(λ1p,1)
    λ2i = inv(λ2)
    Ap = Array{promote_type(eltype(U),eltype(λ2i))}(undef,i1,i2,len)
    Bp = Array{promote_type(eltype(V),eltype(λ2i))}(undef,len,i3,i4)
    @tensor Ap[i,j,k] = λ2i[i,1]*U[1,j,k]
    @tensor Bp[i,j,k] = V[i,j,1]*λ2i[1,k]
    Ap, Bp, λ1p, λ2
end
#--- TEBD
mutable struct TEBD
    mps     # iMPS state
    gate    # Quantum gate that generate time revolution
    dt      # Time steps
    n       # Number of times
    N       # Period of canonicalize
    bound   # Truncation
    tol     # Schmidt value threshold
end

function tebd(mps,H,dt; bound=BOUND,tol=TOL,N=0)
    expH = exp(-1im * dt * H)
    d = Int(sqrt(size(H,1)))
    gate = reshape(expH,d,d,d,d)
    TEBD(mps,gate,dt,0,N,bound,tol)
end

function run!(tebd::TEBD)
    A,B,λ1,λ2 = tebd.mps
    gate = tebd.gate
    A,B,λ1,λ2 = applygate(gate,A,B,λ1,λ2, bound=tebd.bound,tol=tebd.tol)
    B,A,λ2,λ1 = applygate(gate,B,A,λ2,λ1, bound=tebd.bound,tol=tebd.tol)
    if tebd.N>0 && tebd.n % tebd.N == 0
        A,B,λ1,λ2 = canonical(A,B,λ1,λ2)
    end
    tebd.mps = (A,B,λ1,λ2)
    tebd.n += 1
end
end # module
