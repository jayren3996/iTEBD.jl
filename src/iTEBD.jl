module iTEBD
#--- Import
using LinearAlgebra
using TensorOperations
include("Canonical.jl")
include("SpinOperators.jl")
using .Canonical: transfermat, dominentvector
using .Canonical: canonical, applygate, overlap
using .SpinOperators
#--- Export
export transfermat, dominentvector
export canonical, applygate, overlap
export spinop
export TEBD
#--- CONSTANT
const BOUND = 50
const TOL = 1e-7
#--- TEBD
mutable struct TEBD{T}
    gate::Array{T,4}
    dt::Float64
    N::Int64
    bound::Int64
    tol::Float64
end

function TEBD(
    H::Matrix, dt::Float64;
    mode::String="real", bound::Int64=BOUND, tol::Float64=TOL)

    if mode == "real" || mode == "r"
        expH = exp(-1im * dt * H)
    elseif mode == "imag" || mode == "i" || mode == "imaginary"
        expH = exp(-dt * H)
    end
    d = Int(sqrt(size(H,1)))
    gate = reshape(expH,d,d,d,d)
    TEBD(gate, dt, 0, bound, tol)
end

function (tebd::TEBD)(mps)
    A,B,λ1,λ2 = mps
    gate = tebd.gate
    bound = tebd.bound
    tol = tebd.tol

    A,λ1,B = applygate(gate,A,B,λ2, bound=bound,tol=tol)
    B,λ2,A = applygate(gate,B,A,λ1, bound=bound,tol=tol)
    tebd.N += 1
    A,B,λ1,λ2
end

end # module
