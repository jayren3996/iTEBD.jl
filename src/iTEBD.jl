module iTEBD
#--- Import
using LinearAlgebra
using TensorOperations
include("Gate.jl")
include("Canonical.jl")
include("SpinOperators.jl")
include("TransferMatrix.jl")
using .Gate: applygate
using .Canonical: canonical
using .SpinOperators: spinop
using .TransferMatrix: trm, gtrm, utrm, normalization, inner, symrep
using .TransferMatrix: dominent_eigen, dominent_eigen!
using .TransferMatrix: dominent_eigval, dominent_eigval!
#--- Export
export applygate, canonical, spinop
export TEBD
export trm, gtrm, utrm, normalization, inner, symrep
export dominent_eigen, dominent_eigen!, dominent_eigval, dominent_eigval!
#--- CONSTANT
const BOUND = 50
const TOL = 1e-7
#--- TEBD type
mutable struct TEBD{T}
    gate::Array{T,4}
    dt::Float64
    N::Int64
    bound::Int64
    tol::Float64
end
function TEBD(H, dt; mode::String="r", bound=BOUND, tol=TOL)
    if mode == "r" || mode == "real"
        expH = exp(-1im * dt * H)
    elseif mode == "i" || mode == "imag"
        expH = exp(-dt * H)
    end
    d = Int(sqrt(size(H,1)))
    gate = reshape(expH,d,d,d,d)
    TEBD(gate, dt, 0, bound, tol)
end
#--- Run TEBD
function (tebd::TEBD)(mps::Tuple)
    A,B,λ1,λ2 = mps
    gate = tebd.gate
    bound = tebd.bound
    tol = tebd.tol

    A,λ1,B = applygate(gate,A,B,λ2, bound=bound,tol=tol)
    B,λ2,A = applygate(gate,B,A,λ1, bound=bound,tol=tol)
    tebd.N += 1
    A,B,λ1,λ2
end
function (tebd::TEBD)(mps::Tuple, n::Int64)
    A,B,λ1,λ2 = mps
    for i = 1:n
        A,B,λ1,λ2 = tebd((A,B,λ1,λ2))
    end
    A,B,λ1,λ2
end

end # module iTEBD
