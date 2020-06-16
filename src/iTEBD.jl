module iTEBD
#--- CONSTANT
const BOUND = 50
const SVDTOL = 1e-7
const SQRTTOL = 1e-5
const SORTTOL = 1e-3
#--- Import
using LinearAlgebra
using TensorOperations
#--- Include
include("Circuits.jl")
include("Canonical.jl")
include("SpinOperators.jl")
include("TransferMatrix.jl")
using .SpinOperators: spinop
#--- Export
export applygate, canonical, spinop
export TEBD, energy
export trm, gtrm, utrm, normalization, inner, symrep
export dominent_eigen, dominent_eigen!, dominent_eigval, dominent_eigval!
#--- Helper functions
commontype(T...) = promote_type(eltype.(T)...)
pindex(L,i) = begin
    i = mod(i-1, length(L)) + 1
    vcat(L[i:end],L[1:i-1])
end
#--- TEBD type
mutable struct TEBD{T}
    site::Int64
    gate::Array{T,2}
    dt::Float64
    N::Int64
    bound::Int64
    tol::Float64
end
function TEBD(H, dt; site=2, mode::String="r", bound=BOUND, tol=SVDTOL)
    if mode == "r" || mode == "real"
        expH = exp(-1im * dt * H)
    elseif mode == "i" || mode == "imag"
        expH = exp(-dt * H)
    end
    TEBD(site, expH, dt, 0, bound, tol)
end
#--- Run TEBD
function (tebd::TEBD)(tensors, schmidtvals)
    gate = tebd.gate
    bound = tebd.bound
    tol = tebd.tol
    T,V = applygate!(gate,schmidtvals[end],tensors..., bound=bound, tol=tol)
    for i=2:tebd.site
        T,V = applygate!(gate,V[1],pindex(T,2)..., bound=bound,tol=tol)
    end
    tebd.N += 1
    pindex(T,2), pindex(V,2)
end
function (tebd::TEBD)(tensors, schmidtvals, n)
    for i = 1:n
        tensors, schmidtvals = tebd(tensors, schmidtvals)
    end
    tensors, schmidtvals
end

end # module iTEBD
