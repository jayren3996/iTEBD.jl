module iTEBD
#---------------------------------------------------------------------------------------------------
# CONSTANTS
#---------------------------------------------------------------------------------------------------
const MAXDIM::Int = 50
const SVDTOL::Float64 = 1e-12
const SORTTOL::Float64 = 1e-3
const ZEROTOL::Float64 = 1e-20
export MAXDIM, SVDTOL, expect, ent_S
#---------------------------------------------------------------------------------------------------
# INCLUDE
#---------------------------------------------------------------------------------------------------
using LinearAlgebra, SparseArrays, TensorOperations, KrylovKit
using ITensors, ITensorMPS
import Base: eltype, getindex, setindex!
import LinearAlgebra: conj

include("TensorAlgebra.jl")
include("Contractions.jl")
include("iMPS.jl")
include("ITensors.jl")
include("Gate.jl")
include("Schmidt.jl")
include("Krylov.jl")
include("Miscellaneous.jl")
include("ScarFinder.jl")

end
