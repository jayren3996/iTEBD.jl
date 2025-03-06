module iTEBD
#---------------------------------------------------------------------------------------------------
# CONSTANTS
#---------------------------------------------------------------------------------------------------
const MAXDIM = 50
const SVDTOL = 1e-12
const SORTTOL = 1e-3
const ZEROTOL = 1e-20
const KRLOV_POWER = 100
#---------------------------------------------------------------------------------------------------
# INCLUDE
#---------------------------------------------------------------------------------------------------
using LinearAlgebra, SparseArrays, TensorOperations, KrylovKit
import Base: eltype, getindex, setindex!
import LinearAlgebra: conj

include("TensorAlgebra.jl")
include("Contractions.jl")
include("iMPS.jl")
include("Gate.jl")
include("Schmidt.jl")
include("Block.jl")
include("Krylov.jl")
include("Miscellaneous.jl")


end
