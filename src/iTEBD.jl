module iTEBD
#---------------------------------------------------------------------------------------------------
# CONSTANTS
#---------------------------------------------------------------------------------------------------
const BOUND = 50
const SVDTOL = 1e-7
const SORTTOL = 1e-5
#---------------------------------------------------------------------------------------------------
# INCLUDE
#---------------------------------------------------------------------------------------------------
using LinearAlgebra
using SparseArrays
using TensorOperations
using KrylovKit
import LinearAlgebra: conj

include("TensorAlgebra.jl")
include("MPS.jl")
include("Core.jl")
include("Canonical.jl")
include("Miscellaneous.jl")

end # module iTEBD
