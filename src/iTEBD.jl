module iTEBD
#---------------------------------------------------------------------------------------------------
# CONSTANTS
#---------------------------------------------------------------------------------------------------
const MAXDIM::Int = 50
const SVDTOL::Float64 = 1e-12
const SORTTOL::Float64 = 1e-3
const ZEROTOL::Float64 = 1e-20
export MAXDIM, SVDTOL, DenseIMPS, expect, ent_S
#---------------------------------------------------------------------------------------------------
# INCLUDE
#---------------------------------------------------------------------------------------------------
using LinearAlgebra, TensorOperations, KrylovKit
using ITensors, ITensorMPS
import Base: eltype, getindex, setindex!
import LinearAlgebra: conj

include("TensorAlgebra.jl")
include("Contractions.jl")
include("iMPS.jl")
include("ITensorsInterop.jl")
include("Gate.jl")
include("Schmidt.jl")
include("Krylov.jl")
include("Miscellaneous.jl")
include("ScarFinder.jl")
include("SymmetricStubs.jl")

#---------------------------------------------------------------------------------------------------
# PRECOMPILE WORKLOAD
#---------------------------------------------------------------------------------------------------
using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    X = ComplexF64[0 1; 1 0]
    G = kron(X, X)
    @compile_workload begin
        ψ = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
        applygate!(ψ, G, 1, 2; maxdim=4)
        evolve!(ψ, [(G, 1, 2), (G, 2, 1)], 3; maxdim=4)
        expect(ψ, G, 1, 2)
        ent_S(ψ, 1)
    end
end

end
