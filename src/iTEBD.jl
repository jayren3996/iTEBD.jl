module iTEBD
#--- CONSTANT
const BOUND = 50
const SVDTOL = 1e-7
const SQRTTOL = 1e-5
const SORTTOL = 1e-3
const Tensor{T} = Array{T,3} where T<:Number
const TensorArray{T} = Array{Array{T,3}, 1} where T<:Number
const ValuesArray{T} = Vector{Vector{T}} where T<:Number
const GTensor{T} = Array{T,4} where T<:Number
commontype(T...) = promote_type(eltype.(T)...)
#--- Include
using LinearAlgebra
include("Circuits.jl")
include("Canonical.jl")
include("SpinOperators.jl")
include("TransferMatrix.jl")
include("Core.jl")

end # module iTEBD
