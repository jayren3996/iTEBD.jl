#---------------------------------------------------------------------------------------------------
# Contractions
#---------------------------------------------------------------------------------------------------
"""
    ocontract(Ts, O, λl)

Contraction of:
   ⋅---Ā₁---Ā₂- ⋯ -Āₙ---⋅
   |   |    |      |    |
   |   -------------    |
   λ²  | ...O......|    |
   |   -------------    |
   |   |    |      |    |
   ⋅---A₁---A₂- ⋯ -Aₙ---⋅
"""
function ocontract(
    Ts::AbstractVector{<:AbstractArray{<:Number, 3}},
    O::AbstractMatrix,
    λl::AbstractVector
)
    Γ = tensor_group(Ts)
    tensor_lmul!(λl, Γ)
    Γ2 = tensor_umul(O, Γ)
    dot(Γ, Γ2)
end
