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
    M = otrm(Γ, O, Γ)
    vl = reshape(Diagonal(λl .^2), :)
    vr = reshape(I(size(Γ, 3)), :)
    dot(vl, M * vr)
end
