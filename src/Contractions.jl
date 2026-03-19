#---------------------------------------------------------------------------------------------------
# Contractions
#---------------------------------------------------------------------------------------------------
"""
    ocontract(Ts, O, λl)

Contract a contiguous block of stored local tensors with a dense local operator.

Parameters:
- `Ts`
  Vector of stored local tensors representing the support of the operator.
- `O`
  Dense local operator acting on the fused physical Hilbert space of that
  support.
- `λl`
  Schmidt values on the bond immediately to the left of the block.

Returns:
- Scalar contraction value, typically interpreted as a local expectation value
  before taking the real part.

Notes:
- This helper is used internally by [`expect`](@ref).
- The tensors in `Ts` are assumed to follow the package storage convention
  `B_i = Γ_i λ_i`.
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
