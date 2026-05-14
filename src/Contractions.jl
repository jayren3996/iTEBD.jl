#---------------------------------------------------------------------------------------------------
# Contractions
#---------------------------------------------------------------------------------------------------
function _operator_quadratic_form(
    Γ::AbstractArray{<:Number, 3},
    O::AbstractMatrix{<:Number},
)
    _, p, _ = size(Γ)
    size(O) == (p, p) || throw(DimensionMismatch(
        "operator has size $(size(O)); expected ($p, $p) for grouped physical leg"
    ))

    acc = zero(typeof(conj(zero(eltype(Γ))) * zero(eltype(O)) * zero(eltype(Γ))))
    @inbounds for b in axes(Γ, 3), sout in axes(Γ, 2), sin in axes(Γ, 2)
        coeff = O[sout, sin]
        for a in axes(Γ, 1)
            acc += conj(Γ[a, sout, b]) * coeff * Γ[a, sin, b]
        end
    end
    return acc
end

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
    _operator_quadratic_form(Γ, O)
end
