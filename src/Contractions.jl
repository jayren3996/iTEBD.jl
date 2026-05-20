#---------------------------------------------------------------------------------------------------
# Contractions
#---------------------------------------------------------------------------------------------------
function _operator_quadratic_form(
    Γ::AbstractArray{<:Number, 3},
    O::AbstractMatrix{<:Number},
)
    Dl, p, Dr = size(Γ)
    size(O) == (p, p) || throw(DimensionMismatch(
        "operator has size $(size(O)); expected ($p, $p) for grouped physical leg"
    ))

    # acc = ∑_{a,b,sout,sin} conj(Γ[a, sout, b]) · O[sout, sin] · Γ[a, sin, b].
    # Slice Γ by the slow `b` index and BLAS-contract O on the physical leg of
    # each (Dl, p) slice — one small preallocated work buffer is reused across
    # all Dr slices, keeping allocations O(Dl·p) rather than O(Dl·p·Dr).
    T = promote_type(eltype(Γ), eltype(O))
    work = Matrix{T}(undef, Dl, p)
    Ot = transpose(O)
    acc = zero(T)
    @inbounds for b in axes(Γ, 3)
        Γ_b = @view Γ[:, :, b]
        mul!(work, Γ_b, Ot)              # work[a, sout] = ∑_sin Γ[a, sin, b] · O[sout, sin]
        acc += dot(Γ_b, work)
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
