#---------------------------------------------------------------------------------------------------
# Basic Tensor Multiplication
#---------------------------------------------------------------------------------------------------
"""
    tensor_lmul!(λ, Γ)

Multiply a tensor by a diagonal matrix built from `λ` on its leftmost bond.

Parameters:
- `λ`
  Vector of coefficients to apply on the leftmost virtual leg.
- `Γ`
  Tensor whose first dimension must have length `length(λ)`.

Returns:
- The mutated tensor `Γ`.

Notes:
- This is an in-place helper used throughout the package.
- The tensor is reshaped internally into matrix form, multiplied, and then left
  in its original array storage.
"""
function tensor_lmul!(λ::AbstractVector{<:Number}, Γ::AbstractArray)
    α = size(Γ, 1)
    Γ_reshaped = reshape(Γ, α, :)
    lmul!(Diagonal(λ), Γ_reshaped)
end
#---------------------------------------------------------------------------------------------------
"""
    tensor_rmul!(Γ, λ)

Multiply a tensor by a diagonal matrix built from `λ` on its rightmost bond.

Parameters:
- `Γ`
  Tensor whose last dimension must have length `length(λ)`.
- `λ`
  Vector of coefficients to apply on the rightmost virtual leg.

Returns:
- The mutated tensor `Γ`.

Notes:
- This is the right-bond companion of [`tensor_lmul!`](@ref).
- The operation is in place.
"""
function tensor_rmul!(Γ::AbstractArray, λ::AbstractVector{<:Number})
    β = size(Γ)[end]
    Γ_reshaped = reshape(Γ, :, β)
    rmul!(Γ_reshaped, Diagonal(λ))
end
#---------------------------------------------------------------------------------------------------
"""
    tensor_umul(umat, Γ) 

Apply a dense operator `umat` to the physical leg of a local three-leg tensor.

Parameters:
- `umat`
  Dense operator acting on the physical Hilbert space.
- `Γ`
  Local tensor with shape `(Dl, d, Dr)`, where the second index is the physical
  leg acted on by `umat`.

Returns:
- A new tensor with the same shape as `Γ`.

Notes:
- This is not in-place.
- The matrix dimension of `umat` must match the physical dimension `d`.
"""
function tensor_umul(umat::AbstractMatrix, Γ::AbstractArray{<:Number, 3})
    @tensor Γ_new[:] := umat[-2,1] * Γ[-1,1,-3]
    Γ_new
end

#---------------------------------------------------------------------------------------------------
# Tensor Grouping
#---------------------------------------------------------------------------------------------------
"""
    tensor_group_2(ΓA, ΓB) 

Group two neighboring local tensors into a single three-leg block tensor.

Parameters:
- `ΓA`, `ΓB`
  Neighboring local tensors with matching internal bond dimensions.

Returns:
- A grouped tensor with shape `(Dl, dA * dB, Dr)`.

Notes:
- The two physical dimensions are fused into a single composite physical leg.
"""
function tensor_group_2(ΓA::AbstractArray{<:Number, 3}, ΓB::AbstractArray{<:Number, 3})
    α, d1, χ = size(ΓA)
    d2 = size(ΓB, 2)
    β = size(ΓB, 3)
    tensor = reshape(ΓA, α*d1, χ) * reshape(ΓB, χ, d2*β)
    reshape(tensor, α, d1*d2, β)
end
#---------------------------------------------------------------------------------------------------
"""
    tensor_group_3(ΓA, ΓB, ΓC) 

Group three neighboring local tensors into one three-leg block tensor.

Parameters:
- `ΓA`, `ΓB`, `ΓC`
  Consecutive local tensors with compatible internal bond dimensions.

Returns:
- A grouped tensor whose physical leg is the product of the three local
  physical dimensions.
"""
function tensor_group_3(ΓA::AbstractArray, ΓB::AbstractArray, ΓC::AbstractArray)
    α, β = size(ΓA, 1), size(ΓC, 3)
    χ1, χ2 = size(ΓA, 3), size(ΓB, 3)
    tensor = reshape(ΓA, :, χ1) * reshape(ΓB, χ1, :)
    tensor = reshape(tensor, :, χ2) * reshape(ΓC, χ2, :)
    reshape(tensor, α, :, β)
end
#---------------------------------------------------------------------------------------------------
"""
    tensor_group(Γs) 

Group a contiguous list of local tensors into a single three-leg block tensor.

Parameters:
- `Γs`
  Vector of local three-leg tensors with compatible neighboring bond
  dimensions.

Returns:
- A grouped tensor with the same left and right bond dimensions as the first and
  last tensors, and with a fused physical leg.

Notes:
- This is the basic grouping primitive used before block SVDs and gate
  application.
- For `length(Γs) == 1`, a copy of the input tensor is returned.
"""
function tensor_group(Γs::AbstractVector{<:AbstractArray{<:Number, 3}})
    tensor = Γs[1]
    n = length(Γs) 
    isone(n) && return copy(tensor)
    for i in 2:n
        χ = size(Γs[i], 1) 
        tensor = reshape(tensor, :, χ) * reshape(Γs[i], χ, :)
    end
    α, β = size(Γs[1], 1), size(Γs[n], 3)
    reshape(tensor, α, :, β)
end

#---------------------------------------------------------------------------------------------------
# Tensor Decomposition
#---------------------------------------------------------------------------------------------------
"""
    svd_trim(mat; maxdim, cutoff, renormalize)

Compute an SVD of `mat` and truncate the spectrum.

Parameters:
- `mat`
  Dense matrix to decompose.

Keyword arguments:
- `maxdim`
  Maximum number of singular values to keep.
- `cutoff`
  Singular-value threshold. Singular values below this threshold are discarded.
- `renormalize`
  If `true`, renormalize the retained singular values after truncation.

Returns:
- `(U, S, V)` where `U * Diagonal(S) * V` is the truncated decomposition.

Notes:
- The routine first tries Julia's default `svd`.
- If LAPACK throws an exception, a small diagonal perturbation is added and the
  decomposition is retried with divide-and-conquer.
- The truncation rule is "keep values until hitting either `cutoff` or
  `maxdim`".
"""
function svd_trim(
    mat::AbstractMatrix;
    maxdim::Integer=MAXDIM,
    cutoff::Real=SVDTOL,
    renormalize::Bool=false
)
    res = try
        svd(mat)  # Standard, faster path
    catch e
        if e isa LAPACKException
            ϵ = SVDTOL  # small positive number

            for i in 1:min(size(mat)...)
                # this actually altered mat, but only marginally. 
                # So I didn't rename the function to svd_trim!
                mat[i,i] += ϵ
            end
            svd(mat; alg=LinearAlgebra.DivideAndConquer())
        else
            rethrow(e)  # rethrow if it’s an unexpected error
        end
    end
    vals = res.S
    (maxdim > length(vals)) && (maxdim = length(vals))
    len::Int64 = 1
    while true
        if isless(vals[len], cutoff)
            len -= 1
            break
        end
        isequal(len, maxdim) && break
        len += 1
    end
    U = res.U[:, 1:len]
    S = vals[1:len]
    V = res.Vt[1:len, :]
    if renormalize
        S ./= norm(S)
    end
    U, S, V
end
#---------------------------------------------------------------------------------------------------
"""
    tensor_svd(T; maxdim, curoff, renormalize)

Perform an SVD of a four-leg block tensor by fusing its left and right halves.

Parameters:
- `T`
  Four-leg tensor with shape `(Dl, d1, d2, Dr)`.

Keyword arguments:
- `maxdim`, `cutoff`, `renormalize`
  Passed through to [`svd_trim`](@ref).

Returns:
- `(U, S, V)` where:
  `U` has shape `(Dl, d1, χ)`,
  `S` is the retained singular-value vector,
  `V` has shape `(χ, d2, Dr)`.

Notes:
- This is the core two-site decomposition primitive used after local gate
  application.
"""
function tensor_svd(
    T::AbstractArray{<:Number, 4};
    maxdim=MAXDIM, cutoff=SVDTOL, renormalize=false
)
    α, d1, d2, β = size(T)
    mat = reshape(T, α*d1, :)
    u, S, v = svd_trim(mat; maxdim, cutoff, renormalize)
    U = reshape(u, α, d1, :)
    V = reshape(v, :, d2, β)
    U, S, V
end
#---------------------------------------------------------------------------------------------------
"""
    tensor_decomp!(Γ, λl, n; maxdim, cutoff, renormalize)

Decompose a grouped block tensor back into `n` site-local stored tensors.

Parameters:
- `Γ`
  Grouped three-leg tensor representing `n` consecutive sites.
- `λl`
  Schmidt values on the left bond entering that block.
- `n`
  Number of sites into which the grouped tensor should be decomposed.

Keyword arguments:
- `maxdim`, `cutoff`, `renormalize`
  Truncation and normalization controls forwarded to the internal SVD steps.

Returns:
- `(Γs, λs)` where `Γs` is a vector of `n` stored local tensors and `λs` is the
  vector of Schmidt spectra on the `n - 1` internal bonds.

Notes:
- The local physical dimension is inferred by assuming the fused physical leg of
  `Γ` has size `d^n`.
- The returned local tensors follow the package storage convention with absorbed
  right Schmidt values.
"""
function tensor_decomp!(
    Γ::AbstractArray{<:Number, 3},
    λl::AbstractVector{<:Real},
    n::Integer;
    maxdim=MAXDIM, cutoff=SVDTOL, renormalize=false
)
    β = size(Γ, 3)
    d = round(Int, size(Γ, 2)^(1/n))
    Γs = Vector{Array{eltype(Γ), 3}}(undef, n)
    λs = Vector{Vector{eltype(λl)}}(undef, n-1)
    Ti, λi = Γ, λl
    for i=1:n-2
        Ti_reshaped = reshape(Ti, size(Ti,1), d, :, β)
        Ai, Λ, Ti = tensor_svd(Ti_reshaped; maxdim, cutoff, renormalize)
        tensor_lmul!(1 ./ λi, Ai)
        tensor_rmul!(Ai, Λ)
        tensor_lmul!(Λ, Ti)
        Γs[i] = Ai
        λs[i] = Λ
        λi = Λ
    end
    Ti_reshaped = reshape(Ti, size(Ti,1), d, :, β)
    Ai, Λ, Ti = tensor_svd(Ti_reshaped; maxdim, cutoff, renormalize)
    tensor_lmul!(1 ./ λi, Ai)
    tensor_rmul!(Ai, Λ)
    Γs[n-1] = Ai
    λs[n-1] = Λ
    Γs[n] = Ti
    Γs, λs
end



#---------------------------------------------------------------------------------------------------
# General transfer matrix
#---------------------------------------------------------------------------------------------------
"""
    gtrm(T1, T2)

Build the mixed transfer matrix between two local tensors.

If `T1` and `T2` are local three-leg tensors with virtual indices
`(α_{i-1}, α_i)` and `(β_{i-1}, β_i)`, this routine builds the dense matrix
representation of the mixed transfer operator

`E(T1, T2)_{(β_{i-1}, α_{i-1}), (β_i, α_i)} =
    sum_s conj(T1[α_{i-1}, s, α_i]) * T2[β_{i-1}, s, β_i]`.

Parameters:
- `T1`, `T2`
  Local three-leg tensors with compatible physical dimensions.

Returns:
- Dense matrix representation of the mixed transfer operator.

Notes:
- `gtrm(T, T)` is the ordinary transfer matrix of `T`.
- This matrix is the one-site building block used in overlap and fixed-point
  computations.
"""
function gtrm(T1::AbstractArray{<:Number, 3}, T2::AbstractArray{<:Number, 3})
    i1, _, k1 = size(T2)
    i2, _, k2 = size(T1)
    T1c = conj(T1)
    ctype = promote_type(eltype(T1c), eltype(T2))
    transfer_mat = Array{ctype, 4}(undef, i1, i2, k1, k2)
    @tensor transfer_mat[:] = T1c[-2,1,-4] * T2[-1,1,-3]
    reshape(transfer_mat, i1*i2, :)
end
#---------------------------------------------------------------------------------------------------
"""
    gtrm(T1s, T2s)

Build the mixed transfer matrix for two full unit cells.

For unit cells `T1s = [T1^(1), ..., T1^(n)]` and `T2s = [T2^(1), ..., T2^(n)]`,
the returned matrix is the ordered product

`E_cell(T1s, T2s) = E(T1^(1), T2^(1)) * ... * E(T1^(n), T2^(n))`.

Parameters:
- `T1s`, `T2s`
  Vectors of local tensors representing two unit cells of the same length.

Returns:
- Dense matrix representation of the mixed transfer operator for the full unit
  cell.

Notes:
- This is the transfer matrix whose dominant eigenvalue is used by
  [`inner_product`](@ref) to define the overlap per unit cell.
"""
function gtrm(
    T1s::AbstractVector{<:AbstractArray{<:Number, 3}},
    T2s::AbstractVector{<:AbstractArray{<:Number, 3}}
)
    n = length(T1s)
    M = gtrm(T1s[1], T2s[1])
    for i in 2:n
        M = M * gtrm(T1s[i], T2s[i])
    end
    M
end
#---------------------------------------------------------------------------------------------------
"""
    trm(T::AbstractArray{<:Number, 3})

Transfer matrix for a single local MPS tensor `T`.

This is shorthand for `gtrm(T, T)`.

Notes:
- For a whole unit cell stored as a vector of tensors, use `gtrm(Ts, Ts)` to
  construct the unit-cell transfer matrix.
"""
function trm(T::AbstractArray{<:Number, 3}) 
    gtrm(T, T)
end
#---------------------------------------------------------------------------------------------------
"""
    otrm(T1::AbstractArray, O::AbstractMatrix, T2::AbstractArray)

Build the operator transfer matrix between two local tensors with a local
operator inserted on the physical leg.

Parameters:
- `T1`, `T2`
  Local three-leg tensors.
- `O`
  Dense one-site operator acting on the physical leg.

Returns:
- Dense matrix representation of the resulting operator transfer matrix.
"""
function otrm(
    T1::AbstractArray{<:Number, 3},
    O::AbstractMatrix{<:Number},
    T2::AbstractArray{<:Number, 3}
)
    i1, j1, k1 = size(T2)
    i2, j2, k2 = size(T1)
    T1c = conj(T1)
    ctype = promote_type(eltype(T1c), eltype(O), eltype(T2))
    transfer_mat = Array{ctype, 4}(undef, i1, i2, k1, k2)
    @tensor transfer_mat[:] = T1c[-2,1,-4] * O[1,2] * T2[-1,2,-3]
    reshape(transfer_mat, i1*i2, :)
end
#---------------------------------------------------------------------------------------------------
"""
    otrm(T1s::AbstractVector, O::AbstractMatrix, T2s::AbstractVector)

Build an operator transfer matrix for a full unit cell with the same local
operator inserted on every site.

Parameters:
- `T1s`, `T2s`
  Vectors of local tensors.
- `O`
  One-site operator inserted at each site.

Returns:
- Dense matrix representation of the operator transfer matrix for the full unit
  cell.
"""
function otrm(
    T1s::AbstractVector{<:AbstractArray{<:Number, 3}},
    O::AbstractMatrix{<:Number},
    T2s::AbstractVector{<:AbstractArray{<:Number, 3}}
)
    n = length(T1s)
    M = otrm(T1s[1], O, T2s[1])
    for i=2:n
        M = M * otrm(T1s[i], O, T2s[i])
    end
    M
end
#---------------------------------------------------------------------------------------------------
"""
    otrm(T1s::AbstractVector, Os::AbstractVector, T2s::AbstractVector)

Build an operator transfer matrix for a full unit cell with site-dependent
operators.

Parameters:
- `T1s`, `T2s`
  Vectors of local tensors.
- `Os`
  Vector of one-site operators, one for each site.

Returns:
- Dense matrix representation of the operator transfer matrix for the full unit
  cell.
"""
function otrm(
    T1s::AbstractVector{<:AbstractArray{<:Number, 3}},
    Os::AbstractVector{<:AbstractMatrix{<:Number}},
    T2s::AbstractVector{<:AbstractArray{<:Number, 3}}
)
    n = length(T1s)
    M = otrm(T1s[1], Os[1], T2s[1])
    for i=2:n
        M = M * otrm(T1s[i], Os[i], T2s[i])
    end
    M
end

#---------------------------------------------------------------------------------------------------
# Normalization
#---------------------------------------------------------------------------------------------------
"""
    tensor_renormalize!(Γ)

Normalize a local tensor using the square root of its self-overlap.

Parameters:
- `Γ`
  Local three-leg tensor to normalize in place.

Returns:
- The mutated tensor `Γ`.

Notes:
- This helper uses [`inner_product`](@ref) on the single-tensor transfer matrix.
- It is primarily useful for low-level normalization workflows.
"""
function tensor_renormalize!(Γ::Array{<:Number, 3})
    Γ_norm = sqrt(inner_product(Γ))
    Γ ./= Γ_norm
end
