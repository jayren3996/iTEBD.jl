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
    Γ_reshaped .= λ .* Γ_reshaped
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
    Γ_reshaped .= Γ_reshaped .* reshape(λ, 1, :)
end

function _finite_entries(x, name::AbstractString)
    all(isfinite, x) || throw(ArgumentError("$name must contain only finite values"))
    return nothing
end

function _support_tol(vals::AbstractVector{<:Number}; atol::Real=ZEROTOL, rtol::Real=sqrt(eps(Float64)))
    mags = Float64.(abs.(vals))
    scale = isempty(mags) ? 0.0 : maximum(mags)
    return max(Float64(atol), Float64(rtol) * scale)
end

"""
    BondStat

Concrete struct holding truncation statistics for a single bond.
"""
struct BondStat
    bond::Int
    chi_req::Int
    chi_keep::Int
    discarded_weight::Float64
    saturated::Bool
    target_met::Bool
    smallest_kept_sv::Float64
    largest_discarded_sv::Float64
end

function _safe_reciprocal(
    vals::AbstractVector{<:Number};
    atol::Real=ZEROTOL,
    rtol::Real=eps(Float64),
)
    _finite_entries(vals, "Schmidt values")
    T = promote_type(eltype(vals), Float64)
    invvals = Vector{T}(undef, length(vals))
    tol = _support_tol(vals; atol, rtol)
    for i in eachindex(vals)
        invvals[i] = abs(vals[i]) > tol ? inv(T(vals[i])) : zero(T)
    end
    return invvals
end

function _renormalize_singular_values!(S::AbstractVector)
    isempty(S) && throw(ArgumentError("cannot renormalize empty singular-value vector"))
    nrm = norm(S)
    isfinite(nrm) && nrm > 0 || throw(ArgumentError("cannot renormalize singular values with zero or non-finite norm"))
    S ./= nrm
    return S
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

"""
    tensor_umul!(umat, Γ)

Apply a dense operator `umat` to the physical leg of a local three-leg tensor
in place.

Parameters:
- `umat`
  Dense operator acting on the physical Hilbert space.
- `Γ`
  Local tensor with shape `(Dl, d, Dr)`, which is mutated in place.

Returns:
- The mutated tensor `Γ`.

Notes:
- This uses a pre-allocated temporary buffer and copies the result back into
  `Γ`, avoiding replacement of the array reference in the parent `iMPS`.
"""
function tensor_umul!(umat::AbstractMatrix, Γ::AbstractArray{<:Number, 3})
    @tensor tmp[:] := umat[-2,1] * Γ[-1,1,-3]
    Γ .= tmp
    return Γ
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
    @inbounds tensor = Γs[1]
    n = length(Γs) 
    isone(n) && return copy(tensor)
    @inbounds for i in 2:n
        χ = size(Γs[i], 1) 
        tensor = reshape(tensor, :, χ) * reshape(Γs[i], χ, :)
    end
    @inbounds α, β = size(Γs[1], 1), size(Γs[n], 3)
    reshape(tensor, α, :, β)
end

#---------------------------------------------------------------------------------------------------
# Tensor Decomposition
#---------------------------------------------------------------------------------------------------
function _svd_with_fallback(mat::AbstractMatrix)
    try
        svd(mat)
    catch e
        if e isa LAPACKException
            ϵ = SVDTOL * max(norm(mat), 1.0)
            perturbed = copy(mat)
            for i in 1:min(size(mat)...)
                perturbed[i, i] += ϵ
            end
            svd(perturbed; alg=LinearAlgebra.DivideAndConquer())
        else
            rethrow(e)
        end
    end
end

function _iterative_svd_trim(
    mat::AbstractMatrix;
    maxdim::Integer=MAXDIM,
    svd_min::Real=SVDTOL,
    renormalize::Bool=false,
)
    m, n = size(mat)
    k = min(maxdim, min(m, n))
    T = eltype(mat)
    SType = typeof(sqrt(float(real(zero(T)))))

    # Use eigsolve on the gram matrix to get the top singular values
    # For a matrix A, the singular values are sqrt(eigenvalues of A'*A)
    # We compute the top k eigenvalues of A'*A
    # WARNING: This path squares condition numbers and is only safe for well-conditioned matrices.
    maxabs = maximum(abs, mat)
    if maxabs > 0 && minimum(abs, mat) / maxabs < sqrt(eps(Float64))
        @warn "iterative SVD path used on a potentially ill-conditioned matrix; " *
              "the Gram-matrix approach squares condition numbers and may lose accuracy"
    end

    # Choose which gram matrix to use based on dimensions
    if m <= n
        # Compute A * A' (m × m)
        f = x -> mat * (mat' * x)
        v0 = randn(T, m)
    else
        # Compute A' * A (n × n)
        f = x -> mat' * (mat * x)
        v0 = randn(T, n)
    end

    # eigsolve operates on the Gram matrix, whose eigenvalues are σ². The
    # residual tolerance therefore needs to be on the σ² scale, not σ — passing
    # tol=svd_min would give singular-value precision of only sqrt(svd_min).
    # Clamp below at machine precision to avoid asking for sub-eps residuals.
    eig_tol = max(svd_min^2, eps(real(SType)))
    vals, vecs, info = eigsolve(f, v0, k, :LM; ishermitian=true, tol=eig_tol)
    if info.converged < min(k, length(vals))
        @warn "Iterative SVD eigsolve did not fully converge" info=info wanted=k
    end

    # Clip negative eigenvalues from numerical noise, but keep at least the
    # leading Ritz value to match the dense path when every singular value is
    # below svd_min.
    ritz = sort(
        [(SType(sqrt(max(real(val), 0.0))), vec) for (val, vec) in zip(vals, vecs)];
        by=first,
        rev=true,
    )
    svals = SType[]
    svecs = Vector{Vector{T}}()
    for (sval, vec) in ritz
        push!(svals, sval)
        push!(svecs, vec)
    end

    len = min(maxdim, count(>=(svd_min), svals))
    if !isempty(svals)
        len = max(1, len)
    end

    S = svals[1:len]
    if any(s -> 0 < s <= ZEROTOL, S)
        res = _svd_with_fallback(mat)
        vals = res.S
        dense_len = min(maxdim, count(>=(svd_min), vals))
        if !isempty(vals)
            dense_len = max(1, dense_len)
        end
        U = res.U[:, 1:dense_len]
        Sdense = vals[1:dense_len]
        V = res.Vt[1:dense_len, :]
        if renormalize
            _renormalize_singular_values!(Sdense)
        end
        return U, Sdense, V
    end

    if m <= n
        # U vectors are the eigenvectors of A*A'
        U_mat = hcat(svecs[1:len]...)
        # V = S^{-1} * U' * A
        V_mat = similar(mat, len, n)
        for i in 1:len
            if S[i] > 0
                V_mat[i, :] = (U_mat[:, i]' * mat) / S[i]
            else
                V_mat[i, :] .= zero(T)
            end
        end
    else
        # V vectors are the eigenvectors of A'*A
        Vt_mat = hcat(svecs[1:len]...)
        # U = A * V * S^{-1}
        U_mat = similar(mat, m, len)
        for i in 1:len
            if S[i] > 0
                U_mat[:, i] = (mat * Vt_mat[:, i]) / S[i]
            else
                U_mat[:, i] .= zero(T)
            end
        end
        V_mat = Vt_mat'
    end

    if renormalize
        _renormalize_singular_values!(S)
    end

    return U_mat, S, V_mat
end

"""
    svd_trim(mat; maxdim, svd_min, renormalize, use_iterative)

Compute an SVD of `mat` and truncate the spectrum.

Parameters:
- `mat`
  Dense matrix to decompose.

Keyword arguments:
- `maxdim`
  Maximum number of singular values to keep.
- `svd_min`
  Singular-value threshold. Singular values below this threshold are discarded.
- `renormalize`
  If `true`, renormalize the retained singular values after truncation.
- `use_iterative`
  If `true`, use an iterative Krylov-based SVD instead of dense LAPACK.
  This is beneficial when `maxdim` is much smaller than the matrix dimensions.
  If `nothing`, the method is chosen automatically based on matrix size and
  `maxdim`. The default is `false`.

Returns:
- `(U, S, V)` where `U * Diagonal(S) * V` is the truncated decomposition.

Notes:
- The routine first tries Julia's default `svd`.
- If LAPACK throws an exception, a small diagonal perturbation is added and the
  decomposition is retried with divide-and-conquer.
- The truncation rule is "keep values until hitting either `svd_min` or
  `maxdim`".
- Iterative SVD uses KrylovKit's `eigsolve` on the Gram matrix and is most
  efficient when only a few leading singular values are needed from a large
  matrix.
"""
function svd_trim(
    mat::AbstractMatrix;
    maxdim::Integer=MAXDIM,
    svd_min::Real=SVDTOL,
    renormalize::Bool=false,
    use_iterative::Union{Bool,Nothing}=false,
)
    maxdim > 0 || throw(ArgumentError("maxdim must be positive"))
    isfinite(svd_min) && svd_min >= 0 || throw(ArgumentError("svd_min must be finite and non-negative"))
    _finite_entries(mat, "SVD input matrix")

    m, n = size(mat)
    # Auto-select iterative only when maxdim is extremely small relative to matrix dimensions
    if isnothing(use_iterative)
        use_iterative = maxdim < min(m, n) ÷ 10 && min(m, n) > 200
    end

    if use_iterative
        return _iterative_svd_trim(mat; maxdim, svd_min, renormalize)
    end

    res = _svd_with_fallback(mat)
    vals = res.S
    len = min(maxdim, count(>=(svd_min), vals))
    if !isempty(vals)
        len = max(1, len)
    end
    U = res.U[:, 1:len]
    S = vals[1:len]
    V = res.Vt[1:len, :]
    if renormalize
        _renormalize_singular_values!(S)
    end
    U, S, V
end
#---------------------------------------------------------------------------------------------------
"""
    _discarded_weight_choice(s; mindim=1, maxdim=MAXDIM, truncerr=0.0, svd_min=0.0)

Choose the kept bond dimension directly from a singular-value spectrum `s`.

This internal helper normalizes the weights `abs2.(s)`, finds the smallest
`χ_req` with discarded weight at most `truncerr`, and then applies the requested
bounds. The optional `svd_min` acts as an additional absolute floor on kept
singular values after the discarded-weight target is evaluated.
"""
function _discarded_weight_choice(
    s::AbstractVector;
    mindim::Integer=1,
    maxdim::Integer=MAXDIM,
    truncerr::Real=0.0,
    svd_min::Real=0.0
)
    mindim > 0 || throw(ArgumentError("mindim must be positive"))
    maxdim > 0 || throw(ArgumentError("maxdim must be positive"))
    maxdim >= mindim || throw(ArgumentError("maxdim must be at least mindim"))
    truncerr >= 0 || throw(ArgumentError("truncerr must be non-negative"))
    svd_min >= 0 || throw(ArgumentError("svd_min must be non-negative"))
    mindim_i = Int(mindim)
    maxdim_i = Int(maxdim)
    truncerr_f = Float64(truncerr)
    svd_min_f = Float64(svd_min)
    target_slack = sqrt(eps(Float64))

    n = length(s)
    n == 0 && return (
        chi_req=0,
        chi_keep=0,
        weights=Float64[],
        discarded_weight=0.0,
        saturated=false,
        target_met=true,
        smallest_kept_sv=0.0,
        largest_discarded_sv=0.0,
    )

    total_weight = 0.0
    for si in s
        total_weight += Float64(abs2(si))
    end
    total_weight > 0 || return (
        chi_req=1,
        chi_keep=1,
        weights=zeros(Float64, n),
        discarded_weight=0.0,
        saturated=false,
        target_met=true,
        smallest_kept_sv=Float64(abs(s[1])),
        largest_discarded_sv=n > 1 ? Float64(abs(s[2])) : 0.0,
    )

    probs = Vector{Float64}(undef, n)
    chi_by_floor = 0
    use_floor = svd_min_f > 0

    chi_req = n
    cumulative = 0.0
    for χ in 1:n
        @inbounds si = s[χ]
        @inbounds probs[χ] = Float64(abs2(si)) / total_weight
        if use_floor && abs(si) >= svd_min_f
            chi_by_floor += 1
        end
    end

    for χ in 1:n
        @inbounds cumulative += probs[χ]
        discarded_weight = χ < n ? 1.0 - cumulative : 0.0
        if discarded_weight <= truncerr_f + target_slack
            chi_req = χ
            break
        end
    end

    chi_keep = min(n, clamp(chi_req, mindim_i, maxdim_i))
    if use_floor
        chi_keep = min(chi_keep, max(chi_by_floor, min(mindim_i, n)))
    end

    kept_weight = 0.0
    @inbounds for i in 1:chi_keep
        kept_weight += probs[i]
    end
    discarded_weight = chi_keep < n ? 1.0 - kept_weight : 0.0
    (
        chi_req=chi_req,
        chi_keep=chi_keep,
        weights=probs,
        discarded_weight=discarded_weight,
        saturated=chi_req > maxdim_i,
        target_met=discarded_weight <= truncerr_f + target_slack,
        smallest_kept_sv=Float64(abs(s[chi_keep])),
        largest_discarded_sv=chi_keep < n ? Float64(abs(s[chi_keep + 1])) : 0.0,
    )
end

function _svd_truncate_by_error(
    mat::AbstractMatrix;
    maxdim::Integer=MAXDIM,
    mindim::Integer=1,
    truncerr::Real=0.0,
    svd_min::Real=0.0,
    renormalize::Bool=false
)
    _finite_entries(mat, "SVD input matrix")
    res = _svd_with_fallback(mat)
    choice = _discarded_weight_choice(
        res.S;
        mindim,
        maxdim,
        truncerr,
        svd_min,
    )
    len = choice.chi_keep
    U = res.U[:, 1:len]
    S = res.S[1:len]
    V = res.Vt[1:len, :]
    if renormalize
        _renormalize_singular_values!(S)
    end
    U, S, V, choice
end
#---------------------------------------------------------------------------------------------------
"""
    tensor_svd(T; maxdim, mindim, truncerr, svd_min, renormalize, return_stats)

Perform an SVD of a four-leg block tensor by fusing its left and right halves.

Parameters:
- `T`
  Four-leg tensor with shape `(Dl, d1, d2, Dr)`.

Keyword arguments:
- `maxdim`, `mindim`, `truncerr`, `svd_min`, `renormalize`
  Passed through to the relevant internal SVD truncation rule.

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
    maxdim::Integer=MAXDIM,
    mindim::Integer=1,
    truncerr::Union{Nothing,Real}=nothing,
    svd_min::Real=SVDTOL,
    renormalize::Bool=false,
    return_stats::Bool=false
)
    α, d1, d2, β = size(T)
    mat = reshape(T, α*d1, :)
    if isnothing(truncerr)
        u, S, v = svd_trim(mat; maxdim, svd_min, renormalize)
        stats = nothing
    else
        u, S, v, stats = _svd_truncate_by_error(
            mat;
            maxdim,
            mindim,
            truncerr,
            svd_min,
            renormalize,
        )
    end
    U = reshape(u, α, d1, :)
    V = reshape(v, :, d2, β)
    return_stats ? (U, S, V, stats) : (U, S, V)
end
#---------------------------------------------------------------------------------------------------
"""
    tensor_decomp!(Γ, λl, n; maxdim, mindim, truncerr, svd_min, renormalize)

Decompose a grouped block tensor back into `n` site-local stored tensors.

Parameters:
- `Γ`
  Grouped three-leg tensor representing `n` consecutive sites.
- `λl`
  Schmidt values on the left bond entering that block.
- `n`
  Number of sites into which the grouped tensor should be decomposed.

Keyword arguments:
- `maxdim`, `mindim`, `truncerr`, `svd_min`, `renormalize`
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
    maxdim::Integer=MAXDIM,
    mindim::Integer=1,
    truncerr::Union{Nothing,Real}=nothing,
    svd_min::Real=SVDTOL,
    renormalize::Bool=false,
    return_stats::Bool=false,
    bond_indices::Union{Nothing,AbstractVector{<:Integer}}=nothing,
    atol::Real=ZEROTOL,
    rtol::Real=eps(Float64),
)
    β = size(Γ, 3)
    d = round(Int, size(Γ, 2)^(1/n))
    d^n == size(Γ, 2) || throw(ArgumentError(
        "inferred physical dimension d=$d does not satisfy d^$n == $(size(Γ, 2)); " *
        "grouped tensor has malformed physical leg size"
    ))
    Γs = Vector{Array{eltype(Γ), 3}}(undef, n)
    λs = Vector{Vector{eltype(λl)}}(undef, n-1)
    bond_stats = return_stats ? BondStat[] : nothing
    Ti, λi = Γ, λl
    for i=1:n-2
        Ti_reshaped = reshape(Ti, size(Ti,1), d, :, β)
        if return_stats
            Ai, Λ, Ti, stats = tensor_svd(
                Ti_reshaped;
                maxdim,
                mindim,
                truncerr,
                svd_min,
                renormalize,
                return_stats=true,
            )
            if !isnothing(stats)
                bond = isnothing(bond_indices) ? i : bond_indices[i]
                push!(bond_stats, BondStat(
                    bond,
                    stats.chi_req,
                    stats.chi_keep,
                    stats.discarded_weight,
                    stats.saturated,
                    stats.target_met,
                    stats.smallest_kept_sv,
                    stats.largest_discarded_sv,
                ))
            end
        else
            Ai, Λ, Ti = tensor_svd(
                Ti_reshaped;
                maxdim,
                mindim,
                truncerr,
                svd_min,
                renormalize,
            )
        end
        tensor_lmul!(_safe_reciprocal(λi; atol, rtol), Ai)
        tensor_rmul!(Ai, Λ)
        tensor_lmul!(Λ, Ti)
        Γs[i] = Ai
        λs[i] = Λ
        λi = Λ
    end
    Ti_reshaped = reshape(Ti, size(Ti,1), d, :, β)
    if return_stats
        Ai, Λ, Ti, stats = tensor_svd(
            Ti_reshaped;
            maxdim,
            mindim,
            truncerr,
            svd_min,
            renormalize,
            return_stats=true,
        )
        if !isnothing(stats)
            bond = isnothing(bond_indices) ? n - 1 : bond_indices[n - 1]
            push!(bond_stats, BondStat(
                bond,
                stats.chi_req,
                stats.chi_keep,
                stats.discarded_weight,
                stats.saturated,
                stats.target_met,
                stats.smallest_kept_sv,
                stats.largest_discarded_sv,
            ))
        end
    else
        Ai, Λ, Ti = tensor_svd(
            Ti_reshaped;
            maxdim,
            mindim,
            truncerr,
            svd_min,
            renormalize,
        )
    end
    tensor_lmul!(_safe_reciprocal(λi; atol, rtol), Ai)
    tensor_rmul!(Ai, Λ)
    Γs[n-1] = Ai
    λs[n-1] = Λ
    Γs[n] = Ti
    return_stats ? (Γs, λs, bond_stats) : (Γs, λs)
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
# Matrix-free transfer-map application
#---------------------------------------------------------------------------------------------------
"""
    apply_transfer(T1, T2, ρ; dir=:r)

Apply the single-site mixed transfer map E(T1, T2) to a matrix `ρ` without
materializing the χ²×χ² dense transfer matrix produced by [`gtrm`](@ref).

With `T1` of shape `(χL1, d, χR1)` and `T2` of shape `(χL2, d, χR2)`:

* `dir=:r` takes `ρ` of shape `(χR2, χR1)` to a matrix of shape `(χL2, χL1)`:

  `ρ'[a, b] = ∑_{a',b',s} T2[a, s, a'] * ρ[a', b'] * conj(T1[b, s, b'])`

* `dir=:l` takes `ρ` of shape `(χL2, χL1)` to a matrix of shape `(χR2, χR1)`
  via the corresponding adjoint sweep.

Each call costs `O(d·χ³)` and is the matrix-free building block underlying the
fast `inner_product` Krylov path.
"""
function apply_transfer(
    T1::AbstractArray{<:Number, 3},
    T2::AbstractArray{<:Number, 3},
    ρ::AbstractMatrix;
    dir::Symbol=:r,
)
    χL1, d, χR1 = size(T1)
    χL2, d2, χR2 = size(T2)
    d == d2 || throw(DimensionMismatch(
        "physical dimensions of T1 ($d) and T2 ($d2) must match"
    ))
    if dir === :r
        size(ρ) == (χR2, χR1) || throw(DimensionMismatch(
            "ρ has size $(size(ρ)); expected ($χR2, $χR1) for dir=:r"
        ))
        # Step 1: tmp[χL2·d, χR1] = T2_mat[χL2·d, χR2] * ρ[χR2, χR1]
        T2_mat = reshape(T2, χL2 * d, χR2)
        tmp = T2_mat * ρ
        # Step 2: ρ'[χL2, χL1] = tmp_view[χL2, d·χR1] * adjoint(T1_mat[χL1, d·χR1])
        T1_mat = reshape(T1, χL1, d * χR1)
        return reshape(tmp, χL2, d * χR1) * adjoint(T1_mat)
    elseif dir === :l
        size(ρ) == (χL2, χL1) || throw(DimensionMismatch(
            "ρ has size $(size(ρ)); expected ($χL2, $χL1) for dir=:l"
        ))
        # Step 1: tmp[χL1, d·χR2] = transpose(ρ)[χL1, χL2] * T2_mat[χL2, d·χR2]
        T2_mat = reshape(T2, χL2, d * χR2)
        tmp = transpose(ρ) * T2_mat                           # (χL1, d·χR2)
        tmp = reshape(tmp, χL1 * d, χR2)
        # Step 2: ρ'[χR2, χR1] = adjoint(conj(T1))[χR1, χL1·d] * tmp[χL1·d, χR2]
        #   ⇔ ρ'[χR2, χR1] = transpose(tmp)[χR2, χL1·d] * (T1_mat_conj_flat)[χL1·d, χR1]
        T1_mat = reshape(T1, χL1 * d, χR1)
        return transpose(tmp) * conj(T1_mat)
    else
        throw(ArgumentError("Illegal direction: $dir."))
    end
end

"""
    apply_chain_transfer(T1s, T2s, ρ; dir=:r)

Apply the n-site unit-cell mixed transfer
`E_cell = E_1 · E_2 · … · E_n` to a matrix `ρ` via a site-by-site sweep,
without materializing any χ²×χ² intermediate.

Cost is `O(n·d·χ³)` per call versus `O(n·χ⁶)` for the dense product, with
identical numerical result up to roundoff. For `dir=:r` the sweep applies
`E_n` first, then `E_{n-1}`, …, then `E_1`, matching the matrix-product
ordering returned by [`gtrm`](@ref).
"""
function apply_chain_transfer(
    T1s::AbstractVector{<:AbstractArray{<:Number, 3}},
    T2s::AbstractVector{<:AbstractArray{<:Number, 3}},
    ρ::AbstractMatrix;
    dir::Symbol=:r,
)
    n = length(T1s)
    n == length(T2s) || throw(ArgumentError(
        "T1s and T2s have different lengths: $n vs $(length(T2s))"
    ))
    n > 0 || throw(ArgumentError("T1s and T2s must be non-empty"))
    current = ρ
    if dir === :r
        for i in n:-1:1
            current = apply_transfer(T1s[i], T2s[i], current; dir=:r)
        end
    elseif dir === :l
        for i in 1:n
            current = apply_transfer(T1s[i], T2s[i], current; dir=:l)
        end
    else
        throw(ArgumentError("Illegal direction: $dir."))
    end
    return current
end

# Workspace used by the in-place transfer-chain sweep. Buffers are sized to the
# maximum intermediate needed across all sites, so a single workspace suffices
# even when bond dimensions vary inside the unit cell. dir=:r only (the
# matrix-free inner-product / dominant-eigenvalue path); dir=:l falls back to
# the allocating `apply_chain_transfer`.
struct ChainTransferWorkspace{T<:Number}
    tmp::Vector{T}        # holds the (χL2·d, χR1) per-site intermediate
    buf_a::Vector{T}      # ping-pong buffers for the running ρ
    buf_b::Vector{T}
end

function ChainTransferWorkspace(T::Type, T1s, T2s)
    n = length(T1s)
    tmp_cap = 0
    rho_cap = 0
    for i in 1:n
        χL1, d, χR1 = size(T1s[i])
        χL2, _, χR2 = size(T2s[i])
        tmp_cap = max(tmp_cap, χL2 * d * χR1)
        rho_cap = max(rho_cap, χR2 * χR1, χL2 * χL1)
    end
    ChainTransferWorkspace{T}(
        Vector{T}(undef, tmp_cap),
        Vector{T}(undef, rho_cap),
        Vector{T}(undef, rho_cap),
    )
end

# Convenience constructor: infer the element type from the input tensors.
ChainTransferWorkspace(T1s, T2s) = ChainTransferWorkspace(
    promote_type(eltype(T1s[1]), eltype(T2s[1])), T1s, T2s,
)

"""
    apply_chain_transfer!(ws, T1s, T2s, ρ; dir=:r)

In-place version of [`apply_chain_transfer`](@ref) that reuses the buffers
stored in `ws::ChainTransferWorkspace`. Returns a reshape view into one of the
workspace's ping-pong buffers; the returned matrix aliases workspace memory and
is invalidated on the next call.

Only `dir=:r` is supported in-place; this is the path used by the matrix-free
Krylov inner-product loop. Allocations are O(1) per call (just reshape views),
independent of the unit cell length.
"""
function apply_chain_transfer!(
    ws::ChainTransferWorkspace{T},
    T1s::AbstractVector{<:AbstractArray{<:Number, 3}},
    T2s::AbstractVector{<:AbstractArray{<:Number, 3}},
    ρ::AbstractMatrix;
    dir::Symbol=:r,
) where {T}
    dir === :r || throw(ArgumentError(
        "apply_chain_transfer! only supports dir=:r; use apply_chain_transfer for dir=:l"
    ))
    n = length(T1s)
    n == length(T2s) || throw(ArgumentError(
        "T1s and T2s have different lengths: $n vs $(length(T2s))"
    ))
    n > 0 || throw(ArgumentError("T1s and T2s must be non-empty"))

    χ_in1, χ_in2 = size(ρ)
    copyto!(view(ws.buf_a, 1:(χ_in1 * χ_in2)), vec(ρ))
    in_buf = ws.buf_a
    out_buf = ws.buf_b
    cur_rows, cur_cols = χ_in1, χ_in2

    @inbounds for i in n:-1:1
        χL1, d, χR1 = size(T1s[i])
        χL2, d2, χR2 = size(T2s[i])
        d == d2 || throw(DimensionMismatch(
            "physical dimensions of T1[$i] ($d) and T2[$i] ($d2) must match"
        ))
        cur_rows == χR2 && cur_cols == χR1 || throw(DimensionMismatch(
            "ρ shape mismatch at site $i: have ($cur_rows, $cur_cols), expected ($χR2, $χR1)"
        ))
        ρ_view = reshape(view(in_buf, 1:(χR2 * χR1)), χR2, χR1)
        T2_mat = reshape(T2s[i], χL2 * d, χR2)
        T1_mat = reshape(T1s[i], χL1, d * χR1)
        tmp_view = reshape(view(ws.tmp, 1:(χL2 * d * χR1)), χL2 * d, χR1)
        mul!(tmp_view, T2_mat, ρ_view)
        out_view = reshape(view(out_buf, 1:(χL2 * χL1)), χL2, χL1)
        mul!(out_view, reshape(tmp_view, χL2, d * χR1), adjoint(T1_mat))
        cur_rows, cur_cols = χL2, χL1
        in_buf, out_buf = out_buf, in_buf
    end
    return reshape(view(in_buf, 1:(cur_rows * cur_cols)), cur_rows, cur_cols)
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
