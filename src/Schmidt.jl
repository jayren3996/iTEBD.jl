#---------------------------------------------------------------------------------------------------
# Schmidt Canonical Form
#---------------------------------------------------------------------------------------------------
function canonical_gauging(
    Γ::AbstractArray{<:Number, 3},
    L::AbstractMatrix,
    R::AbstractMatrix
)
    @tensor Γ2[:] := R[-1,1] * Γ[1,-2,2] * L[2,-3]
    Γ2
end

function _sanitize_schmidt_values(S::AbstractVector)
    !isempty(S) || throw(ArgumentError("incoming Schmidt spectrum must be nonempty"))
    _finite_entries(S, "incoming Schmidt spectrum")
    return Float64.(abs.(S))
end

function _tolerance(scale::Real; zerotol::Real=ZEROTOL, rtol::Real=sqrt(eps(Float64)))
    return max(Float64(zerotol), Float64(rtol) * max(Float64(scale), 1.0))
end

function _tolerance(vals::AbstractVector{<:Number}; zerotol::Real=ZEROTOL, rtol::Real=sqrt(eps(Float64)))
    scale = isempty(vals) ? 0.0 : maximum(abs.(vals))
    return _tolerance(scale; zerotol, rtol)
end

function _positive_eigensystem(H::AbstractMatrix; zerotol::Real=ZEROTOL, rtol::Real=sqrt(eps(Float64)))
    vals, vecs = eigen(Hermitian(H))
    realvals = Float64.(real.(vals))
    _finite_entries(realvals, "fixed-point eigenvalues")

    tol = _tolerance(realvals; zerotol, rtol)
    clean = similar(realvals)
    for i in eachindex(realvals)
        clean[i] = max(realvals[i], 0.0)
    end

    support = findall(>(tol), clean)
    if isempty(support)
        idx = argmax(abs.(realvals))
        clean[idx] = max(abs(realvals[idx]), eps(Float64))
        support = [idx]
    end
    return clean[support], vecs[:, support]
end

function _transfer_degeneracy(Γ::AbstractArray{<:Number,3})
    Dl, _, Dr = size(Γ)
    Dl == Dr || return (degenerate=false, count=0, leading=0.0, tol=0.0)

    if Dl * Dr > 2500
        sector = _simple_sector_selection(Γ)
        if !isnothing(sector)
            return (degenerate=true, count=sector.sectors, leading=NaN, tol=0.0)
        end
        # Degeneracy detection is diagnostic. Avoid a Krylov preflight here:
        # on moderately large random tensors it can dominate or stall
        # canonicalization before the actual fixed-point solve starts.
        return (degenerate=false, count=0, leading=0.0, tol=0.0)
    end

    vals = eigvals(kraus_mat(Γ, conj(Γ); dir=:r))
    isempty(vals) && return (degenerate=false, count=0, leading=0.0, tol=0.0)
    mags = sort!(Float64.(abs.(vals)); rev=true)
    leading = mags[1]
    tol = _tolerance(leading; zerotol=100*ZEROTOL, rtol=100*sqrt(eps(Float64)))
    leading_count = count(m -> abs(m - leading) <= tol, mags)
    return (degenerate=leading_count > 1 && leading > tol, count=leading_count, leading=leading, tol=tol)
end

function _simple_sector_selection(Γ::AbstractArray{<:Number,3})
    Dl, _, Dr = size(Γ)
    Dl == Dr || return nothing

    weights = zeros(Float64, Dl)
    offdiag = 0.0
    for l in 1:Dl, r in 1:Dr
        w = sum(abs2, @view Γ[l, :, r])
        if l == r
            weights[l] += w
        else
            offdiag += w
        end
    end

    total = sum(weights) + offdiag
    tol = _tolerance(total; zerotol=ZEROTOL, rtol=100*eps(Float64))
    active = findall(>(tol), weights)
    (length(active) > 1 && offdiag <= tol) || return nothing

    choice = active[argmax(weights[active])]
    sector = Array(Γ[choice:choice, :, choice:choice])
    nrm = sqrt(sum(abs2, sector))
    if isfinite(nrm) && nrm > 0
        sector ./= nrm
    end
    return (Γ=sector, S=ones(Float64, 1), index=choice, sectors=length(active))
end

function _handle_noninjective!(degeneracy, noninjective::Symbol)
    if degeneracy.degenerate && noninjective == :error
        throw(ArgumentError(
            "likely non-injective/degenerate transfer spectrum detected " *
            "($(degeneracy.count) leading eigenvalues within $(degeneracy.tol))"
        ))
    end
    return nothing
end
#---------------------------------------------------------------------------------------------------
"""
schmidt_canonical(Γ; kerwords)

Schmidt Canonical Form
1. Return a Schmidt canonical form.
2. This algorithm assume there is no degeneracy.

Parameters:
- `Γ`
  Local three-leg tensor or grouped local tensor representing one periodic block.
- `S`
  Schmidt values on the incoming bond of that block.

Keyword arguments:
- `maxdim=MAXDIM`
  Maximum number of Schmidt values to retain in the canonicalized result.
- `cutoff=SVDTOL`
  Singular values smaller than this threshold are discarded.
- `renormalize=true`
  Whether to renormalize the retained Schmidt values.
- `zerotol=ZEROTOL`
  Threshold used when selecting positive eigenvalues of the transfer-matrix
  fixed points.

Returns:
- `(Γ_new, S_new)` where `Γ_new` is the gauged canonical tensor and `S_new` is
  the resulting Schmidt spectrum.

Notes:
- This routine is the low-level canonicalization kernel used by
  [`canonical!`](@ref).
- It assumes the state is in the non-degenerate injective setting.
- The returned tensor has the right Schmidt values absorbed on its right bond,
  matching the package storage convention.
- The pseudoinverse construction implicitly performs low-rank compression when
  near-zero eigenvalues of the transfer-matrix fixed points are filtered by
  `_positive_eigensystem`. A warning is emitted if the rank of either fixed
  point is significantly smaller than the bond dimension.
"""
function schmidt_canonical(
    Γ::AbstractArray{<:Number,3}, S::AbstractVector;
    maxdim=MAXDIM, cutoff=SVDTOL, renormalize=true,
    zerotol=ZEROTOL,
    noninjective::Symbol=:warn,
    symmetry_break::Symbol=:none,
    tol::Union{Nothing,Real}=nothing,
    maxiter::Union{Nothing,Integer}=nothing,
)
    _validate_canonical_options(maxdim, cutoff, noninjective, symmetry_break)
    S_in = _sanitize_schmidt_values(S)
    length(S_in) == size(Γ, 3) ||
        throw(ArgumentError("incoming Schmidt spectrum length must match the right bond dimension"))
    length(S_in) == size(Γ, 1) ||
        throw(ArgumentError("incoming Schmidt spectrum length must match the left bond dimension"))

    degeneracy = _transfer_degeneracy(Γ)
    _handle_noninjective!(degeneracy, noninjective)
    if degeneracy.degenerate && symmetry_break == :auto
        sector = _simple_sector_selection(Γ)
        if !isnothing(sector)
            if noninjective == :warn
                @warn "Detected likely non-injective/degenerate transfer spectrum; symmetry_break=:auto selected virtual sector $(sector.index) of $(sector.sectors). Proceeding with a symmetry-broken sector; full non-injective block canonical decomposition is not implemented."
            end
            return schmidt_canonical(
                sector.Γ,
                sector.S;
                maxdim,
                cutoff,
                renormalize,
                zerotol,
                noninjective=:ignore,
                symmetry_break=:none,
                tol,
                maxiter,
            )
        end
    end
    if degeneracy.degenerate && noninjective == :warn
        @warn "Detected likely non-injective/degenerate transfer spectrum; proceeding without symmetry-sector selection. Full non-injective block canonical decomposition is not implemented."
    end

    # Right eigenvector
    R = steady_mat(Γ; dir=:r, tol, maxiter)
    er, vr = _positive_eigensystem(R; zerotol)
    bond_dim_r = size(vr, 1)
    if length(er) < bond_dim_r
        @warn "Low-rank compression in schmidt_canonical: right fixed-point rank $(length(er)) < bond dimension $(bond_dim_r)."
    end

    # Left eigenvector
    Γc = copy(Γ)
    iTEBD.tensor_rmul!(Γc, _safe_reciprocal(S_in; atol=ZEROTOL, rtol=0.0))
    Γl = copy(Γc)
    iTEBD.tensor_lmul!(S_in, Γl)
    L = steady_mat(Γl; dir=:l, tol, maxiter)
    el, vl = _positive_eigensystem(L; zerotol)
    bond_dim_l = size(vl, 1)
    if length(el) < bond_dim_l
        @warn "Low-rank compression in schmidt_canonical: left fixed-point rank $(length(el)) < bond dimension $(bond_dim_l)."
    end

    # Avoid Diagonal allocations by using broadcasting
    sqrt_er = sqrt.(er)
    sqrt_el = sqrt.(el)
    inv_sqrt_er = _safe_reciprocal(sqrt_er; atol=ZEROTOL, rtol=0.0)
    inv_sqrt_el = _safe_reciprocal(sqrt_el; atol=ZEROTOL, rtol=0.0)

    X = (vr .* sqrt_er') * vr'
    Yt = (vl .* sqrt_el') * vl'
    X_inv = (vr .* inv_sqrt_er') * vr'
    Yt_inv = (vl .* inv_sqrt_el') * vl'

    U, S_new, V = svd_trim(Yt * (S_in .* X); maxdim, svd_min=cutoff, renormalize)
    R_mat = Yt_inv * U
    L_mat = V * X_inv
    Γ_new = canonical_gauging(Γc, R_mat, L_mat)
    tensor_rmul!(Γ_new, S_new)
    Γ_new, S_new
end
#---------------------------------------------------------------------------------------------------
# Multiple tensors
"""
    schmidt_canonical(Γs, S; maxdim=MAXDIM, cutoff=SVDTOL, renormalize=false)

Canonicalize a full unit cell represented as a vector of local tensors.

Parameters:
- `Γs`
  Vector of local three-leg tensors forming one periodic unit cell.
- `S`
  Schmidt spectrum on the incoming bond to the grouped unit cell.

Keyword arguments:
- `maxdim`, `cutoff`, `renormalize`
  Passed through to the single-block [`schmidt_canonical`](@ref) kernel and the
  subsequent decomposition.

Returns:
- `(Γs_new, λs_new)` where `Γs_new` are the stored right-canonical tensors for
  the unit cell and `λs_new` are the Schmidt spectra on all bonds of that unit
  cell.

Notes:
- For `length(Γs) == 1`, this routine returns a one-site unit cell and applies a
  final normalization so the stored tensor is properly right-canonical.
- For longer unit cells, the grouped canonical tensor is decomposed back into
  site-local tensors with [`tensor_decomp!`](@ref).
"""
function schmidt_canonical(
    Γs::AbstractVector{<:AbstractArray{<:Number, 3}}, S::AbstractVector;
    maxdim=MAXDIM,
    cutoff=SVDTOL,
    renormalize=false,
    noninjective::Symbol=:warn,
    symmetry_break::Symbol=:none,
    tol::Union{Nothing,Real}=nothing,
    maxiter::Union{Nothing,Integer}=nothing,
)
    n = length(Γs)
    Γ_grouped = tensor_group(Γs)

    A, λ = schmidt_canonical(
        Γ_grouped,
        S;
        maxdim,
        cutoff,
        renormalize,
        noninjective,
        symmetry_break,
        tol,
        maxiter,
    )
    if isone(n)
        Dl, d, Dr = size(A)
        overlap = zeros(eltype(A), Dl, Dl)
        for s in 1:d
            As = reshape(A[:, s, :], Dl, Dr)
            overlap .+= As * As'
        end
        scale = sqrt(real(tr(overlap)) / Dl)
        if isfinite(scale) && scale > 0
            A ./= scale
        end
        return [A], [λ]
    end
    tensor_lmul!(λ, A)
    Γs, λs = tensor_decomp!(A, λ, n; maxdim, svd_min=cutoff, renormalize)
    Γs, push!(λs, λ)
end
#---------------------------------------------------------------------------------------------------
#=
export canonical_trim
function canonical_trim(
    Ts::AbstractVector{<:AbstractArray{<:Number, 3}};
    maxdim=MAXDIM, cutoff=SVDTOL, renormalize=true,  
)
    n = length(Ts)
    T = tensor_group(Ts)
    T_RC = right_canonical(T)
    T_BRC = block_trim(T_RC)
    A, λ = schmidt_canonical(T_BRC; renormalize)
    tensor_lmul!(λ, A)
    tensor_decomp!(A, λ, n; maxdim, svd_min=cutoff, renormalize)
end
=#
