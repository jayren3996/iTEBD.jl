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
"""
function schmidt_canonical(
    Γ::AbstractArray{<:Number,3}, S::AbstractVector;
    maxdim=MAXDIM, cutoff=SVDTOL, renormalize=true,
    zerotol=ZEROTOL
)
    # Right eigenvector
    R = steady_mat(Γ; dir=:r)
    er, vr = eigen(R)
    n = findfirst(x -> x>zerotol, er)
    er, vr = er[n:end], vr[:,n:end]

    # Left eigenvector
    Γc = deepcopy(Γ)
    iTEBD.tensor_rmul!(Γc, 1 ./ S)
    Γl = deepcopy(Γc)
    iTEBD.tensor_lmul!(S, Γl)
    L = steady_mat(Γl; dir=:l)
    el, vl = eigen(L)
    n = findfirst(x -> x>zerotol, el)
    el, vl = el[n:end], vl[:,n:end]

    X = vr * Diagonal(sqrt.(er)) * vr' 
    Yt = vl * Diagonal(sqrt.(el)) * vl'
    X_inv = vr * Diagonal(er .^ (-0.5)) * vr' 
    Yt_inv = vl * Diagonal(el .^ (-0.5)) * vl' 

    U, S, V = svd_trim(Yt * Diagonal(S) * X; maxdim, cutoff, renormalize)
    R_mat = Yt_inv * U
    L_mat = V * X_inv
    Γ_new = canonical_gauging(Γc, R_mat, L_mat)
    tensor_rmul!(Γ_new, S)
    Γ_new, S
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
    maxdim=MAXDIM, cutoff=SVDTOL, renormalize=false
)
    n = length(Γs)
    Γ_grouped = tensor_group(Γs)

    A, λ = schmidt_canonical(Γ_grouped, S; maxdim, cutoff, renormalize)
    if isone(n)
        Dl, d, Dr = size(A)
        overlap = zeros(eltype(A), Dl, Dl)
        for s in 1:d
            As = reshape(A[:, s, :], Dl, Dr)
            overlap .+= As * As'
        end
        scale = sqrt(real(tr(overlap)) / Dl)
        A ./= scale
        return [A], [λ]
    end
    tensor_lmul!(λ, A)
    Γs, λs = tensor_decomp!(A, λ, n; maxdim, cutoff, renormalize)
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
    tensor_decomp!(A, λ, n; maxdim, cutoff, renormalize)
end
=#
