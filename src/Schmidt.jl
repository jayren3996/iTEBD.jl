#---------------------------------------------------------------------------------------------------
# Schmidt Canonical Form
#---------------------------------------------------------------------------------------------------
"""
schmidt_canonical(Γ; kerwords)

Schmidt Canonical Form
1. Return a Schmidt canonical form.
2. This algorithm assume there is no degeneracy.
"""
function schmidt_canonical(
    Γ::AbstractArray{<:Number,3};
    maxdim=MAXDIM, cutoff=SVDTOL, renormalize=false,
    zerotol=ZEROTOL
)
    # Right eigenvector
    R = steady_mat(Γ; dir=:r)
    er, vr = eigen(R)
    n = findfirst(x -> x>zerotol, er)
    er, vr = er[n:end], vr[:,n:end]

    # Left eigenvector
    L = steady_mat(Γ; dir=:l)
    el, vl = eigen(L)
    n = findfirst(x -> x>zerotol, el)
    el, vl = el[n:end], vl[:,n:end]

    X = vr * Diagonal(sqrt.(er)) * vr' 
    Yt = vl * Diagonal(sqrt.(el)) * vl'
    X_inv = vr * Diagonal(er .^ (-0.5)) * vr' 
    Yt_inv = vl * Diagonal(el .^ (-0.5)) * vl' 

    U, S, V = svd_trim(Yt * X; maxdim, cutoff, renormalize)
    R_mat = Yt_inv * U * Diagonal(S)
    L_mat = V * X_inv
    Γ_new = canonical_gauging(Γ, R_mat, L_mat)
    Γ_new, S
end
#---------------------------------------------------------------------------------------------------
# Multiple tensors
function schmidt_canonical(
    Γs::AbstractVector{<:AbstractArray{<:Number, 3}};
    maxdim=MAXDIM, cutoff=SVDTOL, renormalize=false
)
    n = length(Γs)
    Γ_grouped = tensor_group(Γs)

    A, λ = schmidt_canonical(Γ_grouped; maxdim, cutoff, renormalize)
    tensor_lmul!(λ, A)
    Γs, λs = tensor_decomp!(A, λ, n; maxdim, cutoff, renormalize)
    Γs, push!(λs, λ)
end
#---------------------------------------------------------------------------------------------------
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
