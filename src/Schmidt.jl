#---------------------------------------------------------------------------------------------------
# Schmidt Canonical Form
#---------------------------------------------------------------------------------------------------
"""
schmidt_canonical(Γ; kerwords)

Schmidt Canonical Form
1. Given a right canonical form, return a Schmidt canonical form.
2. This algorithm assume there is no degeneracy.
"""
function schmidt_canonical(
    Γ::AbstractArray{<:Number,3};
    maxdim=MAXDIM, cutoff=SVDTOL, renormalize=false,
    zerotol=ZEROTOL
)
    R = steady_mat(Γ; dir=:r)
    L = steady_mat(Γ; dir=:l)
    er, vr = eigen(R)
    el, vl = eigen(L)
    n = findlast(x -> x>zerotol, er)
    @assert findlast(x -> x>zerotol, el) == n "zeros of left & right eigenvalues are different."
    vr, vl = vr[:,1:n], vl[:,1:n]
    er, el = er[1:n], rl[1:n]
    X = vr * Diagonal(sqrt.(er)) * vr' 
    Yt = vl * Diagonal(sqrt.(el)) * vl'
    X_inv = vr * Diagonal(er .^ (-0.5)) * vr' 
    Yt_inv = vr * Diagonal(el .^ (-0.5)) * vr' 

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
