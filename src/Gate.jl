#---------------------------------------------------------------------------------------------------
# QUANTUM GATE
#---------------------------------------------------------------------------------------------------
"""
    tensor_applygate!(G, Γs, λl; keywords...)

Apply Gate:
    |
    G
    |              |   |        |
  --Γ--  ==> --λl--Γ₁--Γ₂-- ⋯ --Γₙ--,
where:
    |          |
  --Γₙ--  =  --Aₙ--λₙ-- 

Return list tensor list [Γ₁,⋯,Γₙ], and values list [λ₁,⋯,λₙ₋₁].
"""
function tensor_applygate!(
    G::AbstractMatrix{<:Number}, Γs::AbstractVector{<:AbstractArray{<:Number, 3}},
    λl::AbstractVector{<:Number};
    maxdim=MAXDIM, cutoff=SVDTOL, renormalize=false
)
    n = length(Γs)
    isone(n) && return ([GΓ], [])
    Γ = tensor_group(Γs)
    tensor_lmul!(λl, Γ)
    GΓ = tensor_umul(G, Γ)
    tensor_decomp!(GΓ, λl, n; maxdim, cutoff, renormalize)
end
#---------------------------------------------------------------------------------------------------
export applygate!
function applygate!(
    ψ::iMPS, G::AbstractMatrix,
    i::Integer, j::Integer;
    maxdim=MAXDIM, cutoff=SVDTOL, renormalize=true
)
    if isequal(i, j)
        ψ.Γ[i] = tensor_umul(G, ψ.Γ[i])
        return ψ
    end
    inds = j>i ? collect(i:j) : [i:ψ.n; 1:j]
    Γs = ψ.Γ[inds]
    λl = ψ.λ[mod(i-2,ψ.n)+1]
    Γs, λs = tensor_applygate!(G, Γs, λl; maxdim, cutoff, renormalize)
    push!(λs, ψ.λ[j])
    for i in eachindex(inds) 
        ψ[inds[i]] = Γs[i], λs[i]
    end
    return ψ
end

#---------------------------------------------------------------------------------------------------
# Multi-Site Operators
#---------------------------------------------------------------------------------------------------
"""
    convert_operator(mat, d, n)

Convert dⁿ×dⁿ matrix that's compatible to the column-major convention.
"""
function convert_operator(mat::AbstractMatrix, d::Integer, n::Integer)
    tensor = reshape(mat, fill(d, 2n)...)
    perm = [n:-1:1; 2n:-1:n+1]
    tensor = permutedims(tensor, perm)
    reshape(tensor, size(mat))
end
