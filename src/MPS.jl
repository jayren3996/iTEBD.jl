#---------------------------------------------------------------------------------------------------
# MPS type
#---------------------------------------------------------------------------------------------------
export iMPS
struct iMPS{TΓ<:Number, Tλ<:Number}
    Γ::Vector{Array{TΓ, 3}}
    λ::Vector{Vector{Tλ}}
    n::Integer
end
#---------------------------------------------------------------------------------------------------
function iMPS(Γ::AbstractVector{<:AbstractArray{<:Number, 3}})
    n = length(Γ)
    λ = [ones(size(Γi, 3)) for Γi in Γ]
    iMPS(Γ, λ, n)
end
#---------------------------------------------------------------------------------------------------
export rand_iMPS
function rand_iMPS(
    n::Integer,
    d::Integer,
    dim::Integer
)
    Γ = [rand(dim, d, dim) for i=1:n]
    λ = [ones(dim) for i=1:n]
    iMPS(Γ, λ, n)
end
#---------------------------------------------------------------------------------------------------
function conj(mps::iMPS)
    Γ, λ, n = mps.Γ, mps.λ, mps.n
    iMPS(conj.(Γ), λ, n)
end
#---------------------------------------------------------------------------------------------------
function canonical(mps::iMPS)
    Γ, n = mps.Γ, mps.n
    Γ, λ = canonical(Γ)
    iMPS(Γ, λ, n)
end
#---------------------------------------------------------------------------------------------------
function applygate!(
    G::AbstractMatrix{<:Number},
    mps::iMPS;
    renormalize::Bool=false,
    bound::Int64=BOUND,
    tol::Float64=SVDTOL
)
    Γ, λ, n = mps.Γ, mps.λ, mps.n
    Γ, λ = applygate!(G, Γ, λ[n], renormalize=renormalize, bound=bound, tol=tol)
    iMPS(Γ, λ, n)
end
#---------------------------------------------------------------------------------------------------
gtrm(mps1::iMPS, mps2::iMPS) = gtrm(mps1.Γ, mps2.Γ)