#---------------------------------------------------------------------------------------------------
# MPS TYPE
#
# Parameters:
# Γ : Vector of tensors.
# λ : Vector of Schmidt values.
# n : Number of tensors in the periodic blocks.
#---------------------------------------------------------------------------------------------------
export iMPS
"""
    iMPS(Γ, λ, n)

iMPS type 

# Parameters:

Γ : Vector of tensors.

λ : Vector of Schmidt values.

n : Number of tensors in the periodic blocks.
"""
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
# INITIATE MPS
#
# 1. rand_iMPS    : Randomly generate iMPS with given bond dimension.
# 2. product_iMPS : Return iMPS from product state.
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
# BASIC MANIPULATION
#
# 1. conj       : Complex conjugation of iMPS.
# 2. applygate! : Apply gate to iMPS, return the result. The initial one will be altered.
# 3. gtrm       : Transfer matrix of the block periodic iMPS.
# 4. entropy    : Return entanglement entropy across bond i.
#---------------------------------------------------------------------------------------------------
function conj(mps::iMPS)
    Γ, λ, n = mps.Γ, mps.λ, mps.n
    iMPS(conj.(Γ), λ, n)
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
function applygate!(
    G::AbstractMatrix{<:Number},
    mps::iMPS,
    inds::AbstractVector{<:Integer};
    renormalize::Bool=false,
    bound::Int64=BOUND,
    tol::Float64=SVDTOL
)
    n = mps.n
    inds = mod.(inds .- 1, n) .+ 1
    indl = mod((inds[1]-2), n) + 1
    indr = inds[end]
    Γs = mps.Γ[inds]
    λl = mps.λ[indl]
    λr = mps.λ[indr]
    Γs_new, λs_new = applygate!(G, Γs, λl, λr, renormalize=renormalize, bound=bound, tol=tol)
    mps.Γ[inds] .= Γs_new
    mps.λ[inds] .= λs_new
    mps
end
#---------------------------------------------------------------------------------------------------
function mps_promote_type(
    T::DataType,
    mps::iMPS
)
    Γ, λ, n = mps.Γ, mps.λ, mps.n
    Γ_new = Array{T}.(Γ)
    iMPS(Γ_new, λ, n)
end
#---------------------------------------------------------------------------------------------------
gtrm(mps1::iMPS, mps2::iMPS) = gtrm(mps1.Γ, mps2.Γ)
#---------------------------------------------------------------------------------------------------
export entropy
function entropy(
    mps::iMPS,
    i::Integer
)
    j = mod(i-1, mps.n) + 1
    λj = mps.λ[j].^2
    entanglement_entropy(λj)
end

#---------------------------------------------------------------------------------------------------
# CANONICAL FORMS
#---------------------------------------------------------------------------------------------------
export canonical
function canonical(
    mps::iMPS;
    trim::Bool=false,
    bound::Integer=BOUND,
    tol::Real=SVDTOL
)
    Γ, n = mps.Γ, mps.n
    Γ, λ = if trim
        canonical_trim(Γ, bound=bound, tol=tol)
    else
        schmidt_canonical(Γ, bound=bound, tol=tol)
    end
    iMPS(Γ, λ, n)
end
