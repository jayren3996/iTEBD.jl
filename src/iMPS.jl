#---------------------------------------------------------------------------------------------------
# iMPS 
#---------------------------------------------------------------------------------------------------
export iMPS
"""
    iMPS

Infinite MPS. 

Parameters:
- Γ : Vector of tensors.
- λ : Vector of Schmidt values.
- n : Number of tensors in the periodic blocks.

Note that the tensor `Γ` has absorbed the `λ` in, so it's in right canonical form.  
"""
struct iMPS{T<:Number}
    Γ::Vector{Array{T, 3}}
    λ::Vector{Vector{Float64}}
    n::Int64
end
#---------------------------------------------------------------------------------------------------
function iMPS(
    T::DataType,
    Γs::AbstractVector{<:AbstractArray{<:Number, 3}}
)
    n = length(Γs)
    Γ = Array{T}.(Γs)
    λ = [ones(Float64, size(Γi, 3)) for Γi in Γs]
    iMPS(Γ, λ, n)
end
#---------------------------------------------------------------------------------------------------
function iMPS(Γs::AbstractVector{<:AbstractArray{<:Number, 3}})
    type_list = eltype.(Γs)
    T = promote_type(type_list...)
    iMPS(T, Γs)
end

#---------------------------------------------------------------------------------------------------
# INITIATE MPS
#---------------------------------------------------------------------------------------------------
export rand_iMPS
function rand_iMPS(
    T::DataType,
    n::Integer,
    d::Integer,
    dim::Integer
)
    Γ = [rand(T, dim, d, dim) for i=1:n]
    λ = [ones(dim) for i=1:n]
    iMPS(Γ, λ, n)
end
rand_iMPS(n, d, dim) = rand_iMPS(Float64, n, d, dim)
#---------------------------------------------------------------------------------------------------
export product_iMPS
function product_iMPS(
    T::DataType,
    v::AbstractVector{<:AbstractVector{<:Number}}
)
    n = length(v)
    d = length(v[1])
    Γ = [zeros(T, 1, d, 1) for i=1:n]
    λ = [ones(1) for i=1:n]
    for i=1:n
        Γ[i][1,:,1] .= v[i]
    end
    iMPS(Γ, λ, n)
end
#---------------------------------------------------------------------------------------------------
function product_iMPS(v::AbstractVector{<:AbstractVector{<:Number}})
    T = promote_type(eltype.(v)...)
    product_iMPS(T, v)
end
#---------------------------------------------------------------------------------------------------
function canonical!(ψ::iMPS; maxdim=MAXDIM, cutoff=SVDTOL, renormalize=true)
    ψ.Γ[:], ψ.λ[:] = schmidt_canonical(ψ.Γ; maxdim, cutoff, renormalize)
    ψ
end

#---------------------------------------------------------------------------------------------------
# BASIC PROPERTIES
#---------------------------------------------------------------------------------------------------
get_data(mps::iMPS) = mps.Γ, mps.λ, mps.n
eltype(::iMPS{T}) where T = T
#---------------------------------------------------------------------------------------------------
function getindex(mps::iMPS, i::Integer)
    i = mod(i-1, mps.n) + 1
    mps.Γ[i], mps.λ[i]
end
#---------------------------------------------------------------------------------------------------
function setindex!(
    mps::iMPS, 
    v::Tuple{<:AbstractArray{<:Number, 3}, <:AbstractVector{<:Real}},
    i::Integer
)
    i = mod(i-1, mps.n) + 1
    mps.Γ[i] = v[1]
    mps.λ[i] = v[2]
end
#---------------------------------------------------------------------------------------------------
function mps_promote_type(
    T::DataType,
    mps::iMPS
)
    Γ, λ, n = get_data(mps)
    Γ_new = Array{T}.(Γ)
    iMPS(Γ_new, λ, n)
end
#---------------------------------------------------------------------------------------------------
function ent_S(mps::iMPS, i::Integer)
    j = mod(i-1, mps.n) + 1
    ρ = mps.λ[j] .^ 2
    entanglement_entropy(ρ)
end
#---------------------------------------------------------------------------------------------------
function expect(ψ::iMPS, O::AbstractMatrix, i::Integer, j::Integer)
    Γ = ψ.Γ[j>=i ? (i:j) : [i:ψ.n; 1:j]]
    λl = ψ.λ[mod(i-2,ψ.n)+1]
    ocontract(Γ, O, λl) |> real
end


#---------------------------------------------------------------------------------------------------
# MANIPULATION
#---------------------------------------------------------------------------------------------------
function conj(mps::iMPS)
    Γ, λ, n = get_data(mps)
    iMPS(conj.(Γ), λ, n)
end

#---------------------------------------------------------------------------------------------------
gtrm(mps1::iMPS, mps2::iMPS) = gtrm(mps1.Γ, mps2.Γ)


