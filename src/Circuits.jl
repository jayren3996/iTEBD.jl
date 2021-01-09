#---------------------------------------------------------------------------------------------------
# Tensor Grouping
#---------------------------------------------------------------------------------------------------
function tensor_lmul!(
    λ::AbstractVector{<:Number},
    Γ::AbstractArray
)
    α = size(Γ, 1)
    Γ_reshaped = reshape(Γ, α, :)
    lmul!(Diagonal(λ), Γ_reshaped)
end
#---------------------------------------------------------------------------------------------------
function tensor_rmul!(
    Γ::AbstractArray,
    λ::AbstractVector{<:Number}
)
    β = size(Γ)[end]
    Γ_reshaped = reshape(Γ, :, β)
    rmul!(Γ_reshaped, Diagonal(λ))
end
#---------------------------------------------------------------------------------------------------
function tensor_umul(
    umat::AbstractMatrix,
    Γ::AbstractArray{<:Number, 3}
)
    α, d, β = size(Γ)
    ctype = promote_type(eltype(umat), eltype(Γ))
    Γ_new = Array{ctype, 3}(undef, α, d, β)
    @tensor Γ_new[:] = umat[-2,1] * Γ[-1,1,-3]
    Γ_new
end
#---------------------------------------------------------------------------------------------------
function tensor_group(
    Γs::AbstractVector{<:AbstractArray{T, 3}}
) where T<:Number
    number = length(Γs)
    index = begin
        index_temp = [[i, -i-1, i+1] for i=1:number]
        index_temp[1][1] = -1
        index_temp[number][3] = -number-2
        index_temp
    end
    Γ_contracted = ncon(Γs, index)::Array{T, number+2}
    α = size(Γs[1], 1)
    β = size(Γs[number], 3)
    reshape(Γ_contracted, α, :, β)
end
#---------------------------------------------------------------------------------------------------
tensor_group(Γs::AbstractArray{<:Number, 3}...) = tensor_group([Γs...])
#---------------------------------------------------------------------------------------------------
# Tensor Decomposition
#---------------------------------------------------------------------------------------------------
function tensor_svd(
    T::AbstractArray{<:Number, 4};
    renormalize::Bool=false,
    bound::Integer=BOUND,
    tol::AbstractFloat=SVDTOL
)
    α, d1, d2, β = size(T)
    U, S, V = begin
        svd_res = svd(reshape(T, α*d1, :))
        svd_res.U, svd_res.S, svd_res.Vt
    end
    len = bound==0 ? sum(S .> tol) : min(sum(S .> tol), bound)
    s = renormalize ? normalize(S[1:len]) : S[1:len]
    u = reshape(U[:, 1:len], α, d1, :)
    v = reshape(V[1:len, :], :, d2, β)
    u, s, v
end
#---------------------------------------------------------------------------------------------------
function tensor_decomp!(
    tensor::AbstractArray{<:Number, 3},
    λ::AbstractVector,
    n::Integer;
    renormalize::Bool=false,
    bound::Integer=BOUND,
    tol::AbstractFloat=SVDTOL
)
    β = size(tensor, 3)
    d = round(Int, size(tensor, 2)^(1/n))
    Ts = Vector{Array{eltype(tensor), 3}}(undef, n)
    λs = Vector{Vector{eltype(λ)}}(undef, n)
    Ti, λi = tensor, λ
    for i=1:n-2
        Ti = reshape(Ti, size(Ti,1), d, :, β)
        Ai, Λ, Ti = tensor_svd(Ti, renormalize=renormalize, bound=bound, tol=tol)
        tensor_lmul!(1 ./ λi, Ai)
        tensor_rmul!(Ai, Λ)
        Ts[i] = Ai
        λs[i] = Λ
        λi = Λ
        tensor_lmul!(λi, Ti)
    end
    Ti = reshape(Ti, size(Ti,1), d, :, β)
    Ai, Λ, Ti = tensor_svd(Ti, renormalize=renormalize, bound=bound, tol=tol)
    tensor_lmul!(1 ./ λi, Ai)
    tensor_rmul!(Ai, Λ)
    Ts[n-1] = Ai
    λs[n-1] = Λ
    Ts[n] = Ti
    λs[n] = λ
    Ts, λs
end
#---------------------------------------------------------------------------------------------------
# Gate
#---------------------------------------------------------------------------------------------------
function applygate!(
    G::AbstractMatrix{<:Number},
    Γ::AbstractVector{<:AbstractArray{<:Number, 3}},
    λ::AbstractVector{<:Number};
    renormalize::Bool=false,
    bound::Integer=BOUND,
    tol::AbstractFloat=SVDTOL
)
    n = length(Γ)
    ΓΓ = tensor_group(Γ)
    tensor_lmul!(λ, ΓΓ)
    GΓΓ = tensor_umul(G, ΓΓ)
    tensor_decomp!(GΓΓ, λ, n, renormalize=renormalize, bound=bound, tol=tol)
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
