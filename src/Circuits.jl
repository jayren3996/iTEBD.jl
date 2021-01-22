#---------------------------------------------------------------------------------------------------
# Basic Tensor Multiplication
#
# 1. tensor_lmul!:
#      |
# --λ--Γ--
#
# 2. tensor_lmul!:
#   |
# --Γ--λ--
#
# 3. tensor_umul:
#     |
#     U
#     |
# ----Γ----
#---------------------------------------------------------------------------------------------------
function tensor_lmul!(λ::AbstractVector{<:Number}, Γ::AbstractArray)
    α = size(Γ, 1)
    Γ_reshaped = reshape(Γ, α, :)
    lmul!(Diagonal(λ), Γ_reshaped)
end
#---------------------------------------------------------------------------------------------------
function tensor_rmul!(Γ::AbstractArray, λ::AbstractVector{<:Number})
    β = size(Γ)[end]
    Γ_reshaped = reshape(Γ, :, β)
    rmul!(Γ_reshaped, Diagonal(λ))
end
#---------------------------------------------------------------------------------------------------
function tensor_umul(umat::AbstractMatrix, Γ::AbstractArray{<:Number, 3})
    α, d, β = size(Γ)
    ctype = promote_type(eltype(umat), eltype(Γ))
    Γ_new = Array{ctype, 3}(undef, α, d, β)
    @tensor Γ_new[:] = umat[-2,1] * Γ[-1,1,-3]
    Γ_new
end

#---------------------------------------------------------------------------------------------------
# Tensor Grouping
#
# 1. tensor_group_2:
#   |  |           |
# --Γ--Γ--  ==>  --Γs--
#
# 2. tensor_group_3:
#   |  |  |           |
# --Γ--Γ--Γ--  ==>  --Γs--
#
# 3. tensor_group:
#   |  |   ...   |           |
# --Γ--Γ-- ... --Γ--  ==>  --Γs--
#---------------------------------------------------------------------------------------------------
function tensor_group_2(ΓA::AbstractArray{T, 3}, ΓB::AbstractArray{T, 3}) where T<:Number
    α = size(ΓA, 1)
    d = size(ΓA, 2)
    β = size(ΓB, 3)
    Γ_contracted = Array{T}(undef, α, d, d, β)
    @tensor Γ_contracted[:] = ΓA[-1,-2,1] * ΓB[1,-3,-4]
    reshape(Γ_contracted, α, :, β)
end
#---------------------------------------------------------------------------------------------------
function tensor_group_3(ΓA::AbstractArray{T, 3}, ΓB::AbstractArray{T, 3}, ΓC::AbstractArray{T, 3}) where T<:Number
    α = size(ΓA, 1)
    d = size(ΓA, 2)
    β = size(ΓC, 3)
    Γ_contracted = Array{T}(undef, α, d, d, d, β)
    @tensor Γ_contracted[:] = ΓA[-1,-2,1] * ΓB[1,-3,2] * ΓC[2,-4,-5]
    reshape(Γ_contracted, α, :, β)
end
#---------------------------------------------------------------------------------------------------
function tensor_group(Γs::AbstractVector{<:AbstractArray{T, 3}}) where T<:Number
    number = length(Γs)
    if number == 2
        return tensor_group_2(Γs[1], Γs[2])
    elseif number == 3
        return tensor_group_3(Γs[1], Γs[2], Γs[3])
    end
    α = size(Γs[1], 1)
    β = size(Γs[number], 3)
    index = begin
        index_temp = [[i, -i-1, i+1] for i=1:number]
        index_temp[1][1] = -1
        index_temp[number][3] = -number-2
        index_temp
    end
    Γ_contracted = ncon(Γs, index)::Array{T, number+2}
    reshape(Γ_contracted, α, :, β)
end

#---------------------------------------------------------------------------------------------------
# Tensor Decomposition
#
# 1. tensor_svd:
#   |   |           |      |
# --BLOCK--  ==>  --ΓA--λ--ΓB--
#
# 2. tensor_decomp:
#      |          |       |        ...   |
# --λ--Γ--  ==> --Γ1--λ1--Γ2--λ2-- ... --Γn--λn--
#---------------------------------------------------------------------------------------------------
function tensor_svd(
    T::AbstractArray{<:Number, 4};
    renormalize::Bool=false,
    bound::Integer=BOUND,
    tol::AbstractFloat=SVDTOL
)
    """
    renormalize : controls whether to normalize singular values.
    bound       : maximal number of singular values.
    tol         : minimal value of singular values.
    """
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
# Apply Gate
#
#      |
#      G
#      |          |       |        ...   |
# --λ--Γ--  ==> --Γ1--λ1--Γ2--λ2-- ... --Γn--λn--
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
# General transfer matrix
# 
# 2 ---A*-- 4
#      |
# 1 ---A--- 3
#---------------------------------------------------------------------------------------------------
function gtrm(
    T1::AbstractArray{<:Number, 3},
    T2::AbstractArray{<:Number, 3}
)
    i1, j1, k1 = size(T2)
    i2, j2, k2 = size(T1)
    T1c = conj(T1)
    ctype = promote_type(eltype(T1c), eltype(T2))
    transfer_mat = Array{ctype, 4}(undef, i1, i2, k1, k2)
    @tensor transfer_mat[:] = T1c[-2,1,-4] * T2[-1,1,-3]
    reshape(transfer_mat, i1*i2, :)
end
#---------------------------------------------------------------------------------------------------
function gtrm(
    T1s::AbstractVector{<:AbstractArray{<:Number, 3}},
    T2s::AbstractVector{<:AbstractArray{<:Number, 3}}
)
    n = length(T1s)
    M = gtrm(T1s[1], T2s[1])
    for i=2:n
        M = M * gtrm(T1s[i], T2s[i])
    end
    M
end
#---------------------------------------------------------------------------------------------------
trm(T) = gtrm(T, T)
#---------------------------------------------------------------------------------------------------
# Operator transfer matrix
#
# 2 ---A*-- 4
#      |
#      O
#      |
# 1 ---A--- 3
#---------------------------------------------------------------------------------------------------
function otrm(
    T1::AbstractArray{<:Number, 3},
    O::AbstractMatrix{<:Number},
    T2::AbstractArray{<:Number, 3}
)
    i1, j1, k1 = size(T2)
    i2, j2, k2 = size(T1)
    T1c = conj(T1)
    ctype = promote_type(eltype(T1c), eltype(O), eltype(T2))
    transfer_mat = Array{ctype, 4}(undef, i1, i2, k1, k2)
    @tensor transfer_mat[:] = T1c[-2,1,-4] * O[1,2] * T2[-1,2,-3]
    reshape(transfer_mat, i1*i2, :)
end
#---------------------------------------------------------------------------------------------------
function otrm(
    T1s::AbstractVector{<:AbstractArray{<:Number, 3}},
    O::AbstractMatrix{<:Number},
    T2s::AbstractVector{<:AbstractArray{<:Number, 3}}
)
    n = length(T1s)
    M = otrm(T1s[1], O, T2s[1])
    for i=2:n
        M = M * otrm(T1s[i], O, T2s[i])
    end
    M
end
#---------------------------------------------------------------------------------------------------
function otrm(
    T1s::AbstractVector{<:AbstractArray{<:Number, 3}},
    O::Vector{<:AbstractMatrix{<:Number}},
    T2s::AbstractVector{<:AbstractArray{<:Number, 3}}
)
    n = length(T1s)
    M = otrm(T1s[1], O[1], T2s[1])
    for i=2:n
        M = M * otrm(T1s[i], O[i], T2s[i])
    end
    M
end