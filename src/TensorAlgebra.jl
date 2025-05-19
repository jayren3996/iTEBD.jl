#---------------------------------------------------------------------------------------------------
# Basic Tensor Multiplication
#---------------------------------------------------------------------------------------------------
"""
    tensor_lmul!(λ, Γ)

Contraction of:
       |
  --λ--Γ--
"""
function tensor_lmul!(λ::AbstractVector{<:Number}, Γ::AbstractArray)
    α = size(Γ, 1)
    Γ_reshaped = reshape(Γ, α, :)
    lmul!(Diagonal(λ), Γ_reshaped)
end
#---------------------------------------------------------------------------------------------------
"""
    tensor_rmul!(Γ, λ)

Contraction of:
    |
  --Γ--λ--
"""
function tensor_rmul!(Γ::AbstractArray, λ::AbstractVector{<:Number})
    β = size(Γ)[end]
    Γ_reshaped = reshape(Γ, :, β)
    rmul!(Γ_reshaped, Diagonal(λ))
end
#---------------------------------------------------------------------------------------------------
"""
    tensor_umul(umat, Γ) 

Contraction of:
    |
    U
    |
  --Γ--
"""
function tensor_umul(umat::AbstractMatrix, Γ::AbstractArray{<:Number, 3})
    @tensor Γ_new[:] := umat[-2,1] * Γ[-1,1,-3]
    Γ_new
end

#---------------------------------------------------------------------------------------------------
# Tensor Grouping
#---------------------------------------------------------------------------------------------------
"""
    tensor_group_2(ΓA, ΓB) 

Contraction: 
    |  |           |
  --Γ--Γ--  ==>  --Γs--
"""
function tensor_group_2(ΓA::AbstractArray{<:Number, 3}, ΓB::AbstractArray{<:Number, 3})
    α, d1, χ = size(ΓA)
    d2 = size(ΓB, 2)
    β = size(ΓB, 3)
    tensor = reshape(ΓA, α*d1, χ) * reshape(ΓB, χ, d2*β)
    reshape(tensor, α, d1*d2, β)
end
#---------------------------------------------------------------------------------------------------
"""
    tensor_group_3(ΓA, ΓB, ΓC) 

Contraction: 
    |  |  |           |
  --Γ--Γ--Γ--  ==>  --Γs--
"""
function tensor_group_3(ΓA::AbstractArray, ΓB::AbstractArray, ΓC::AbstractArray)
    α, β = size(ΓA, 1), size(ΓC, 3)
    χ1, χ2 = size(ΓA, 3), size(ΓB, 3)
    tensor = reshape(ΓA, :, χ1) * reshape(ΓB, χ1, :)
    tensor = reshape(tensor, :, χ2) * reshape(ΓC, χ2, :)
    reshape(tensor, α, :, β)
end
#---------------------------------------------------------------------------------------------------
"""
    tensor_group(Γs) 

Contraction:
    |  |   ...   |           |
  --Γ--Γ-- ... --Γ--  ==>  --Γs--
"""
function tensor_group(Γs::AbstractVector{<:AbstractArray{<:Number, 3}})
    tensor = Γs[1]
    n = length(Γs)
    for i in 2:n
        χ = size(Γs[i], 1) 
        tensor = reshape(tensor, :, χ) * reshape(Γs[i], χ, :)
    end
    α, β = size(Γs[1], 1), size(Γs[n], 3)
    reshape(tensor, α, :, β)
end

#---------------------------------------------------------------------------------------------------
# Tensor Decomposition
#---------------------------------------------------------------------------------------------------
"""
    svd_trim(mat; maxdim, cutoff, renormalize)

SVD with compression.

Parameters:
-----------
- mat        : matrix 
- maxdim     : the maximum number of singular values to keep
- cutoff     : set the desired truncation error of the SVD
- renormalize: renormalize the singular values
"""
function svd_trim(
    mat::AbstractMatrix;
    maxdim::Integer=MAXDIM,
    cutoff::Real=SVDTOL,
    renormalize::Bool=false
)
    res = try
        svd(mat)  # Standard, faster path
    catch e
        if e isa LAPACKException
            # Use a robust fallback algorithm
            svd(mat; alg=LinearAlgebra.DivideAndConquer())
        else
            rethrow(e)  # rethrow if it’s an unexpected error
        end
    end
    vals = res.S
    (maxdim > length(vals)) && (maxdim = length(vals))
    len::Int64 = 1
    while true
        if isless(vals[len], cutoff)
            len -= 1
            break
        end
        isequal(len, maxdim) && break
        len += 1
    end
    U = res.U[:, 1:len]
    S = vals[1:len]
    V = res.Vt[1:len, :]
    if renormalize
        S ./= norm(S)
    end
    U, S, V
end
#---------------------------------------------------------------------------------------------------
"""
    tensor_svd(T; maxdim, curoff, renormalize)

Tensor SVD with compression:
    |   |           |     |
  --BLOCK--  ==>  --U--S--V--

Parameters:
-----------
- T          : 4-leg tensor
- maxdim     : the maximum number of singular values to keep
- cutoff     : set the desired truncation error of the SVD
- renormalize: renormalize the singular values
"""
function tensor_svd(
    T::AbstractArray{<:Number, 4};
    maxdim=MAXDIM, cutoff=SVDTOL, renormalize=false
)
    α, d1, d2, β = size(T)
    mat = reshape(T, α*d1, :)
    u, S, v = svd_trim(mat; maxdim, cutoff, renormalize)
    U = reshape(u, α, d1, :)
    V = reshape(v, :, d2, β)
    U, S, V
end
#---------------------------------------------------------------------------------------------------
"""
    tensor_decomp!(Γ, λl, n; maxdim, cutoff, renormalize)

Multiple decomposition:
    |              |   |        |
  --Γ--  ==> --λl--Γ₁--Γ₂-- ⋯ --Γₙ--
where:
    |          |
  --Γₙ--  =  --Aₙ--λₙ--

Return list tensor list [Γ₁,⋯,Γₙ], and values list [λ₁,⋯,λₙ₋₁].
"""
function tensor_decomp!(
    Γ::AbstractArray{<:Number, 3},
    λl::AbstractVector{<:Real},
    n::Integer;
    maxdim=MAXDIM, cutoff=SVDTOL, renormalize=false
)
    β = size(Γ, 3)
    d = round(Int, size(Γ, 2)^(1/n))
    Γs = Vector{Array{eltype(Γ), 3}}(undef, n)
    λs = Vector{Vector{eltype(λl)}}(undef, n-1)
    Ti, λi = Γ, λl
    for i=1:n-2
        Ti_reshaped = reshape(Ti, size(Ti,1), d, :, β)
        Ai, Λ, Ti = tensor_svd(Ti_reshaped; maxdim, cutoff, renormalize)
        tensor_lmul!(1 ./ λi, Ai)
        tensor_rmul!(Ai, Λ)
        tensor_lmul!(Λ, Ti)
        Γs[i] = Ai
        λs[i] = Λ
        λi = Λ
    end
    Ti_reshaped = reshape(Ti, size(Ti,1), d, :, β)
    Ai, Λ, Ti = tensor_svd(Ti_reshaped; maxdim, cutoff, renormalize)
    tensor_lmul!(1 ./ λi, Ai)
    tensor_rmul!(Ai, Λ)
    Γs[n-1] = Ai
    λs[n-1] = Λ
    Γs[n] = Ti
    Γs, λs
end



#---------------------------------------------------------------------------------------------------
# General transfer matrix
#---------------------------------------------------------------------------------------------------
"""
    gtrm(T1, T2)

General transfer matrix:
  2 ---Ā--- 4
       |
  1 ---B--- 3
"""
function gtrm(T1::AbstractArray{<:Number, 3}, T2::AbstractArray{<:Number, 3})
    i1, _, k1 = size(T2)
    i2, _, k2 = size(T1)
    T1c = conj(T1)
    ctype = promote_type(eltype(T1c), eltype(T2))
    transfer_mat = Array{ctype, 4}(undef, i1, i2, k1, k2)
    @tensor transfer_mat[:] = T1c[-2,1,-4] * T2[-1,1,-3]
    reshape(transfer_mat, i1*i2, :)
end
#---------------------------------------------------------------------------------------------------
"""
    gtrm(T1s, T2s)

General transfer matrix:
  2 ---Ā₁---Ā₂- ⋯ -Āₙ--- 4
       |    |      |
  1 ---B₁---B₂- ⋯ -Bₙ--- 3
"""
function gtrm(
    T1s::AbstractVector{<:AbstractArray{<:Number, 3}},
    T2s::AbstractVector{<:AbstractArray{<:Number, 3}}
)
    n = length(T1s)
    M = gtrm(T1s[1], T2s[1])
    for i in 2:n
        M = M * gtrm(T1s[i], T2s[i])
    end
    M
end
#---------------------------------------------------------------------------------------------------
"""
    trm(T::AbstractArray{<:Number, 3})

Transfer matrix for MPS tensor `T`.
"""
function trm(T::AbstractArray{<:Number, 3}) 
    gtrm(T, T)
end
#---------------------------------------------------------------------------------------------------
"""
    otrm(T1::AbstractArray, O::AbstractMatrix, T2::AbstractArray)

Operator transfer matrix
  2 ---Ā--- 4
       |
       O
       |
  1 ---B--- 3
"""
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
"""
    otrm(T1s::AbstractVector, O::AbstractMatrix, T2s::AbstractVector)

Operator transfer matrix
  2 ---Ā₁---Ā₂- ⋯ -Āₙ--- 4
       |    |      |
       O    O   ⋯  O 
       |    |      |
  1 ---B₁---B₂- ⋯ -Bₙ--- 3
"""
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
"""
    otrm(T1s::AbstractVector, Os::AbstractVector, T2s::AbstractVector)

Operator transfer matrix
  2 ---Ā₁---Ā₂- ⋯ -Āₙ--- 4
       |    |      |
       O₁   O₂  ⋯  Oₙ
       |    |      |
  1 ---B₁---B₂- ⋯ -Bₙ--- 3
"""
function otrm(
    T1s::AbstractVector{<:AbstractArray{<:Number, 3}},
    Os::AbstractVector{<:AbstractMatrix{<:Number}},
    T2s::AbstractVector{<:AbstractArray{<:Number, 3}}
)
    n = length(T1s)
    M = otrm(T1s[1], Os[1], T2s[1])
    for i=2:n
        M = M * otrm(T1s[i], Os[i], T2s[i])
    end
    M
end

#---------------------------------------------------------------------------------------------------
# Normalization
#---------------------------------------------------------------------------------------------------
function tensor_renormalize!(Γ::Array{<:Number, 3})
    Γ_norm = sqrt(inner_product(Γ))
    Γ ./= Γ_norm
end

