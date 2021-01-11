#---------------------------------------------------------------------------------------------------
# Factorization
#---------------------------------------------------------------------------------------------------
function matsqrt(mat::AbstractMatrix{<:Number})
    vals, vecs = eigen(Hermitian(mat*mat'))
    vals_sqrt = Diagonal(sqrt.(sqrt.(vals)))
    vecs * vals_sqrt
end
#---------------------------------------------------------------------------------------------------
# Canonical form
#---------------------------------------------------------------------------------------------------
export canonical
function canonical(
    tensor::AbstractArray{<:Number,3};
    renormalize::Bool=false,
    bound::Integer=BOUND,
    tol::AbstractFloat=SVDTOL
)
    emax, rvec, lvec = dominent_eigvecs(trm(tensor))
    X, Yt = begin
        α, d, β = size(tensor)
        rmat = reshape(rvec, α, :)
        lmat = reshape(lvec, β, :)
        matsqrt(rmat), transpose(matsqrt(lmat))
    end
    U, S, V = begin
        res = svd(Yt * X)
        len = if bound==0 
            sum(res.S .> tol)
        else
            min(sum(res.S .> tol), bound)
        end
        s = if renormalize
            normalize(res.S[1:len])
        else
            res.S[1:len]
        end
        u = res.U[:, 1:len]
        v = res.Vt[1:len, :]
        u, s, v
    end
    canonicalT = begin
        lmat = V / X
        rmat = (Yt \ U) * Diagonal(S)
        ctype = promote_type(eltype(lmat), eltype(rmat), eltype(tensor))
        temp = Array{ctype}(undef, size(tensor))
        @tensor temp[:] = lmat[-1,1] * tensor[1,-2,2] * rmat[2,-3]
        renormalize ? temp / sqrt(emax) : temp
    end
    canonicalT, S
end
#---------------------------------------------------------------------------------------------------
function canonical(
    Ts::AbstractVector{<:AbstractArray{<:Number, 3}};
    renormalize::Bool=false,
    bound::Integer=BOUND,
    tol::AbstractFloat=SVDTOL
)
    n = length(Ts)
    T = tensor_group(Ts)
    A, λ = canonical(T)
    tensor_lmul!(λ, A)
    tensor_decomp!(A, λ, n, renormalize=renormalize, bound=bound, tol=tol)
end
