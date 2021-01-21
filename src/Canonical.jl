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
    A, λ = canonical(T, renormalize=renormalize, bound=bound, tol=tol)
    tensor_lmul!(λ, A)
    tensor_decomp!(A, λ, n, renormalize=renormalize, bound=bound, tol=tol)
end
#---------------------------------------------------------------------------------------------------
# Block decomposition
#---------------------------------------------------------------------------------------------------
hermitianize(mat::AbstractMatrix) = Hermitian( (1+1im) * mat + (1-1im) * mat' )
function fixed_point(Γ::AbstractArray{<:Number, 3})
    α = size(Γ, 1)
    trans_mat = trm(Γ)
    vals, vecs = eigen(trans_mat)
    pos = argmax(real.(vals))
    fixed_point = vecs[:, pos]
    fixed_point_mat = reshape(fixed_point, α, α)
    hermitianize(fixed_point_mat)
end
#---------------------------------------------------------------------------------------------------
function right_cannonical(
    Γ::AbstractArray{<:Number, 3};
    tol::AbstractFloat=SORTTOL
)
    fixed_mat = fixed_point(Γ)
    vals, vecs = eigen(fixed_mat)
    pos = sum(vals .< tol)
    if pos == length(vals)
        vals *= -1
        pos = sum(vals .> tol)
    end
    if pos == 0 || pos == length(vals)
        X = vecs * Diagonal(sqrt.(vals))
        Xi = inv(X)
        @tensor Γ_new[:] := Xi[-1,1] * Γ[1,-2,2] * X[2,-3]
        return [Γ_new]
    else
        p1, p2 = vecs[:, 1:pos], vecs[:, pos+1:end]
        @tensor Γ1[:] := p1'[-1,1] * Γ[1,-2,2] * p1[2,-3]
        @tensor Γ2[:] := p2'[-1,1] * Γ[1,-2,2] * p2[2,-3]
        return vcat(right_cannonical(Γ1, tol=tol), right_cannonical(Γ2, tol=tol))
    end
end
#---------------------------------------------------------------------------------------------------
function block_decomp(
    Γ::AbstractArray{<:Number, 3};
    tol::AbstractFloat=SORTTOL
)
    α = size(Γ, 1)
    rvec = reshape(I(α) , α^2)
    trans_mat = trm(Γ)
    trans_mat_i = trans_mat - (trans_mat * rvec) * rvec' / α
    vals, vecs = eigen(trans_mat_i)
    pos = argmin( abs.(vals .- 1) )
    
    if abs(vals[pos] - 1) > tol
        return [Γ]
    else
        fixed_mat = hermitianize(reshape(vecs[:, pos], α, α))
        vals, vecs = eigen(fixed_mat)
        pos = sum(maximum(vals) .- vals .< tol)
        p1, p2 = vecs[:, 1:pos], vecs[:, pos+1:end]
        @tensor Γ1[:] := p1'[-1,1] * Γ[1,-2,2] * p1[2,-3]
        @tensor Γ2[:] := p2'[-1,1] * Γ[1,-2,2] * p2[2,-3]
        return vcat(block_decomp(Γ1, tol=tol), block_decomp(Γ2, tol=tol))
    end
end
#---------------------------------------------------------------------------------------------------
function block_canonical(
    Γ::AbstractArray{<:Number, 3};
    tol::AbstractFloat=SORTTOL
)
    Γs = right_cannonical(Γ)
    vcat((block_decomp(Γi) for Γi in Γs)...)
end

function block_canonical(
    Γs::AbstractVector{<:AbstractArray{<:Number, 3}};
    bound::Integer=BOUND,
    tol::AbstractFloat=SVDTOL
)
    n = length(Γs)
    Γ = tensor_group(Γs)
    Ts = block_canonical(Γ)
    n_block = length(Ts)
    ΓS = Vector{Vector{eltype(Ts[1])}}(undef, n_block)
    ΛS = Vector{Vector{Int64}}(undef, n_block)
    for i=1:n_block
        T = Ts[i]
        A, λ = canonical(T)
        tensor_lmul!(λ, A)
        ΓS[i], ΛS[i] = tensor_decomp!(A, λ, n, renormalize=renormalize, bound=bound, tol=tol)
    end
    ΓS, ΛS
end