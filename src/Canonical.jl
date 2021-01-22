#---------------------------------------------------------------------------------------------------
# Canonical Form With No Degeneracy
#---------------------------------------------------------------------------------------------------
function matsqrt(mat::AbstractMatrix{<:Number})
    vals, vecs = eigen(Hermitian(mat))
    vals_sqrt = Diagonal(sqrt.(vals))
    vecs * vals_sqrt
end
#---------------------------------------------------------------------------------------------------
export canonical
function canonical(
    tensor::AbstractArray{<:Number,3};
    renormalize::Bool=false,
    bound::Integer=BOUND,
    tol::AbstractFloat=SVDTOL
)
    transfer_mat = trm(tensor)
    emax, rvec = krylov_eigen(transfer_mat)
    emax, lvec = krylov_eigen(transpose(transfer_mat))
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
# Block Decomposition With Degeneracy
#---------------------------------------------------------------------------------------------------
function fixed_point(
    Γ::AbstractArray{<:Number, 3};
    tol::AbstractFloat=SVDTOL,
    max_itr::Integer=10000
)
    α = size(Γ, 1)
    trans_mat = trm(Γ)
    val, vec = krylov_eigen(trans_mat, tol=tol, max_itr=max_itr)
    val, Hermitian(reshape(vec, α, α))
end
#---------------------------------------------------------------------------------------------------
function right_cannonical(
    Γ::AbstractArray{<:Number, 3};
    tol::AbstractFloat=SORTTOL
)
    Γnorm, fixed_mat = fixed_point(Γ)
    if Γnorm < SORTTOL
        #println("RC counter zero block")
        return [(0.0, Γ)]
    end
    vals, vecs = eigen(fixed_mat)
    pos = sum(vals .< tol)
    #println("$pos / $(length(vals))")
    if pos == 0
        #println("RC jump out: $pos")
        sqrtvals = sqrt.(vals)
        X = vecs * Diagonal(sqrtvals)
        Xi = Diagonal(1 ./ sqrtvals) * vecs'
        @tensor Γ_new[:] := Xi[-1,1] * Γ[1,-2,2] * X[2,-3]
        Γnorms = sqrt(Γnorm)
        Γ_new /= Γnorms
        return [(Γnorms, Γ_new)]
    else
        #println("RC split: $pos / $(length(vals))")
        p1, p2 = vecs[:, 1:pos], vecs[:, pos+1:end]
        @tensor Γ1[:] := p1'[-1,1] * Γ[1,-2,2] * p1[2,-3]
        @tensor Γ2[:] := p2'[-1,1] * Γ[1,-2,2] * p2[2,-3]
        return vcat(right_cannonical(Γ1, tol=tol), right_cannonical(Γ2, tol=tol))
    end
end
#---------------------------------------------------------------------------------------------------
hermitianize(mat) = Hermitian( (1+1im) * mat + (1-1im) * mat' )
function block_decomp(
    Γ::AbstractArray{<:Number, 3};
    tol::AbstractFloat=SORTTOL
)
    α = size(Γ, 1)
    trans_mat = trm(Γ)
    vals, vecs = eigen(trans_mat)
    pos = abs.(vals .- 1) .< tol 
    vecs = vecs[:, pos]
    #println("$(sum(pos)) / $(length(vals))")
    if size(vecs, 2) == 1
        #println("BD jump out: $(sum(pos))")
        return [Γ]
    elseif size(vecs, 2) == 0
        #println("BD counter zero-block")
        return [Γ]
    else
        #println("BD Start splitting: $(sum(pos)) / $(length(vals))")
        fixed_mat_2 = begin
            # check the dominent right vector.
            mat_temp = hermitianize(reshape(vecs[:, 1], α, α))
            id_mat = I(α) * mat_temp[1,1]
            # if the eigen vector is close to identity, choose another one
            if norm(id_mat - mat_temp) .< tol
                println("Counter id mat, chage one.")
                # Here I assume the second vector is good
                # !!! THE NUMERICAL STABILITY IS NOT GUARANTEED !!!
                mat_temp = hermitianize(reshape(vecs[:, 2], α, α))
            end
            mat_temp
        end
        vals, vecs = eigen(fixed_mat_2)
        pos = sum(maximum(vals) .- vals .< tol)
        #println("BD split: $pos / $(length(vals))")
        # eigenvalues is from small to large
        p1, p2 = vecs[:, 1:end-pos], vecs[:, end-pos+1:end]
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
    norm_list = []
    tensor_list = []
    for Γi in Γs
        normi = Γi[1]
        Γis = block_decomp(Γi[2])
        ni = length(Γis)
        norm_list = vcat(norm_list, fill(normi, ni))
        tensor_list = vcat(tensor_list, Γis)
    end
    norm_list, tensor_list
end