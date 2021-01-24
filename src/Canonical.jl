#---------------------------------------------------------------------------------------------------
# Dominent eigensystem
#
# Dominent eigen system
# Krylov method ensure Hermitian and semi-positive.
#---------------------------------------------------------------------------------------------------
function dominent_eigen(mat::AbstractMatrix)
    vals, vecs = eigen(mat)
    vals_abs = real.(vals)
    pos = argmax(vals_abs)
    vals_abs[pos], vecs[:, pos]
end
#---------------------------------------------------------------------------------------------------
function dominent_eigval(mat::AbstractMatrix; sort="a")
    vals = eigvals(mat)
    if sort == "r"
        return maximum(real.(vals))
    elseif sort == "a"
        return maximum(abs.(vals))
    end
end

# inner product
export inner_product
inner_product(T) = dominent_eigval(trm(T))
inner_product(T1, T2) = dominent_eigval(gtrm(T1, T2))
#---------------------------------------------------------------------------------------------------
function krylov_eigen(
    mat::AbstractMatrix;
    tol::AbstractFloat=1e-7,
    max_itr::Integer=1000
)
    """
    Using krylov iteration
    The resulting dominent eigenvector for transfer matrix is always Hermitian and semi-positive.
    
    tol     : tolerace for norm deference.
    max_itr : maximal nuber of terations.
    """
    α = round(Int64, sqrt(size(mat, 1)))
    expmat = exp(mat)
    va = begin
        diag_vect = Array(reshape(I(α), α^2))
        normalize(expmat * diag_vect)
    end
    vb = normalize(expmat * va)
    vc = va - vb
    err = norm(vc)
    itr = 1
    while err > tol
        mul!(va, expmat, vb)
        normalize!(va)
        mul!(vb, expmat, va)
        normalize!(vb)
        @. vc = va - vb
        err = norm(vc)
        itr += 1
        # print a warning string and exit the loop if maximal iteration times is reached.
        if itr > max_itr
            vals = eigvals(mat)
            println("Krylov method failed to converge within maximum number of iterations.")
            println("eigval = $(norm(mat * va)) / $(norm(va)), error = $err")
            break;
        end
    end
    mul!(va, mat, vb)
    val = norm(va)
    val, vb
end
#---------------------------------------------------------------------------------------------------
function fixed_point(
    Γ::AbstractArray{<:Number, 3};
    tol::AbstractFloat=1e-7,
    max_itr::Integer=1000
)
    α = size(Γ, 1)
    trans_mat = trm(Γ)
    val, vec = krylov_eigen(trans_mat, tol=tol, max_itr=max_itr)
    val, Hermitian(reshape(vec, α, α))
end

#---------------------------------------------------------------------------------------------------
# Right Canonical Form
#
# - The algorithm will tensor that is right-normalized.
# - If a degeneracy is encuntered, there would be multiple outputs.
# - While it is NOT guaranteed that the outputs are non-degenerate.
#---------------------------------------------------------------------------------------------------
function right_cannonical(
    Γ::AbstractArray{<:Number, 3};
    tol::AbstractFloat=1e-20
)
    Γnorm, fixed_mat = fixed_point(Γ)
    if Γnorm < tol
        # Zero block
        return [(0.0, Γ)]
    end
    vals, vecs = eigen(fixed_mat)
    # dimension of null space
    pos = sum(vals .< tol)
    if pos == 0
        # fixed-mat is invertible
        sqrtvals = sqrt.(vals)
        X = vecs * Diagonal(sqrtvals)
        Xi = Diagonal(1 ./ sqrtvals) * vecs'
        @tensor Γ_new[:] := Xi[-1,1] * Γ[1,-2,2] * X[2,-3]
        Γnorms = sqrt(Γnorm)
        Γ_new ./= sqrt(inner_product(Γ_new))
        return [(Γnorms, Γ_new)]
    else
        # contain null space, split the tensor
        p1, p2 = vecs[:, pos+1:end], vecs[:, 1:pos]
        @tensor Γ1[:] := p1'[-1,1] * Γ[1,-2,2] * p1[2,-3]
        @tensor Γ2[:] := p2'[-1,1] * Γ[1,-2,2] * p2[2,-3]
        # recurence
        return vcat(right_cannonical(Γ1, tol=tol), right_cannonical(Γ2, tol=tol))
    end
end

#---------------------------------------------------------------------------------------------------
# Block Decomposition

# - This agorithm further decompose the right-renormalized tensor into right-renormalized tensors.
# - This algorithm ensure the outputs are non-degenerate.
#---------------------------------------------------------------------------------------------------
function block_split(
    Γ::AbstractArray{<:Number, 3},
    vecs::AbstractArray{<:Number, 2};
    tol::AbstractFloat=1e-5
)
    α = size(Γ, 1)
    fixed_mat = begin
        # check the dominent right vector.
        eigen_mat = reshape(vecs[:, 1], α, α)
        mat_temp = (1+1im) * eigen_mat + (1-1im) * eigen_mat'
        id_mat = I(α) * mat_temp[1,1]
        if norm(id_mat - mat_temp) < tol
            # The first eigen matrix is identity.
            # Here I assume in this case the second eigen matrix is good.
            eigen_mat = reshape(vecs[:, 2], α, α)
            mat_temp = (1+1im) * eigen_mat + (1-1im) * eigen_mat'
            id_mat = I(α) * mat_temp[1,1]
            @assert norm(id_mat - mat_temp) > tol "Second identity. Get trouble!!!"
        end
        Hermitian(mat_temp)
    end
    # Split the space.
    # The null space is spanned by the dominent eigen vector of fixed_mat_2
    vals, vecs = eigen(fixed_mat)
    pos = sum(maximum(vals) .- vals .< tol)
    # eigenvalues is from small to large
    p1, p2 = vecs[:, 1:end-pos], vecs[:, end-pos+1:end]
    @tensor Γ1[:] := p1'[-1,1] * Γ[1,-2,2] * p1[2,-3]
    @tensor Γ2[:] := p2'[-1,1] * Γ[1,-2,2] * p2[2,-3]
    Γ1, Γ2
end
#---------------------------------------------------------------------------------------------------
function block_decomp(
    Γ::AbstractArray{<:Number, 3};
    tol::AbstractFloat=1e-5
)
    α = size(Γ, 1)
    # compute eigen vectors with eigenvalue 1.
    vecs = begin
        trans_mat = trm(Γ)
        trm_vals, trm_vecs = eigen(trans_mat)
        pos = abs.(trm_vals .- 1) .< tol 
        trm_vecs[:, pos]
    end
    # Non-degenerate case:
    if size(vecs, 2) < 2
        return [Γ]
    end
    # Degenerate:
    Γ1, Γ2 = block_split(Γ, vecs, tol=tol)
    Γ1c = block_decomp(Γ1)
    Γ2c = block_decomp(Γ2)
    return [Γ1c; Γ2c]
end
#---------------------------------------------------------------------------------------------------
export block_canonical
function block_canonical(Γ::AbstractArray{<:Number, 3})
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

#---------------------------------------------------------------------------------------------------
# Schmidt Canonical Form
#
# - Given a right canonical form, return a Schmidt canonical form.
# - This algorithm assume there is no degeneracy.
#---------------------------------------------------------------------------------------------------
function schmidt_canonical(
    Γ::AbstractArray{<:Number,3};
    renormalize::Bool=false,
    bound::Integer=BOUND,
    tol::AbstractFloat=SVDTOL
)
    α = size(Γ, 1)
    lmat = begin
        trans_mat_T = transpose(trm(Γ))
        emax, lvec = krylov_eigen(trans_mat_T)
        reshape(lvec, α, :)
    end
    Yt = begin
        vals, vecs = eigen(Hermitian(lmat))
        Diagonal(sqrt.(vals)) * transpose(vecs)
    end
    U, S, V = begin
        res = svd(Yt)
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
    @tensor Γ_new[:] := V[-1,1] * Γ[1,-2,2] * V'[2, -3]
    Γ_new, S
end

#---------------------------------------------------------------------------------------------------
# Non-degenerate case
#---------------------------------------------------------------------------------------------------
export canonical
function canonical(
    Ts::AbstractVector{<:AbstractArray{<:Number, 3}};
    bound::Integer=BOUND,
    tol::AbstractFloat=SVDTOL
)
    n = length(Ts)
    T = tensor_group(Ts)
    nres, tres = block_canonical(T)
    if length(nres) > 1 
        println("Multiple block, keep first one.") 
    end
    A, λ = schmidt_canonical(tres[1], renormalize=true, bound=bound, tol=tol)
    tensor_lmul!(λ, A)
    tensor_decomp!(A, λ, n, renormalize=renormalize, bound=bound, tol=tol)
end
