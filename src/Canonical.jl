#---------------------------------------------------------------------------------------------------
# Helper Funtions
#
# 1. canonical_gauging regauge (sometimes with trucation) the MPS.
# 2. canonical_split splits one MPS to two MPS's.
#---------------------------------------------------------------------------------------------------
function canonical_gauging(
    Γ::AbstractArray{<:Number, 3},
    vals::AbstractVector,
    vecs::AbstractMatrix;
    renormalize::Bool=true
)
    vals_sqrt = sqrt.(vals)
    X = vecs * Diagonal(vals_sqrt)
    Xi = Diagonal(1 ./ vals_sqrt) * vecs'
    @tensor Γ2[:] := Xi[-1,1] * Γ[1,-2,2] * X[2,-3]
    if renormalize
        Γ2n = sqrt(inner_product(Γ2))
        Γ2 ./= Γ2n
    end
    Γ2
end
#---------------------------------------------------------------------------------------------------
function canonical_gauging(
    Γ::AbstractArray{<:Number, 3},
    vecs::AbstractMatrix;
    renormalize::Bool=true
)
    @tensor Γ2[:] := vecs'[-1,1] * Γ[1,-2,2] * vecs[2,-3]
    if renormalize
        Γ2n = sqrt(inner_product(Γ2))
        Γ2 ./= Γ2n
    end
    Γ2
end
#---------------------------------------------------------------------------------------------------
function canonical_split(
    Γ::AbstractArray{<:Number, 3},
    p1::AbstractMatrix,
    p2::AbstractMatrix
)
    @tensor Γ1[:] := p1'[-1,1] * Γ[1,-2,2] * p1[2,-3]
    @tensor Γ2[:] := p2'[-1,1] * Γ[1,-2,2] * p2[2,-3]
    Γ1, Γ2
end
#---------------------------------------------------------------------------------------------------
# Right Canonical Form
#
# 1. The algorithm will tensor that is right-normalized.
# 2. If a degeneracy is encuntered, there would be multiple outputs.
# 3. While it is NOT guaranteed that the outputs are non-degenerate.
# 4. The right_canonical_trim will only keeps one block if degeneracy is encountered.
#---------------------------------------------------------------------------------------------------
function right_cannonical(
    Γ::AbstractArray{<:Number, 3};
    tol::AbstractFloat=1e-14
)
    Γnorm, fixed_mat = fixed_point(Γ)
    if Γnorm < tol
        return [(0.0, Γ)]
    end
    vals, vecs = eigen(fixed_mat)
    pos = sum(vals .< tol)
    if pos == 0
        Γ_new = canonical_gauging(Γ, vals, vecs)
        return [(Γnorm, Γ_new)]
    else
        p1, p2 = vecs[:, pos+1:end], vecs[:, 1:pos]
        Γ1, Γ2 = canonical_split(Γ, p1, p2)
        # recurence
        Γ1_RC = right_cannonical(Γ1, tol=tol)::Vector{Tuple}
        Γ2_RC = right_cannonical(Γ2, tol=tol)::Vector{Tuple}
        return [Γ1_RC; Γ2_RC]
    end
end
#---------------------------------------------------------------------------------------------------
function right_canonical_trim(
    Γ::AbstractArray{<:Number, 3};
    tol::AbstractFloat=1e-14
)
    Γnorm, fixed_mat = fixed_point(Γ)
    vals, vecs = begin
        vals_all, vecs_all = eigen(fixed_mat)
        pos = vals_all .> tol
        vals_all[pos], vecs_all[:, pos]
    end
    canonical_gauging(Γ, vals, vecs)
end

#---------------------------------------------------------------------------------------------------
# Block Decomposition

# 1. Further decompose the right-renormalized tensor into right-renormalized tensors.
# 2. Ensure the outputs are non-degenerate.
# 3. block_trim only return one block when degeneracy is encountered.
#---------------------------------------------------------------------------------------------------
function fixed_mat_2(
    vecs::AbstractMatrix, 
    α::Integer, 
    tol::AbstractFloat
)
    eig_mat = reshape(vecs[:, 1], α, α)
    eig_mat_h = (1+1im) * eig_mat + (1-1im) * eig_mat'
    id_mat = I(α) * eig_mat_h[1,1]
    if norm(id_mat - eig_mat_h) < tol
        # The first eigen matrix is identity.
        # Here I assume in this case the second eigen matrix is good.
        eig_mat = reshape(vecs[:, 2], α, α)
        eig_mat_h = (1+1im) * eig_mat + (1-1im) * eig_mat'
        id_mat = I(α) * eig_mat_h[1,1]
        @assert norm(id_mat - eig_mat_h) > tol "Second identity. Get trouble!!!"
    end
    Hermitian(eig_mat_h)
end
#---------------------------------------------------------------------------------------------------
function block_split(
    Γ::AbstractArray{<:Number, 3},
    vecs::AbstractArray{<:Number, 2};
    tol::AbstractFloat=1e-5
)
    α = size(Γ, 1)
    fixed_mat = fixed_mat_2(vecs, α, tol)
    vals, vecs = eigen(fixed_mat)
    pos = sum(maximum(vals) .- vals .< tol)
    @assert 0 < pos < length(vals) "Illegal block split."
    # eigenvalues is from small to large
    p1, p2 = vecs[:, 1:end-pos], vecs[:, end-pos+1:end]
    canonical_split(Γ, p1, p2)
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
    Γ1c = block_decomp(Γ1)::Vector
    Γ2c = block_decomp(Γ2)::Vector
    return [Γ1c; Γ2c]
end
#---------------------------------------------------------------------------------------------------
function block_trim(
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
        return Γ
    end
    # Degenerate:
    fixed_mat = fixed_mat_2(vecs, α, tol)
    vals, vecs = eigen(fixed_mat)
    pos = maximum(vals) .- vals .< tol
    p1 = vecs[:, pos]
    Γ_trim = canonical_gauging(Γ, p1)
    block_trim(Γ_trim, tol=tol)
end

#---------------------------------------------------------------------------------------------------
# Block Right Canonical Form
#
# 1. All-at-once method.
# 2. Return multiple non-degenerate right-canonical form.
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
# 1. Given a right canonical form, return a Schmidt canonical form.
# 2. This algorithm assume there is no degeneracy.
#---------------------------------------------------------------------------------------------------
function schmidt_canonical(
    Γ::AbstractArray{<:Number,3};
    renormalize::Bool=false,
    bound::Integer=BOUND,
    tol::AbstractFloat=SVDTOL,
    zero_tol::AbstractFloat=1e-14
)
    α = size(Γ, 1)
    lmat = begin
        trans_mat_T = Array(transpose(trm(Γ)))
        emax, lvec = krylov_eigen(trans_mat_T)
        reshape(lvec, α, :)
    end
    Yt = begin
        vals_all, vecs_all = eigen(Hermitian(lmat))
        pos = vals_all .> zero_tol
        vals = vals_all[pos]
        vecs = vecs_all[:, pos]
        Diagonal(sqrt.(vals)) * transpose(vecs)
    end
    U, S, V = svd_trim(Yt)
    @tensor Γ_new[:] := V[-1,1] * Γ[1,-2,2] * V'[2, -3]
    if renormalize
        S ./= norm(S)
        Γ_new ./= sqrt(inner_product(Γ_new))
    end
    Γ_new, S
end

#---------------------------------------------------------------------------------------------------
# Non-degenerate Schmidt Canonical Form
#---------------------------------------------------------------------------------------------------
function schmidt_canonical(
    Ts::AbstractVector{<:AbstractArray{<:Number, 3}};
    renormalize=true,
    bound::Integer=BOUND,
    tol::AbstractFloat=SVDTOL
)
    n = length(Ts)
    T = tensor_group(Ts)
    T_RC = right_canonical_trim(T)
    A, λ = schmidt_canonical(T_RC, renormalize=renormalize)
    tensor_lmul!(λ, A)
    tensor_decomp!(A, λ, n, renormalize=renormalize, bound=bound, tol=tol)
end
#---------------------------------------------------------------------------------------------------
function canonical_trim(
    Ts::AbstractVector{<:AbstractArray{<:Number, 3}};
    renormalize=true,
    bound::Integer=BOUND,
    tol::AbstractFloat=SVDTOL
)
    n = length(Ts)
    T = tensor_group(Ts)
    T_RC = right_canonical_trim(T)
    T_BRC = block_trim(T_RC)
    A, λ = schmidt_canonical(T_BRC, renormalize=renormalize)
    tensor_lmul!(λ, A)
    tensor_decomp!(A, λ, n, renormalize=renormalize, bound=bound, tol=tol)
end
