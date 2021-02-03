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
    renormalize::Bool=false
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
    renormalize::Bool=false
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
    p2::AbstractMatrix;
    renormalize::Bool=false
)
    @tensor Γ1[:] := p1'[-1,1] * Γ[1,-2,2] * p1[2,-3]
    @tensor Γ2[:] := p2'[-1,1] * Γ[1,-2,2] * p2[2,-3]
    if renormalize
        Γ1n = sqrt(inner_product(Γ1))
        Γ2n = sqrt(inner_product(Γ2))
        Γ1 ./= Γ1n
        Γ2 ./= Γ2n
    end
    Γ1, Γ2
end

#---------------------------------------------------------------------------------------------------
# Krylov eigen system 

# Find dominent eigensystem by iterative multiplication.
# Krylov method ensures Hermicity and semi-positivity.
# The trial vector is always choose to be flattened identity matrix.
#---------------------------------------------------------------------------------------------------
function krylov_eigen_iteration!(
    va::Vector,
    vb::Vector,
    mat::AbstractMatrix,
    tol::Real,
    maxitr::Integer
)
    itr = 1
    val = 0.0
    err = 1.0
    while err > tol
        mul!(vb, mat, va)
        mul!(va, mat, vb)
        normalize!(va)
        val_new = dot(va, vb)
        err = abs(val - val_new)
        if itr > maxitr
            # Exit loop and print warning
            println("Krylov method failed to converge within maximum number of iterations.")
            println("Krylov error = $err")
            break
        end
        val = val_new
        itr += 1
    end
end
#---------------------------------------------------------------------------------------------------
function fixed_point(
    mat::AbstractMatrix,
    v0::AbstractVector;
    tol::Real=1e-10,
    maxitr::Integer=10000
)
    # choose pmat to avoid other exp(iθ) eigen values
    pmat = (mat^2 + mat) / 2
    vb = pmat * v0
    va = normalize(pmat * vb)
    krylov_eigen_iteration!(va, vb, pmat, tol, maxitr)
    mul!(vb, mat, va)
    val = dot(va , vb)
    val, va
end

#---------------------------------------------------------------------------------------------------
# Right Canonical Form
#
# 1. The algorithm will tensor that is right-normalized.
# 2. Automatically trim null-space.
# 3. Is NOT guaranteed that the outputs are non-degenerate.
# 4. The algorithm is UNSTABLE if the transfer matrix has huge condition number.
#---------------------------------------------------------------------------------------------------
function right_canonical(
    Γ::AbstractArray{<:Number, 3};
    tol::Real=1e-17,
    renormalize=true
)
    α = size(Γ, 1)
    trmat = trm(Γ)
    v0 = Array(reshape(I(α), :))

    Γ_norm, fixed_vec = fixed_point(trmat, v0)
    fixed_mat = Hermitian(reshape(fixed_vec, α, α))
    vals, vecs = eigen(fixed_mat)

    pos = vals .> tol
    Γ_new = canonical_gauging(Γ, vals[pos], vecs[:, pos])
    if renormalize
        Γ_new ./= sqrt(Γ_norm)
    end
    Γ_new
end

#---------------------------------------------------------------------------------------------------
# Block Decomposition

# 1. Further decompose the right-renormalized tensor into right-renormalized tensors.
# 2. Ensure the outputs are non-degenerate.
# 3. block_trim only return one block when degeneracy is encountered.
#---------------------------------------------------------------------------------------------------
function fixed_mat_2(
    mat::AbstractMatrix, 
    α::Integer, 
    tol::Real=1e-10,
    maxitr::Integer=10000
)
    ρ = Hermitian(rand(ComplexF64, α, α))
    v0 = Array(reshape(ρ, :))
    val, fixed_vec = fixed_point(mat, v0, tol=tol, maxitr=maxitr)
    Hermitian(reshape(fixed_vec, α, α))
end
#---------------------------------------------------------------------------------------------------
function vals_group(
    vals::Vector;
    sorttol::Real=1e-3
)
    pos = Vector{Vector{Int64}}(undef, 0)
    current_val = vals[1]
    temp = zeros(Int64, 0)
    for i=1:length(vals)
        if abs(vals[i]-current_val) .< sorttol
            push!(temp, i)
        else
            push!(pos, temp)
            temp = [i]
            current_val = vals[i]
        end
    end
    push!(pos, temp)
    pos
end
#---------------------------------------------------------------------------------------------------
function block_decomp(
    Γ::AbstractArray{<:Number, 3};
    tol::Real=1e-10,
    maxitr::Integer=10000,
    sorttol::Real=1e-3
)
    α = size(Γ, 1)
    if α == 1
        return [Γ]
    end
    vals, vecs = begin
        trmat = trm(Γ)
        fixed_mat = fixed_mat_2(trmat, α, tol, maxitr)
        eigen(fixed_mat)
    end   
    vgroup = vals_group(vals, sorttol=sorttol)
    res = begin
        num = length(vgroup)
        ctype = promote_type(eltype(Γ), eltype(vecs))
        Vector{Array{ctype, 3}}(undef, num)
    end
    for i=1:num
        p = vecs[:, vgroup[i]]
        Γc = canonical_gauging(Γ, p)
        res[i] = Γc
    end
    res
end
#---------------------------------------------------------------------------------------------------
function block_trim(
    Γ::AbstractArray{<:Number, 3};
    tol::Real=1e-5,
    maxitr::Integer=10000,
    sorttol::Real=1e-3
)
    α = size(Γ, 1)
    if α == 1
        return Γ
    end
    vals, vecs = begin
        trmat = trm(Γ)
        fixed_mat = fixed_mat_2(trmat, α, tol, maxitr)
        eigen(fixed_mat)
    end
    vgroup = vals_group(vals, sorttol=sorttol)
    i = argmin(length.(vgroup))
    p = vecs[:, vgroup[i]]
    canonical_gauging(Γ, p)
end

#---------------------------------------------------------------------------------------------------
# Block Right Canonical Form
#
# 1. All-at-once method.
# 2. Return multiple non-degenerate right-canonical form.
#---------------------------------------------------------------------------------------------------
export block_canonical
function block_canonical(Γ::AbstractArray{<:Number, 3})
    Γ_RC = right_canonical(Γ)
    block_decomp(Γ_RC)
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
    tol::Real=SVDTOL,
    zero_tol::Real=1e-14
)
    α = size(Γ, 1)
    lmat = begin
        trans_mat_T = Array(transpose(trm(Γ)))
        v0 = Array(reshape(I(α), :))
        emax, lvec = fixed_point(trans_mat_T, v0)
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

export schmidt_canonical
function schmidt_canonical(
    Ts::AbstractVector{<:AbstractArray{<:Number, 3}};
    renormalize=true,
    bound::Integer=BOUND,
    tol::Real=SVDTOL
)
    n = length(Ts)
    T = tensor_group(Ts)
    T_RC = right_canonical(T)
    A, λ = schmidt_canonical(T_RC, renormalize=renormalize)
    tensor_lmul!(λ, A)
    tensor_decomp!(A, λ, λ, n, renormalize=renormalize, bound=bound, tol=tol)
end
#---------------------------------------------------------------------------------------------------
export canonical_trim
function canonical_trim(
    Ts::AbstractVector{<:AbstractArray{<:Number, 3}};
    renormalize=true,
    bound::Integer=BOUND,
    tol::Real=SVDTOL
)
    n = length(Ts)
    T = tensor_group(Ts)
    T_RC = right_canonical(T)
    T_BRC = block_trim(T_RC)
    A, λ = schmidt_canonical(T_BRC, renormalize=renormalize)
    tensor_lmul!(λ, A)
    tensor_decomp!(A, λ, λ, n, renormalize=renormalize, bound=bound, tol=tol)
end
