#---------------------------------------------------------------------------------------------------
# Helper Funtions
#
# 1. canonical_gauging regauge (sometimes with trucation) the MPS.
# 2. canonical_split splits one MPS to two MPS's.
#---------------------------------------------------------------------------------------------------
function mat_sqrt(
    mat::Hermitian; 
    zerotol::Real=ZEROTOL
)
    vals_all, vecs_all = eigen(mat)
    pos = vals_all .> zerotol
    vals_sqrt = sqrt.(vals_all[pos])
    vecs = vecs_all[:, pos]
    L = vecs * Diagonal(vals_sqrt)
    R = Diagonal(1 ./ vals_sqrt) * vecs'
    L, R
end
#---------------------------------------------------------------------------------------------------
function canonical_gauging(
    Γ::AbstractArray{<:Number, 3},
    L::AbstractMatrix,
    R::AbstractMatrix
)
    @tensor Γ2[:] := R[-1,1] * Γ[1,-2,2] * L[2,-3]
    Γ2
end
#---------------------------------------------------------------------------------------------------
function vals_group(
    vals::Vector;
    sorttol::Real=SORTTOL
)
    pos = Vector{Vector{Int64}}(undef, 0)
    temp = zeros(Int64, 0)
    current_val = vals[1]
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
# Right Canonical Form
#
# 1. The algorithm will tensor that is right-normalized.
# 2. Automatically trim null-space.
# 3. Is NOT guaranteed that the outputs are non-degenerate.
# 4. The algorithm is UNSTABLE if the transfer matrix has huge condition number.
#---------------------------------------------------------------------------------------------------
function right_canonical(
    Γ::AbstractArray{<:Number, 3};
    krylov_power::Integer=KRLOV_POWER,
    zerotol::Real=ZEROTOL
)
    trmat = trm(Γ)
    ρ = steady_mat(trmat, krylov_power=krylov_power)
    L, R = mat_sqrt(ρ, zerotol=zerotol)
    canonical_gauging(Γ, L, R)
end

#---------------------------------------------------------------------------------------------------
# Block Decomposition

# 1. Further decompose the right-renormalized tensor into right-renormalized tensors.
# 2. Ensure the outputs are non-degenerate.
# 3. block_trim only return one block when degeneracy is encountered.
#---------------------------------------------------------------------------------------------------
function block_decomp(
    Γ::AbstractArray{<:Number, 3};
    krylov_power::Integer=KRLOV_POWER,
    sorttol::Real=SORTTOL
)
    vals, vecs = begin
        trmat = trm(Γ)
        fixed_mat = fixed_point_mat(trmat, krylov_power=krylov_power)
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
        Γc = canonical_gauging(Γ, p, p')
        res[i] = Γc
    end
    res
end
#---------------------------------------------------------------------------------------------------
function block_trim(
    Γ::AbstractArray{<:Number, 3};
    krylov_power::Integer=KRLOV_POWER,
    sorttol::Real=SORTTOL
)
    vals, vecs = begin
        trmat = trm(Γ)
        fixed_mat = fixed_point_mat(trmat, krylov_power=krylov_power)
        eigen(fixed_mat)
    end
    p = begin
        vgroup = vals_group(vals, sorttol=sorttol)
        i = argmin(length.(vgroup))
        vecs[:, vgroup[i]]
    end
    canonical_gauging(Γ, p, p')
end

#---------------------------------------------------------------------------------------------------
# Block Right Canonical Form
#
# 1. All-at-once method.
# 2. Return multiple non-degenerate right-canonical form.
#---------------------------------------------------------------------------------------------------
function block_canonical(
    Γ::AbstractArray{<:Number, 3};
    krylov_power::Integer=KRLOV_POWER,
    sorttol::Real=SORTTOL,
    zerotol::Real=ZEROTOL,
    trim::Bool=false
)
    Γ_RC = right_canonical(Γ, krylov_power=krylov_power, zerotol=zerotol)
    res = if trim
        block_trim(Γ_RC, krylov_power=krylov_power, sorttol=sorttol)
    else
        block_decomp(Γ_RC, krylov_power=krylov_power, sorttol=sorttol)
    end
    res
end

