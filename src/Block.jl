#---------------------------------------------------------------------------------------------------
# Block decomposition
# Developing
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
        X_inv = inv(X)
        @tensor Γ_new[:] := X_inv[-1,1] * Γ[1,-2,2] * X[2,-3]
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
