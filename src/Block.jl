#---------------------------------------------------------------------------------------------------
# Block decomposition
# Developing
#---------------------------------------------------------------------------------------------------
hermitianize(mat::AbstractMatrix) = (1+1im) * mat + (1-1im) * mat'

function fixed_points(
    Γ::AbstrctArray{<:Number, 3};
    tol::AbstractFloat=SORTTOL
)
    α = size(Γ, 1)
    trans_mat = trm(Γ)
    vals, vecs = eigen(trans_mat)
    tensor_norm = maximum(abs.(vals))
    fixed_points_positions = abs.(vals .- tensor_norm) .< tol
    fixed_points = vecs[:, fixed_points_positions]
    [reshape(fixed_points[:, i], α, α) for i=1:size(fixed_points, 2)]
end

function right_cannonical(
    Γ::AbstrctArray{<:Number, 3};
    tol::AbstractFloat=SORTTOL
)
    fixed_mats = fixed_points(Γ)[1]
    herm_mat = hermitianize(fixed_mat)
    vals, vecs = eigen(herm_mat)
    pos = sum(vals .< tol)
    if pos == 0
        X = vecs * Diagonal(sqrt.(vals))
        X_inv = inv(X)
        ctype = promote_type(eltype(Γ), eltype(X))
        Γ_new = Array{ctype, 3}(undef, size(Γ))
        @tensor Γ_new[:] = X_inv[-1,1] * Γ[1,-2,2] * X[2,-3]
        return [Γ_new]
    else
        p1, p2 = vecs[:, 1:pos], vecs[:, pos+1:end]
        Γ1 = p1' * Γ * p1
        Γ2 = p2' * Γ * p2
        return vcat(right_cannonical(Γ1), right_cannonical(Γ1))
    end
end

function block_decomp(
    Γ::AbstrctArray{<:Number, 3};
    tol::AbstractFloat=SORTTOL
)
    fixed_mats = fixed_points(Γ)
    if length(fixed_mats) == 1
        return [Γ]
    else
        for mats in fixed_mats
            herm_mat = hermitianize(mat)
            if norm(herm_mat - I( size(herm_mat, 1) ) ) > tol
                X = herm_mat
                break
            end
        end
        λ = eigmax(X)
        herm_mat = I(size(herm_mat, 1))
    end
end