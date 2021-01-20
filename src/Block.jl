#---------------------------------------------------------------------------------------------------
# Block decomposition
# Developing
#---------------------------------------------------------------------------------------------------
hermitianize(mat::AbstractMatrix) = Hermitian( (1+1im) * mat + (1-1im) * mat' )

function fixed_point(
    Γ::AbstractArray{<:Number, 3};
    tol::AbstractFloat=SORTTOL
)
    α = size(Γ, 1)
    trans_mat = trm(Γ)
    vals, vecs = eigen(trans_mat)
    tensor_norm = maximum(abs.(vals))
    fixed_points_positions = abs.(vals .- tensor_norm) .< tol
    # need one Hermitian fixed points
    fixed_point = vecs[ :, fixed_points_positions[1] ]
    fixed_point_mat = reshape(fixed_point, α, α)
    hermitianize(fixed_point_mat)
end

function right_cannonical(
    Γ::AbstractArray{<:Number, 3};
    tol::AbstractFloat=SORTTOL
)
    fixed_mat = fixed_point(Γ, tol=tol)
    vals, vecs = eigen(herm_mat)
    pos = sum(vals .< tol)
    if pos == length(vals)
        vals *= -1
        pos = sum(vals .< tol)
    end
    if pos == 0
        X = vecs * Diagonal(sqrt.(vals))
        X_inv = inv(X)
        ctype = promote_type(eltype(Γ), eltype(X))
        @tensor Γ_new[:] := X_inv[-1,1] * Γ[1,-2,2] * X[2,-3]
        return [Γ_new]
    else
        p1, p2 = vecs[:, 1:pos], vecs[:, pos+1:end]
        @tensor Γ1[:] := p1'[-1,1] * Γ[1,-2,2] * p1[2,-3]
        @tensor Γ2[:] := p2'[-1,1] * Γ[1,-2,2] * p2[2,-3]
        println("Γ1 = ", size(Γ1))
        println("Γ2 = ", size(Γ2))
        return vcat(right_cannonical(Γ1), right_cannonical(Γ2))
    end
end

function block_decomp(
    Γ::AbstractArray{<:Number, 3};
    tol::AbstractFloat=SORTTOL
)
    
end
