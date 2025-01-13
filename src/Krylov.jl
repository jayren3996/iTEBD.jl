#---------------------------------------------------------------------------------------------------
# Eigen system using Arnodi
#---------------------------------------------------------------------------------------------------
"""
Kraus operation

Right direction:
    2 --U--⋅
        |  |
        |  ρ
        |  |
    1 --L--⋅

Left direction:
    ⋅--U-- 1
    |  |  
    ρ  |  
    |  |  
    ⋅--L-- 2
"""
function kraus(
    KL::AbstractArray{<:Number, 3}, KU::AbstractArray{<:Number, 3},
    ρ::AbstractMatrix; dir::Symbol=:r
)
    if dir == :r
        @tensor out[:] := KL[-1, 3, 1] * ρ[1, 2] * KU[-2, 3, 2]
    elseif dir == :l
        @tensor out[:] := KU[1, 3, -1] * ρ[1, 2] * KL[2, 3, -2]
    else
        error("Illegal direction: $dir.")
    end
    out
end
#---------------------------------------------------------------------------------------------------
function krylov_eigen(
    KL::AbstractArray{<:Number, 3}, KU::AbstractArray{<:Number, 3}, 
    ρ0::Union{AbstractMatrix, Nothing}=nothing;
    dir::Symbol=:r
)
    n = size(KL, 1)
    f = ρ -> kraus(KL, KU, reshape(ρ, n, n); dir)
    n = size(KL, 1)
    T = promote_type(eltype(KL), eltype(KU))
    v0 = if isnothing(ρ0)
        reshape(diagm(ones(T, n)), n^2)
    else
        reshape(ρ0, n^2)
    end

    vals, vecs = eigsolve(f, v0, 1, :LM; ishermitian=false)
    if real(tr(vecs[1])) < 0
        vals[1], -reshape(vecs[1], n,n)
    else
        vals[1], reshape(vecs[1], n,n)
    end
end
#---------------------------------------------------------------------------------------------------
# steady state from identity mat
function steady_mat(K::AbstractArray{<:Number, 3}; dir::Symbol=:r)
    _, vec = krylov_eigen(K, conj(K); dir)
    vec |> Hermitian
end
#---------------------------------------------------------------------------------------------------
# Random fixed-point matrix.
function fixed_point_mat(
    K::AbstractArray{<:Number, 3};
    dir::Symbol=:r
)
    α = size(K, 1)
    ρ0 = rand(ComplexF64, α, α) |> Hermitian |> Array
    _, vec = krylov_eigen(K, conj(K), ρ0;dir)
    vec |> Hermitian
end
