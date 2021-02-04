#---------------------------------------------------------------------------------------------------
# Eigen system using power iteration

# Find dominent eigensystem by iterative multiplication.
# Krylov method ensures Hermicity and semi-positivity.
#---------------------------------------------------------------------------------------------------
# Quantum channel
function kraus!(
    ρ::AbstractMatrix,
    KL::AbstractArray{<:Number, 3},
    KU::AbstractArray{<:Number, 3},
    ρ0::AbstractMatrix,
    dir::Symbol=:r
)
    if dir == :r
        @tensor ρ[:] = KL[-1, 3, 1] * ρ0[1, 2] * KU[-2, 3, 2]
    elseif dir == :l
        @tensor ρ[:] = KU[1, 3, -1] * ρ0[1, 2] * KL[2, 3, -2]
    else
        error("Illegal direction: $dir.")
    end
end

# Check error.
function krylov_eigen!(
    ρ::AbstractMatrix,
    KL::AbstractArray{<:Number, 3},
    KU::AbstractArray{<:Number, 3};
    dir::Symbol=:r,
    tol::Real=1e-10,
    maxitr::Integer=100,
    warning::Bool=true
)
    ρb = similar(ρ)
    ρc = similar(ρ)
    val = NaN
    err = NaN
    for i=1:maxitr
        kraus!(ρb, KL, KU, ρ, dir)
        kraus!(ρc, KL, KU, ρb, dir)
        @. ρ = ρb + ρc
        val_new = norm(ρ)
        ρ ./= val_new
        err = abs(val - val_new)
        if err < tol 
            return 
        end
        val = val_new
    end
    if warning
        println("Krylov method reached maximum iterations, error = $err.")
    end
end
#---------------------------------------------------------------------------------------------------
# With defined iteration times.
function krylov_eigen!(
    ρ::AbstractMatrix,
    KL::AbstractArray{<:Number, 3},
    KU::AbstractArray{<:Number, 3},
    itr::Integer;
    dir::Symbol=:r
)
    ρb = similar(ρ)
    ρc = similar(ρ)
    for i=1:itr
        kraus!(ρb, KL, KU, ρ, dir)
        kraus!(ρc, KL, KU, ρb, dir)
        @. ρ = ρb + ρc
        normalize!(ρ)
    end
end
#---------------------------------------------------------------------------------------------------
# steady state from identity mat
function steady_mat(
    K::AbstractArray{<:Number, 3};
    dir::Symbol=:r,
    krylov_power::Integer=KRLOV_POWER,
)
    α = size(K, 1)
    ρ = Array{eltype(K)}(I(α))
    krylov_eigen!(ρ, K, conj(K), krylov_power, dir=dir)
    Hermitian(ρ)
end
#---------------------------------------------------------------------------------------------------
# Random fixed-point matrix.
function fixed_point_mat(
    K::AbstractArray{<:Number, 3};
    dir::Symbol=:r,
    krylov_power::Integer=KRLOV_POWER,
)
    α = size(K, 1)
    ρ = rand(ComplexF64, α, α) |> Hermitian |> Array
    krylov_eigen!(ρ, K, conj(K), krylov_power, dir=dir)
    Hermitian(ρ)
end
