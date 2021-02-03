#---------------------------------------------------------------------------------------------------
# Eigen system using power iteration

# Find dominent eigensystem by iterative multiplication.
# Krylov method ensures Hermicity and semi-positivity.
#---------------------------------------------------------------------------------------------------
# Check error.
function krylov_eigen!(
    va::Vector,
    mat::AbstractMatrix;
    tol::Real=1e-10,
    maxitr::Integer=1000,
    warning::Bool=true
)
    vb = similar(va)
    vc = similar(va)
    val = NaN
    err = NaN
    for i=1:maxitr
        mul!(vb, mat, va)
        mul!(vc, mat, vb)
        @. va = vb + vc
        normalize!(va)
        val_new = dot(va, vb)
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
    va::Vector,
    mat::AbstractMatrix,
    itr::Integer
)
    vb = similar(va)
    vc = similar(va)
    for i=1:itr
        mul!(vb, mat, va)
        mul!(vc, mat, vb)
        @. va = vb + vc
        normalize!(va)
    end
end
#---------------------------------------------------------------------------------------------------
# steady state from identity mat
function steady_mat(
    mat::AbstractMatrix;
    krylov_power::Integer=KRLOV_POWER,
)
    α = round(Int64, sqrt(size(mat, 1)))
    ρ = Array{eltype(mat)}(I(α))
    v0 = reshape(ρ, :)
    krylov_eigen!(v0, mat, krylov_power)
    Hermitian(ρ)
end
#---------------------------------------------------------------------------------------------------
# Random fixed-point matrix.
function fixed_point_mat(
    mat::AbstractMatrix;
    krylov_power::Integer=KRLOV_POWER,
)
    α = round(Int64, sqrt(size(mat, 1)))
    ρ = rand(ComplexF64, α, α) |> Hermitian |> Array
    v0 = reshape(ρ, :)
    krylov_eigen!(v0, mat, krylov_power)
    Hermitian(ρ)
end
