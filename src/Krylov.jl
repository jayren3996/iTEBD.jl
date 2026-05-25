#---------------------------------------------------------------------------------------------------
# Eigen system using Arnodi
#---------------------------------------------------------------------------------------------------

# Helper for forwarding optional Krylov tolerances into `eigsolve` while
# preserving KrylovKit's defaults when the user does not override them.
# Defined here (the first file with a Krylov caller) so downstream files in
# the include chain can reuse it.
function _krylov_opts(; tol::Union{Nothing,Real}=nothing,
                       maxiter::Union{Nothing,Integer}=nothing)
    opts = NamedTuple()
    isnothing(tol)     || (opts = merge(opts, (; tol)))
    isnothing(maxiter) || (opts = merge(opts, (; maxiter)))
    return opts
end

"""
    kraus(KL, KU, ρ; dir=:r)

Apply a two-sided Kraus-like map to a matrix `ρ`.

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

Parameters:
- `KL`, `KU`
  Left and right local tensors defining the transfer action.
- `ρ`
  Matrix to which the map is applied.

Keyword arguments:
- `dir=:r`
  Direction of the action. Use `:r` for the right fixed-point equation and `:l`
  for the left one.

Returns:
- The transformed matrix.

Notes:
- This is a low-level building block for fixed-point and transfer-matrix
  computations.
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
"""
    kraus_mat(KL, KU; dir=:r)

Matrix representation of the Kraus-like transfer map defined by `KL` and `KU`.

Parameters:
- `KL`, `KU`
  Left and right local tensors defining the map.

Keyword arguments:
- `dir=:r`
  Direction of the action, matching [`kraus`](@ref).

Returns:
- Dense matrix representation of the linear map acting on vectorized matrices.
"""
function kraus_mat(KL::AbstractArray{<:Number, 3}, KU::AbstractArray{<:Number, 3}; dir::Symbol=:r)
    if dir == :r
        @tensor out[:] := KL[-1, 1, -3] * KU[-2, 1, -4]
    elseif dir == :l
        @tensor out[:] := KU[-3, 1, -1] * KL[-4, 1, -2]
    else
        error("Illegal direction: $dir.")
    end
    a, b, c, d = size(out)
    reshape(out, (a*b, c*d))
end
#---------------------------------------------------------------------------------------------------
"""
    krylov_eigen(KL, KU, ρ0=nothing; dir=:r, project_psd=false)

Compute the dominant eigenpair of the transfer map defined by `KL` and `KU`.

Parameters:
- `KL`, `KU`
  Left and right local tensors defining the transfer map.
- `ρ0`
  Optional initial guess for the fixed-point matrix.

Keyword arguments:
- `dir=:r`
  Direction of the fixed-point equation.
- `project_psd=false`
  If `true`, symmetrize the returned eigenmatrix and project it to the positive
  semidefinite cone when needed. This is only appropriate for self-transfer maps.

Returns:
- `(λ, ρ)` where `λ` is the dominant eigenvalue and `ρ` is the corresponding
  eigenmatrix.

Notes:
- Internally this uses `eigsolve` on the vectorized map.
"""
function krylov_eigen(
    KL::AbstractArray{<:Number, 3}, KU::AbstractArray{<:Number, 3},
    ρ0::Union{AbstractMatrix, Nothing}=nothing;
    dir::Symbol=:r,
    project_psd::Bool=false,
    tol::Union{Nothing,Real}=nothing,
    maxiter::Union{Nothing,Integer}=nothing,
)
    input_shape, output_shape = _krylov_fixed_point_shapes(KL, KU; dir)
    prod(input_shape) == prod(output_shape) ||
        throw(ArgumentError(
            "vectorized transfer map must be square; input shape $input_shape maps to output shape $output_shape"
        ))
    if project_psd && !_krylov_can_project_psd(KL, KU, input_shape, output_shape)
        throw(ArgumentError("project_psd=true is only supported for square self-transfer fixed points"))
    end
    f = ρ -> vec(kraus(KL, KU, reshape(ρ, input_shape); dir))
    T = promote_type(eltype(KL), eltype(KU))
    v0 = if isnothing(ρ0)
        seed = zeros(T, input_shape)
        for i in 1:min(input_shape...)
            seed[i, i] = one(T)
        end
        vec(seed)
    else
        size(ρ0) == input_shape ||
            throw(ArgumentError("initial fixed-point matrix has shape $(size(ρ0)); expected $input_shape"))
        vec(ρ0)
    end

    opts = _krylov_opts(; tol, maxiter)
    vals, vecs, info = eigsolve(f, v0, 1, :LM; ishermitian=false, opts...)
    if info.converged < 1
        @warn "Krylov eigen solver did not converge" info
    end
    ρ = reshape(vecs[1], input_shape)
    # Enforce positive trace
    if size(ρ, 1) == size(ρ, 2) && real(tr(ρ)) < 0
        ρ = -ρ
    end
    if project_psd
        ρ = (ρ + ρ') / 2
        min_eig = minimum(real.(eigvals(Hermitian(ρ))))
        if min_eig < -1e-10
            @warn "Fixed-point matrix has negative eigenvalues (min=$min_eig); projecting to PSD cone"
            evals, evecs = eigen(Hermitian(ρ))
            evals_clipped = max.(evals, 0.0)
            ρ = evecs * Diagonal(evals_clipped) * evecs'
        end
    end
    vals[1], ρ
end

function _krylov_fixed_point_shapes(
    KL::AbstractArray{<:Number, 3},
    KU::AbstractArray{<:Number, 3};
    dir::Symbol=:r
)
    size(KL, 2) == size(KU, 2) ||
        throw(ArgumentError("transfer tensors must have matching physical dimensions"))
    if dir == :r
        return (size(KL, 3), size(KU, 3)), (size(KL, 1), size(KU, 1))
    elseif dir == :l
        return (size(KU, 1), size(KL, 1)), (size(KU, 3), size(KL, 3))
    end
    throw(ArgumentError("Illegal direction: $dir."))
end

function _krylov_can_project_psd(KL, KU, input_shape, output_shape)
    input_shape == output_shape || return false
    input_shape[1] == input_shape[2] || return false
    size(KL) == size(KU) || return false
    return all(kl == conj(ku) for (kl, ku) in zip(KL, KU))
end
#---------------------------------------------------------------------------------------------------
"""
    steady_mat(K; dir=:r)

Compute the dominant fixed-point matrix of the transfer map generated by `K`.

Parameters:
- `K`
  Local tensor defining the transfer map.

Keyword arguments:
- `dir=:r`
  Direction of the fixed-point equation.

Returns:
- Hermitian matrix representing the dominant fixed point.

Notes:
- For small bond dimensions a dense eigendecomposition is used; otherwise a
  Krylov solve is used.
"""
function steady_mat(
    K::AbstractArray{<:Number, 3};
    dir::Symbol=:r,
    tol::Union{Nothing,Real}=nothing,
    maxiter::Union{Nothing,Integer}=nothing,
)
    a, b, _ = size(K)
    # The dense path is O(a^4 * b) to build + O(a^6) for full eigen, while the
    # Krylov path costs ~k_arn * a^3 * b per matvec. Crossover is at a^3 ≈ k * b
    # with k_arn ≈ 20–30, so a ≈ (20 b)^{1/3}. For grouped tensors of physical
    # dimension up to ~256 the crossover stays under a = 18; a fixed cutoff of 8
    # is conservative for all realistic d^n while still letting toy χ ≤ 8 cases
    # use the dense path (where Krylov startup overhead can dominate).
    use_dense = a <= 8
    vec = if use_dense
        m = kraus_mat(K, conj(K); dir)
        vals, vecs = eigen(m)
        idx = argmax(abs.(vals))
        v = reshape(vecs[:,idx], a, a)
        real(tr(v)) < 0 ? -v : v
    else
        krylov_eigen(K, conj(K); dir, project_psd=true, tol, maxiter)[2]
    end
    # Explicitly symmetrize before wrapping in Hermitian
    vec = (vec + vec') / 2
    return Hermitian(vec)
end
#---------------------------------------------------------------------------------------------------
# Random fixed-point matrix.
"""
    fixed_point_mat(K; dir=:r)

Compute a fixed-point matrix starting from a random initial guess.

Parameters:
- `K`
  Local tensor defining the transfer map.

Keyword arguments:
- `dir=:r`
  Direction of the fixed-point equation.

Returns:
- Hermitian matrix approximating the dominant fixed point.

Notes:
- This is primarily useful as a diagnostic or alternative initialization path.
"""
function fixed_point_mat(
    K::AbstractArray{<:Number, 3};
    dir::Symbol=:r,
    tol::Union{Nothing,Real}=nothing,
    maxiter::Union{Nothing,Integer}=nothing,
)
    α = size(K, 1)
    ρ0 = rand(ComplexF64, α, α)
    ρ0 = ρ0 * ρ0'
    ρ0 ./= tr(ρ0)
    _, vec = krylov_eigen(K, conj(K), ρ0; dir, project_psd=true, tol, maxiter)
    vec |> Hermitian
end
