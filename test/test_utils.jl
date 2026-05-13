module TestUtils

using LinearAlgebra
using Test
using iTEBD

export AKLT_TENSOR
export bell_gate, deterministic_tensor, pauli_matrices
export assert_normalized_schmidt_spectra, assert_stored_tensor_convention
export right_canonical_overlap, right_canonical_error

const AKLT_TENSOR = begin
    tensor = zeros(ComplexF64, 2, 3, 2)
    tensor[1, 1, 2] = +sqrt(2 / 3)
    tensor[1, 2, 1] = -sqrt(1 / 3)
    tensor[2, 2, 2] = +sqrt(1 / 3)
    tensor[2, 3, 1] = -sqrt(2 / 3)
    tensor
end

function deterministic_tensor(dims::Integer...)
    len = prod(dims)
    data = ComplexF64.(1:len) .+ (ComplexF64(0, 1) .* ComplexF64.(len:-1:1)) ./ max(len, 1)
    return reshape(data, dims...)
end

function pauli_matrices(T::Type=ComplexF64)
    I2 = Matrix{T}(I, 2, 2)
    X = T[0 1; 1 0]
    Y = T[0 -im; im 0]
    Z = T[1 0; 0 -1]
    return (; I=I2, X, Y, Z)
end

function bell_gate(T::Type=ComplexF64)
    H = inv(sqrt(T(2))) * T[1 1; 1 -1]
    CNOT = T[
        1 0 0 0
        0 1 0 0
        0 0 0 1
        0 0 1 0
    ]
    return CNOT * kron(H, Matrix{T}(I, 2, 2))
end

function right_canonical_overlap(Γ::AbstractArray{<:Number, 3})
    Dl, d, Dr = size(Γ)
    overlap = zeros(promote_type(eltype(Γ), ComplexF64), Dl, Dl)
    for s in 1:d
        Bs = reshape(Γ[:, s, :], Dl, Dr)
        overlap .+= Bs * Bs'
    end
    return overlap
end

function right_canonical_error(ψ::iTEBD.iMPS)
    errs = Float64[]
    for Γ in ψ.Γ
        Dl = size(Γ, 1)
        push!(errs, norm(right_canonical_overlap(Γ) - Matrix{ComplexF64}(I, Dl, Dl)))
    end
    return maximum(errs)
end

function assert_normalized_schmidt_spectra(ψ::iTEBD.iMPS; atol::Real=1e-10)
    @test all(all(λ .>= -atol) for λ in ψ.λ)
    @test all(isapprox(norm(λ), 1.0; atol) for λ in ψ.λ)
    return nothing
end

function assert_stored_tensor_convention(ψ::iTEBD.iMPS; atol::Real=1e-12)
    for i in 1:ψ.n
        Γbare, λ = ψ[i]
        stored = copy(Γbare)
        iTEBD.tensor_rmul!(stored, λ)
        @test stored ≈ ψ.Γ[i] atol=atol
    end
    return nothing
end

end
