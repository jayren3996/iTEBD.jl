using Test
using iTEBD

@testset "iMPS parametric struct" begin
    @testset "dense alias" begin
        ψ = rand_iMPS(ComplexF64, 2, 2, 3)
        @test ψ isa DenseIMPS
        @test ψ isa DenseIMPS{ComplexF64, Float64}
        @test eltype(ψ.Γ[1]) === ComplexF64
        @test eltype(ψ.λ[1]) === Float64
        @test typeof(ψ).parameters[1] === Array{ComplexF64, 3}
        @test typeof(ψ).parameters[2] === Vector{Float64}
    end

    @testset "struct exposes Γ, λ, n fields" begin
        ψ = rand_iMPS(ComplexF64, 2, 2, 3)
        @test fieldnames(iMPS) === (:Γ, :λ, :n)
        @test ψ.n == 2
    end
end

@testset "TensorKit-extension stubs (base package, no TensorKit loaded)" begin
    # `graded_space(:U1, …)` without TensorKit gives the actionable
    # "load TensorKit" error rather than a confusing MethodError.
    err = try
        graded_space(:U1, 0=>1)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("TensorKit", err.msg)

    # `schmidt_values` on the dense backend returns a Vector{Float64} that
    # aliases the underlying λ when types already match — verify the type
    # and the per-element values, but don't depend on aliasing (it's an
    # implementation detail of `convert`).
    ψ = rand_iMPS(ComplexF64, 2, 2, 3)
    sv = schmidt_values(ψ, 1)
    @test sv isa Vector{Float64}
    @test sv == ψ.λ[1]
end
