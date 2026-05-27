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

    # `schmidt_values` on the dense backend returns a fresh `Vector{Float64}`.
    # Verify the type, the values, and — critically — that mutating the result
    # does not bleed back into `ψ.λ[i]`. Previously this was a thin
    # `convert(Vector{Float64}, ψ.λ[i])` wrapper that aliased when the eltype
    # already matched, so a caller running `sv = schmidt_values(ψ, 1); sv .= 0`
    # would silently zero out the state's Schmidt spectrum.
    ψ = rand_iMPS(ComplexF64, 2, 2, 3)
    sv = schmidt_values(ψ, 1)
    @test sv isa Vector{Float64}
    @test sv == ψ.λ[1]
    original = copy(ψ.λ[1])
    sv .= 0.0
    @test ψ.λ[1] == original
end
