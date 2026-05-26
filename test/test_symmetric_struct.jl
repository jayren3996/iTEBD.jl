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
