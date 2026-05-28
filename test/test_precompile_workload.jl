using Test
using iTEBD
using iTEBD: product_iMPS, applygate!, evolve!, expect, ent_S, spin_half_ops
using TensorKit: id, ⊗, codomain

# These two @testsets mirror, line for line, the bodies of the
# @compile_workload blocks in src/iTEBD.jl and ext/iTEBDTensorKitExt.jl.
# If the public API drifts, this test fails with a clear error instead of
# surfacing as an opaque precompilation failure that blocks `using iTEBD`.

@testset "precompile workload bodies execute" begin
    @testset "dense core" begin
        X = ComplexF64[0 1; 1 0]
        G = kron(X, X)
        ψ = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
        applygate!(ψ, G, 1, 2; maxdim=4)
        evolve!(ψ, [(G, 1, 2), (G, 2, 1)], 3; maxdim=4)
        @test expect(ψ, G, 1, 2) isa Number
        @test ent_S(ψ, 1) isa Real
    end

    @testset "symmetric U(1)" begin
        ψ = product_iMPS(:U1, [-1, 1], [1, -1])
        P = codomain(ψ.Γ[1])[2]
        Iop = id(ComplexF64, P ⊗ P)
        applygate!(ψ, Iop, 1, 2; maxdim=8)
        evolve!(ψ, [(Iop, 1, 2), (Iop, 2, 1)], 3; maxdim=8)
        Sz, _, _, _ = spin_half_ops(:U1)
        @test expect(ψ, Sz, 1, 1) isa Number
        @test ent_S(ψ, 1) isa Real
    end
end
