using Test
using LinearAlgebra
using iTEBD
using iTEBD: product_iMPS, applygate!, evolve!

const H_GATE = inv(sqrt(2)) * [1.0 1.0; 1.0 -1.0]
const CNOT_GATE = [
    1.0 0 0 0;
    0 1 0 0;
    0 0 0 1;
    0 0 1 0
]
const BELL_GATE = CNOT_GATE * kron(H_GATE, I(2))

@testset "DISCARDED_WEIGHT_SELECTOR" begin
    s = [4.0, 2.0, 1.0, 0.5]
    choice = iTEBD._discarded_weight_choice(s; mindim=1, maxdim=4, truncerr=0.06)

    @test choice.chi_req == 2
    @test choice.chi_keep == 2
    @test choice.discarded_weight <= 0.06
    @test sum(choice.weights[choice.chi_keep + 1:end]) <= 0.06
    @test sum(choice.weights[choice.chi_keep:end]) > 0.06
end

@testset "APPLYGATE_TRUNCERR_IS_ENFORCED_WHEN_NOT_SATURATED" begin
    psi = product_iMPS(ComplexF64, [[1, 0], [1, 0]])

    psi_after, stats = applygate!(
        psi,
        BELL_GATE,
        1,
        2;
        maxdim=4,
        mindim=1,
        truncerr=0.5,
        return_stats=true
    )

    @test psi_after === psi
    @test length(stats.bond_stats) == 1
    @test stats.bond_stats[1].chi_keep == 1
    @test !stats.bond_stats[1].saturated
    @test stats.bond_stats[1].discarded_weight <= 0.5
    @test length(psi.λ[1]) == 1
end

@testset "SATURATION_IS_REPORTED" begin
    s = ones(3) ./ sqrt(3)
    choice = iTEBD._discarded_weight_choice(s; mindim=1, maxdim=1, truncerr=0.1)

    @test choice.chi_req == 3
    @test choice.chi_keep == 1
    @test choice.saturated
    @test !choice.target_met
    @test choice.discarded_weight ≈ 2 / 3 atol=1e-12
end

@testset "EVOLUTION_REGRESSION_VS_TIGHTER_REFERENCE" begin
    X = [0.0 1.0; 1.0 0.0]
    Z = [1.0 0.0; 0.0 -1.0]
    H = kron(X, X) + 0.2 * kron(Z, Z)
    gates = [(exp(-0.05im * H), 1, 2), (exp(-0.05im * H), 2, 1)]

    psi_ref = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
    psi_test = product_iMPS(ComplexF64, [[1, 0], [0, 1]])

    evolve!(psi_ref, gates, 20; maxdim=16, truncerr=1e-12)
    evolve!(psi_test, gates, 20; maxdim=8, truncerr=1e-6)

    @test iTEBD.inner_product(psi_test, psi_ref) > 0.999
end

@testset "BOND_LOCAL_DIMENSIONS_CAN_DIFFER" begin
    psi = product_iMPS(ComplexF64, [[1, 0], [1, 0], [1, 0], [1, 0]])

    applygate!(psi, BELL_GATE, 1, 2; maxdim=4, truncerr=1e-12)

    @test length(psi.λ[1]) == 2
    @test all(length(psi.λ[k]) == 1 for k in 2:4)
end

@testset "LOCAL_DIMENSION_CAN_GROW_AND_SHRINK" begin
    psi = product_iMPS(ComplexF64, [[1, 0], [1, 0]])

    _, stats_grow = applygate!(
        psi,
        BELL_GATE,
        1,
        2;
        maxdim=4,
        truncerr=1e-12,
        return_stats=true
    )
    _, stats_shrink = applygate!(
        psi,
        adjoint(BELL_GATE),
        1,
        2;
        maxdim=4,
        truncerr=1e-12,
        return_stats=true
    )

    @test stats_grow.bond_stats[1].chi_keep == 2
    @test stats_shrink.bond_stats[1].chi_keep == 1
    @test length(psi.λ[1]) == 1
end

@testset "EVOLVE_RETURNS_AGGREGATE_TRUNCATION_STATS" begin
    psi = product_iMPS(ComplexF64, [[1, 0], [1, 0]])

    psi_after, stats = evolve!(
        psi,
        [(BELL_GATE, 1, 2), (adjoint(BELL_GATE), 2, 1)],
        1;
        maxdim=4,
        truncerr=1e-12,
        return_stats=true
    )

    @test psi_after === psi
    @test length(stats.gate_updates) == 2
    @test stats.max_kept_dim == 2
    @test stats.num_saturated == 0
    @test stats.max_discarded_weight >= stats.mean_discarded_weight >= 0
end
