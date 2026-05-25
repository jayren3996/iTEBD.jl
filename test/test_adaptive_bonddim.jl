using Test
import iTEBD
using iTEBD: adaptive_bonddim, natural_bonddim

@testset "NATURAL_BONDDIM" begin
    @test natural_bonddim([1.0, 0.0, 0.0]; alpha=0.0) ≈ 1.0 atol=1e-12

    bell = [inv(sqrt(2)), inv(sqrt(2))]
    @test natural_bonddim(bell; q=1.0, alpha=0.0) ≈ 2.0 atol=1e-12

    narrow = [sqrt(0.95), sqrt(0.05), 0.0]
    broad = [sqrt(0.50), sqrt(0.50), 0.0]
    @test natural_bonddim(broad; q=1.0, alpha=0.0) >
        natural_bonddim(narrow; q=1.0, alpha=0.0)

    tailed = [sqrt(0.8), sqrt(0.15), sqrt(0.05)]
    @test natural_bonddim(tailed; q=1.0, alpha=0.1) >
        natural_bonddim(tailed; q=1.0, alpha=0.0)
    @test natural_bonddim(tailed) ≈
        natural_bonddim(tailed; q=1.0, alpha=0.1) atol=1e-12

    spectrum = [sqrt(0.7), sqrt(0.2), sqrt(0.1)]
    @test natural_bonddim(spectrum; q=0.8, alpha=0.0) >
        natural_bonddim(spectrum; q=1.0, alpha=0.0)
    @test natural_bonddim(spectrum; q=1.0, alpha=0.0) >
        natural_bonddim(spectrum; q=2.0, alpha=0.0)
end

@testset "ADAPTIVE_BONDDIM" begin
    spectrum = [sqrt(0.6), sqrt(0.25), sqrt(0.15)]

    χ_raw = natural_bonddim(spectrum)
    @test adaptive_bonddim(1, spectrum; mindim=2, maxdim=10) ==
        max(2, ceil(Int, χ_raw))

    @test adaptive_bonddim(7, spectrum; mindim=2, maxdim=10) == 7
    @test adaptive_bonddim(1, fill(0.5, 8); mindim=2, maxdim=3) == 3

    @test_throws ArgumentError natural_bonddim(spectrum; q=-0.1)
    @test_throws ArgumentError natural_bonddim(spectrum; alpha=-0.1)
    @test_throws ArgumentError adaptive_bonddim(1, spectrum; mindim=0)
    @test_throws ArgumentError adaptive_bonddim(1, spectrum; mindim=3, maxdim=2)
end

@testset "ADAPTIVE_BONDDIM_RATCHE" begin
    spectrum = [sqrt(0.6), sqrt(0.25), sqrt(0.15)]

    # With ratchet=true (default), bond dimension never decreases
    @test adaptive_bonddim(5, spectrum; maxdim=10) == 5

    # With ratchet=false, bond dimension can decrease
    χ = adaptive_bonddim(5, spectrum; maxdim=10, ratchet=false)
    @test χ < 5
end

@testset "NATURAL_BONDDIM_EXTREME_Q" begin
    # Extreme q values should not overflow/underflow
    spectrum = [sqrt(0.5), sqrt(0.5)]
    @test isfinite(natural_bonddim(spectrum; q=1e6))
    @test isfinite(natural_bonddim(spectrum; q=1e-6))
end

@testset "NATURAL_BONDDIM_Q_ZERO_IS_SUPPORT_COUNT" begin
    # q = 0 reduces to the support-count rank: number of above-cutoff modes
    # in the normalized spectrum.
    bell = [inv(sqrt(2)), inv(sqrt(2))]
    @test natural_bonddim(bell; q=0.0, alpha=0.0) ≈ 2.0 atol=1e-12

    three_uniform = fill(inv(sqrt(3)), 3)
    @test natural_bonddim(three_uniform; q=0.0, alpha=0.0) ≈ 3.0 atol=1e-12

    # Modes below the cutoff are excluded from the support count.
    with_tiny_tail = [sqrt(0.9999999999), sqrt(1e-10 - 1e-15), 1e-15]
    @test natural_bonddim(with_tiny_tail; q=0.0, alpha=0.0, cutoff=1e-12) ≈ 2.0 atol=1e-12
end

@testset "ENTANGLEMENT_ENTROPY_NORMALIZES" begin
    # Unnormalized input should be normalized before cutoff
    S = [0.5, 0.5, 1e-11]
    # After normalization, 1e-11 becomes ~2e-11, which is below cutoff=1e-10
    # So it should be dropped, giving entropy of log(2)
    @test iTEBD.entanglement_entropy(S; cutoff=1e-10) ≈ log(2) atol=1e-8
end
