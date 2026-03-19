@isdefined(iTEBD) || include("../src/iTEBD.jl")
using .iTEBD: iTEBD, natural_bonddim, adaptive_bonddim
using Test

@testset "NATURAL_BONDDIM" begin
    @test natural_bonddim([1.0, 0.0, 0.0]; alpha=0.0) ≈ 1.0 atol=1e-12

    bell = [inv(sqrt(2)), inv(sqrt(2))]
    @test natural_bonddim(bell; q=1.0, alpha=0.0) ≈ 2.0 atol=1e-12

    narrow = [sqrt(0.95), sqrt(0.05), 0.0]
    broad = [sqrt(0.50), sqrt(0.50), 0.0]
    @test natural_bonddim(broad; q=1.0, alpha=0.0) > natural_bonddim(narrow; q=1.0, alpha=0.0)

    tailed = [sqrt(0.8), sqrt(0.15), sqrt(0.05)]
    @test natural_bonddim(tailed; q=1.0, alpha=0.1) > natural_bonddim(tailed; q=1.0, alpha=0.0)
    @test natural_bonddim(tailed) ≈ natural_bonddim(tailed; q=1.0, alpha=0.1) atol=1e-12

    # Smaller q is more conservative: it assigns a larger natural bond dimension.
    spectrum = [sqrt(0.7), sqrt(0.2), sqrt(0.1)]
    @test natural_bonddim(spectrum; q=0.8, alpha=0.0) > natural_bonddim(spectrum; q=1.0, alpha=0.0)
    @test natural_bonddim(spectrum; q=1.0, alpha=0.0) > natural_bonddim(spectrum; q=2.0, alpha=0.0)
end

@testset "ADAPTIVE_BONDDIM" begin
    spectrum = [sqrt(0.6), sqrt(0.25), sqrt(0.15)]

    χ_raw = natural_bonddim(spectrum)
    @test adaptive_bonddim(1, spectrum; mindim=2, maxdim=10) == max(2, ceil(Int, χ_raw))

    @test adaptive_bonddim(7, spectrum; mindim=2, maxdim=10) == 7
    @test adaptive_bonddim(1, fill(0.5, 8); mindim=2, maxdim=3) == 3
end
