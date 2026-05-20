using BenchmarkTools
using LinearAlgebra
using Logging
using Printf
using Random
using iTEBD
using iTEBD: iMPS, canonical!, schmidt_canonical

# Suppress the low-rank-compression warnings that fire during canonicalization
# of random tensors; not relevant to the comparison.
Logging.disable_logging(Logging.Warn)

const SMOKE_MODE = "--smoke" in ARGS
const BENCH_SAMPLES = SMOKE_MODE ? 1 : 10
const BENCH_SECONDS = SMOKE_MODE ? 0.05 : 2.0

# This benchmark compares three canonicalization strategies for an iMPS being
# truncated from bond dim χ_in to χ_trunc:
#
# * `single`:    schmidt_canonical once with maxdim=χ_trunc. The state is
#                approximately canonical because the per-bond SVDs in
#                tensor_decomp! lose their gauge identity when truncating.
#                Fast but right-canonical error scales as the truncation error.
# * `full`:      schmidt_canonical twice. The second pass operates on the
#                already-truncated state and restores exact canonical form
#                without changing the physical state. This is what canonical!
#                does now.
# * `lq_sweep`:  an LQ-sweep gauge restoration (right-to-left LQ + wrap-around
#                SVD) attempted as a cheap replacement for `full`. The cost is
#                O(n·d·χ³) instead of the eigen-based fixed-point solve. In
#                practice it produces a *different* truncated state because
#                the wrap-around singular factor cannot be absorbed back into
#                either bordering site without breaking the per-site
#                right-isometric condition; we keep it in the benchmark to
#                document the empirical fidelity gap.

function build_template(n::Int, d::Int, chi_in::Int)
    Random.seed!(20260520)
    Γs = [randn(ComplexF64, chi_in, d, chi_in) for _ in 1:n]
    λs = [ones(Float64, chi_in) for _ in 1:n]
    ψ = iMPS(Γs, λs, n)
    canonical!(ψ; maxdim=chi_in, noninjective=:ignore)
    return ψ
end

function truncate_single!(ψ::iMPS, chi_trunc::Int)
    ψ.Γ[:], ψ.λ[:] = schmidt_canonical(
        ψ.Γ, ψ.λ[end];
        maxdim=chi_trunc, cutoff=iTEBD.SVDTOL, renormalize=true,
        noninjective=:ignore, symmetry_break=:none,
    )
    return ψ
end

function truncate_full!(ψ::iMPS, chi_trunc::Int)
    bond_before = length.(ψ.λ)
    ψ.Γ[:], ψ.λ[:] = schmidt_canonical(
        ψ.Γ, ψ.λ[end];
        maxdim=chi_trunc, cutoff=iTEBD.SVDTOL, renormalize=true,
        noninjective=:ignore, symmetry_break=:none,
    )
    if length.(ψ.λ) != bond_before
        ψ.Γ[:], ψ.λ[:] = schmidt_canonical(
            ψ.Γ, ψ.λ[end];
            maxdim=chi_trunc, cutoff=iTEBD.SVDTOL, renormalize=true,
            noninjective=:ignore, symmetry_break=:none,
        )
    end
    return ψ
end

function truncate_lq!(ψ::iMPS, chi_trunc::Int)
    bond_before = length.(ψ.λ)
    ψ.Γ[:], ψ.λ[:] = schmidt_canonical(
        ψ.Γ, ψ.λ[end];
        maxdim=chi_trunc, cutoff=iTEBD.SVDTOL, renormalize=true,
        noninjective=:ignore, symmetry_break=:none,
    )
    if length.(ψ.λ) != bond_before
        _lq_gauge_restore!(ψ.Γ, ψ.λ; renormalize=true)
    end
    return ψ
end

function _lq_gauge_restore!(Γs, λs; renormalize::Bool=true)
    n = length(Γs)
    T = eltype(Γs[1])
    L_carry::Union{Nothing,Matrix{T}} = nothing
    for i in n:-1:1
        Γi = Γs[i]
        Dl, d, Dr = size(Γi)
        if !isnothing(L_carry)
            Γi_mat = reshape(Γi, Dl * d, Dr)
            new_mat = Γi_mat * L_carry
            Dr = size(L_carry, 2)
            Γi = reshape(new_mat, Dl, d, Dr)
        end
        M = reshape(Γi, Dl, d * Dr)
        F = LinearAlgebra.lq(M)
        L = Matrix(F.L)
        Q = Matrix(F.Q)
        Γs[i] = reshape(Q, size(Q, 1), d, Dr)
        L_carry = L
    end
    isnothing(L_carry) && return Γs, λs
    F = svd(L_carry)
    U, S, V = F.U, F.S, F.V
    nrm = norm(S)
    S_out = renormalize && nrm > 0 ? S ./ nrm : S
    λs[n] = real.(S_out)
    Γn = Γs[n]; Dln, dn, Drn = size(Γn)
    Γs[n] = reshape(reshape(Γn, Dln * dn, Drn) * U, Dln, dn, size(U, 2))
    Γ1 = Γs[1]; Dl1, d1, Dr1 = size(Γ1)
    Γs[1] = reshape(adjoint(V) * reshape(Γ1, Dl1, d1 * Dr1), size(V, 2), d1, Dr1)
    return Γs, λs
end

function rc_error(ψ::iMPS)
    errs = Float64[]
    for Γ in ψ.Γ
        Dl = size(Γ, 1)
        overlap = zeros(ComplexF64, Dl, Dl)
        for s in axes(Γ, 2)
            Bs = reshape(Γ[:, s, :], Dl, size(Γ, 3))
            overlap .+= Bs * Bs'
        end
        push!(errs, norm(overlap - Matrix{ComplexF64}(I, Dl, Dl)))
    end
    return maximum(errs)
end

bench_time(f) = (f(); median(@benchmark $f() samples=BENCH_SAMPLES evals=1 seconds=BENCH_SECONDS).time / 1e6)

function main()
    println("Canonical-truncation benchmark")
    println("mode = $(SMOKE_MODE ? "smoke" : "full")")
    println("fidelity = |inner_product(truncated, template)| per unit cell; 1.0 = perfect.")
    println("rc_err   = max site right-canonical error.")
    println()
    @printf("%-6s %-6s %-6s %-8s %-10s %-10s %-10s %-9s %-9s %-9s %-12s %-12s %-12s\n",
            "n", "d", "χ_in", "χ_trunc",
            "single_ms", "full_ms", "lq_ms",
            "rc_sng", "rc_full", "rc_lq",
            "fid_single", "fid_full", "fid_lq")

    n, d = 4, 2
    bond_dims = SMOKE_MODE ? [(8, 4)] : [(16, 8), (32, 16), (64, 32), (128, 64)]

    for (chi_in, chi_trunc) in bond_dims
        ψ_template = build_template(n, d, chi_in)

        # One reference run of each strategy for the quality metrics.
        ψ_single = deepcopy(ψ_template); truncate_single!(ψ_single, chi_trunc)
        ψ_full   = deepcopy(ψ_template); truncate_full!(ψ_full, chi_trunc)
        ψ_lq     = deepcopy(ψ_template); truncate_lq!(ψ_lq, chi_trunc)

        rc_sng  = rc_error(ψ_single)
        rc_full = rc_error(ψ_full)
        rc_lq   = rc_error(ψ_lq)

        fid_single = abs(iTEBD.inner_product(ψ_single, ψ_template))
        fid_full   = abs(iTEBD.inner_product(ψ_full,   ψ_template))
        fid_lq     = abs(iTEBD.inner_product(ψ_lq,     ψ_template))

        # Timing — each run starts from a fresh deepcopy.
        t_single = bench_time(() -> truncate_single!(deepcopy(ψ_template), chi_trunc))
        t_full   = bench_time(() -> truncate_full!(deepcopy(ψ_template), chi_trunc))
        t_lq     = bench_time(() -> truncate_lq!(deepcopy(ψ_template), chi_trunc))

        @printf("%-6d %-6d %-6d %-8d %-10.3f %-10.3f %-10.3f %-9.1e %-9.1e %-9.1e %-12.8f %-12.8f %-12.8f\n",
                n, d, chi_in, chi_trunc,
                t_single, t_full, t_lq,
                rc_sng, rc_full, rc_lq,
                fid_single, fid_full, fid_lq)
    end

    SMOKE_MODE && println("canonical-truncation benchmark smoke validation passed")
end

main()
