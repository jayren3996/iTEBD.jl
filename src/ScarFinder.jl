#---------------------------------------------------------------------------------------------------
# ScarFinder
#---------------------------------------------------------------------------------------------------
export operator_span, energy_density, energy_span
export scarfinder_step!, scarfinder!
export floquet_scarfinder_step!, floquet_scarfinder!

"""
    operator_span(ψ, O)

Infer how many sites the local operator `O` acts on from the local dimension of `ψ`.

This assumes a uniform local Hilbert-space dimension across the unit cell. An
`ArgumentError` is thrown if `size(O, 1)` is not an exact power of the local dimension.
"""
function operator_span(ψ::iMPS, O::AbstractMatrix)
    size(O, 1) == size(O, 2) || throw(ArgumentError("operator must be square"))
    d = size(ψ.Γ[1], 2)
    dim = size(O, 1)
    span = 0
    block = 1
    while block < dim
        block *= d
        span += 1
    end
    block == dim || throw(ArgumentError("operator dimension $dim is incompatible with local dimension $d"))
    return span
end

function _evolve_uniform!(ψ::iMPS, G::AbstractMatrix; span::Integer, maxdim::Integer=MAXDIM)
    span >= 1 || throw(ArgumentError("span must be positive"))
    offset = span - 1
    for i in 1:ψ.n
        applygate!(ψ, G, i, mod(i + offset - 1, ψ.n) + 1; maxdim)
    end
    return ψ
end

function _truncate_unitcell!(ψ::iMPS, χ::Integer; cutoff::Real=SVDTOL)
    Γ = tensor_group(ψ.Γ)
    A, λ = schmidt_canonical(Γ, ψ.λ[end]; maxdim=χ, cutoff, renormalize=true)
    tensor_lmul!(λ, A)
    Γs, λs = tensor_decomp!(A, λ, ψ.n; maxdim=χ, cutoff, renormalize=true)
    push!(λs, λ)
    ψ.Γ .= Γs
    ψ.λ .= λs
    canonical!(ψ; maxdim=χ, cutoff, renormalize=true)
    return ψ
end

function _apply_sequence!(
    ψ::iMPS,
    gates::AbstractVector{<:AbstractMatrix};
    spans::AbstractVector{<:Integer},
    maxdim::Integer=MAXDIM
)
    length(gates) == length(spans) || throw(ArgumentError("gates and spans must have the same length"))
    for (G, span) in zip(gates, spans)
        _evolve_uniform!(ψ, G; span, maxdim)
    end
    return ψ
end

_normalize_spans(ψ::iMPS, gates::AbstractVector{<:AbstractMatrix}, spans::Nothing) =
    [operator_span(ψ, G) for G in gates]

function _normalize_spans(
    ψ::iMPS,
    gates::AbstractVector{<:AbstractMatrix},
    spans::Union{Integer,AbstractVector{<:Integer}}
)
    vals = spans isa Integer ? fill(spans, length(gates)) : collect(Int, spans)
    length(vals) == length(gates) || throw(ArgumentError("gates and spans must have the same length"))
    return vals
end

"""
    energy_density(ψ, h; span=operator_span(ψ, h))

Return the unit-cell averaged expectation value of the local operator `h`.

This is the average of `expect(ψ, h, i, i + span - 1)` over all starting sites
in the unit cell.
"""
function energy_density(ψ::iMPS, h::AbstractMatrix; span::Integer=operator_span(ψ, h))
    offset = span - 1
    sum(expect(ψ, h, i, mod(i + offset - 1, ψ.n) + 1) for i in 1:ψ.n) / ψ.n
end

"""
    energy_span(n, d, h; dτ=0.1, Nτ=1000, maxdim=32)

Estimate the lowest- and highest-energy density reachable by imaginary-time iTEBD
using the local operator `h`.

This helper is useful for choosing a target energy when using `scarfinder!`.
The result is `(Emin, Emax, (ψmin, ψmax))`.
"""
function energy_span(
    n::Integer,
    d::Integer,
    h::AbstractMatrix;
    dτ::Real=0.1,
    Nτ::Integer=1000,
    maxdim::Integer=32
)
    ψmin = rand_iMPS(ComplexF64, n, d, 1)
    Gmin = exp(-dτ * h)
    span = operator_span(ψmin, h)
    for _ in 1:Nτ
        _evolve_uniform!(ψmin, Gmin; span, maxdim)
    end
    canonical!(ψmin)
    Emin = energy_density(ψmin, h; span)

    ψmax = rand_iMPS(ComplexF64, n, d, 1)
    Gmax = exp(dτ * h)
    for _ in 1:Nτ
        _evolve_uniform!(ψmax, Gmax; span, maxdim)
    end
    canonical!(ψmax)
    Emax = energy_density(ψmax, h; span)

    return Emin, Emax, (ψmin, ψmax)
end

function _energy_fix!(
    ψ::iMPS,
    h::AbstractMatrix,
    χ::Integer;
    span::Integer,
    target::Real,
    tol::Real=1e-6,
    α::Real=0.1,
    maxstep::Integer=50
)
    dE = energy_density(ψ, h; span) - target
    abs(dE) < tol && return ψ
    dτ = α * dE
    G = exp(-dτ * h)
    for _ in 1:maxstep
        _evolve_uniform!(ψ, G; span, maxdim=χ)
        dE2 = energy_density(ψ, h; span) - target
        if dE * dE2 < 0
            k = abs(dE2) / (abs(dE) + abs(dE2))
            _evolve_uniform!(ψ, exp(k * dτ * h); span, maxdim=χ)
            return ψ
        elseif abs(dE2) > abs(dE)
            _evolve_uniform!(ψ, exp(dτ * h); span, maxdim=χ)
            return ψ
        end
        dE = dE2
    end
    return ψ
end

function _minimize_on_trajectory!(f, step!, ψ::iMPS, samples::Integer)
    ψtrial = deepcopy(ψ)
    values = zeros(Float64, samples)
    for i in 1:samples
        step!(ψtrial)
        values[i] = f(ψtrial)
    end
    _, index = findmin(values)
    for _ in 1:index
        step!(ψ)
    end
    return ψ
end

"""
    scarfinder_step!(ψ, h, dt, χ; keywords...)

Perform one hybrid ScarFinder step:

1. real-time evolution under `h`,
2. truncation back to bond dimension `χ`,
3. optional imaginary-time correction to match a target energy density.

Keyword arguments:
- `nstep=10`: number of real-time TEBD steps before truncation.
- `maxdim=MAXDIM`: temporary bond dimension used during the real-time evolution.
- `span=operator_span(ψ, h)`: number of sites acted on by `h`.
- `target=nothing`: target energy density. If not `nothing`, an energy correction is applied.
- `tol=1e-6`, `α=0.1`, `maxstep=50`: parameters for the energy correction.
- `cutoff=SVDTOL`: truncation cutoff used when compressing back to bond dimension `χ`.
"""
function scarfinder_step!(
    ψ::iMPS,
    h::AbstractMatrix,
    dt::Real,
    χ::Integer;
    nstep::Integer=10,
    maxdim::Integer=MAXDIM,
    span::Integer=operator_span(ψ, h),
    target::Union{Real,Nothing}=nothing,
    tol::Real=1e-6,
    α::Real=0.1,
    maxstep::Integer=50,
    cutoff::Real=SVDTOL
)
    G = exp(-1im * dt * h)
    for _ in 1:nstep
        _evolve_uniform!(ψ, G; span, maxdim)
    end
    _truncate_unitcell!(ψ, χ; cutoff)
    isnothing(target) || _energy_fix!(ψ, h, χ; span, target, tol, α, maxstep)
    return ψ
end

"""
    scarfinder!(ψ, h, dt, χ, N; keywords...)

Run `N` hybrid ScarFinder iterations in place.

By default the final state is additionally refined by scanning a shorter-time
trajectory and selecting the minimum-entanglement point, which matches the workflow
used in the reference `ScarFinder` scripts.

Keyword arguments:
- `refine=true`: enable the minimum-entanglement trajectory scan.
- `refine_dt=dt/10`: time step used during the refinement scan.
- `refine_step=1000`: number of trial points in the refinement scan.

All keyword arguments accepted by `scarfinder_step!` are also supported.

The return value is `ψ`, mutated in place.
"""
function scarfinder!(
    ψ::iMPS,
    h::AbstractMatrix,
    dt::Real,
    χ::Integer,
    N::Integer;
    refine::Bool=true,
    refine_dt::Real=dt / 10,
    refine_step::Integer=1000,
    kwargs...
)
    for _ in 1:N
        scarfinder_step!(ψ, h, dt, χ; kwargs...)
    end
    if refine
        step! = ψ0 -> scarfinder_step!(ψ0, h, refine_dt, χ; kwargs...)
        _minimize_on_trajectory!(x -> ent_S(x, x.n), step!, ψ, refine_step)
    end
    return ψ
end

"""
    floquet_scarfinder_step!(ψ, U, χ; keywords...)
    floquet_scarfinder_step!(ψ, Us, χ; keywords...)

Perform one Floquet ScarFinder step.

This is the Floquet analogue of `scarfinder_step!`. Instead of generating
real-time evolution from a static Hamiltonian, it applies one Floquet period
described by either:
- a single local Floquet gate `U`, or
- a sequence of local gates `Us` making up one drive period.

After one Floquet period, the state is truncated back to bond dimension `χ`.

Keyword arguments:
- `ncycle=1`: number of Floquet periods to apply before truncation.
- `span` / `spans`: operator support of the gate or gates. If omitted, each span
  is inferred with `operator_span`.
- `maxdim=MAXDIM`: temporary bond dimension used during the unitary evolution.
- `cutoff=SVDTOL`: truncation cutoff used when compressing back to bond dimension `χ`.
"""
function floquet_scarfinder_step!(
    ψ::iMPS,
    U::AbstractMatrix,
    χ::Integer;
    ncycle::Integer=1,
    span::Union{Integer,Nothing}=nothing,
    maxdim::Integer=MAXDIM,
    cutoff::Real=SVDTOL
)
    spans = isnothing(span) ? [operator_span(ψ, U)] : [Int(span)]
    gates = [U]
    for _ in 1:ncycle
        _apply_sequence!(ψ, gates; spans, maxdim)
    end
    _truncate_unitcell!(ψ, χ; cutoff)
    return ψ
end

function floquet_scarfinder_step!(
    ψ::iMPS,
    Us::AbstractVector{<:AbstractMatrix},
    χ::Integer;
    ncycle::Integer=1,
    spans::Union{Nothing,Integer,AbstractVector{<:Integer}}=nothing,
    maxdim::Integer=MAXDIM,
    cutoff::Real=SVDTOL
)
    local_spans = _normalize_spans(ψ, Us, spans)
    for _ in 1:ncycle
        _apply_sequence!(ψ, Us; spans=local_spans, maxdim)
    end
    _truncate_unitcell!(ψ, χ; cutoff)
    return ψ
end

"""
    floquet_scarfinder!(ψ, U, χ, N; keywords...)
    floquet_scarfinder!(ψ, Us, χ, N; keywords...)

Run `N` Floquet ScarFinder iterations in place.

This routine repeatedly applies one Floquet period, truncates back to bond
dimension `χ`, and optionally refines the final answer by selecting the
minimum-entanglement point on a short Floquet trajectory.

Keyword arguments:
- `refine=true`: enable the minimum-entanglement refinement scan.
- `refine_step=1000`: number of trial points used in the refinement scan.

All keyword arguments accepted by `floquet_scarfinder_step!` are also supported.
The return value is `ψ`, mutated in place.
"""
function floquet_scarfinder!(
    ψ::iMPS,
    U::AbstractMatrix,
    χ::Integer,
    N::Integer;
    refine::Bool=true,
    refine_step::Integer=1000,
    kwargs...
)
    for _ in 1:N
        floquet_scarfinder_step!(ψ, U, χ; kwargs...)
    end
    if refine
        step! = ψ0 -> floquet_scarfinder_step!(ψ0, U, χ; kwargs...)
        _minimize_on_trajectory!(x -> ent_S(x, x.n), step!, ψ, refine_step)
    end
    return ψ
end

function floquet_scarfinder!(
    ψ::iMPS,
    Us::AbstractVector{<:AbstractMatrix},
    χ::Integer,
    N::Integer;
    refine::Bool=true,
    refine_step::Integer=1000,
    kwargs...
)
    for _ in 1:N
        floquet_scarfinder_step!(ψ, Us, χ; kwargs...)
    end
    if refine
        step! = ψ0 -> floquet_scarfinder_step!(ψ0, Us, χ; kwargs...)
        _minimize_on_trajectory!(x -> ent_S(x, x.n), step!, ψ, refine_step)
    end
    return ψ
end
