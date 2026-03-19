#---------------------------------------------------------------------------------------------------
# ScarFinder
#---------------------------------------------------------------------------------------------------
export operator_span, energy_density, energy_span
export scarfinder_step!, scarfinder!

"""
    operator_span(ψ, O)

Infer how many sites the local operator `O` acts on from the local dimension of `ψ`.

This assumes a uniform local Hilbert-space dimension across the unit cell. An
`ArgumentError` is thrown if `size(O, 1)` is not an exact power of the local dimension.

Parameters:
- `ψ`
  Reference state used only to infer the local Hilbert-space dimension.
- `O`
  Square dense local operator.

Returns:
- Number of sites on which `O` acts.

Example:
```julia
psi = product_iMPS(ComplexF64, [[1, 0], [0, 1]])
span = operator_span(psi, kron([1 0; 0 -1], [1 0; 0 -1]))
```
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

"""
    _evolve_uniform!(ψ, G; span, maxdim=MAXDIM)

Apply the same local gate `G` at every translation of a periodic unit cell.

Parameters:
- `ψ`
  State updated in place.
- `G`
  Local gate to apply uniformly.

Keyword arguments:
- `span`
  Number of sites acted on by `G`.
- `maxdim=MAXDIM`
  Maximum temporary bond dimension used by each local update.

Returns:
- The same object `ψ`, mutated in place.

Notes:
- This is an internal helper used by the ScarFinder routines.
"""
function _evolve_uniform!(ψ::iMPS, G::AbstractMatrix; span::Integer, maxdim::Integer=MAXDIM)
    span >= 1 || throw(ArgumentError("span must be positive"))
    offset = span - 1
    for i in 1:ψ.n
        applygate!(ψ, G, i, mod(i + offset - 1, ψ.n) + 1; maxdim)
    end
    return ψ
end

"""
    _truncate_unitcell!(ψ, χ; cutoff=SVDTOL)

Project a full unit cell back to bond dimension `χ`.

Parameters:
- `ψ`
  State updated in place.
- `χ`
  Target bond dimension after projection.

Keyword arguments:
- `cutoff=SVDTOL`
  Singular-value cutoff used during truncation.

Returns:
- The same object `ψ`, mutated in place.

Notes:
- This groups the unit cell, canonicalizes it, decomposes it back into local
  tensors, and finishes with a canonicalization pass.
"""
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

"""
    _warn_scarfinder_nstep(nstep, mode)

Emit a one-time warning when `nstep == 1` in a ScarFinder real-time stage.

This is an internal helper used to flag the degenerate case where one
ScarFinder iteration contains only a single microscopic evolution step before
projection. That setting is still allowed for backwards compatibility, but it
is usually too aggressive to represent the intended coarse-grained ScarFinder
flow.
"""
function _warn_scarfinder_nstep(nstep::Integer, mode::Symbol)
    nstep >= 1 || throw(ArgumentError("nstep must be positive"))
    if nstep == 1
        label = mode === :hamiltonian ? "Hamiltonian-based" : "Gate-based"
        @warn "$label ScarFinder is using nstep = 1. This is usually too small: one ScarFinder iteration should normally contain multiple microscopic evolution steps before projection. Consider choosing nstep > 1 to avoid an overly aggressive truncation cycle." maxlog=1
    end
    return nothing
end

"""
    energy_density(ψ, h; span=operator_span(ψ, h))

Return the unit-cell averaged expectation value of the local operator `h`.

This is the average of `expect(ψ, h, i, i + span - 1)` over all starting sites
in the unit cell.

Parameters:
- `ψ`
  State in which the expectation value is measured.
- `h`
  Local operator interpreted as a Hamiltonian density or other local observable.

Keyword arguments:
- `span=operator_span(ψ, h)`
  Number of sites acted on by `h`.

Returns:
- Unit-cell averaged expectation value of `h`.

Notes:
- This is the quantity used by ScarFinder when `target` energy fixing is
  enabled.
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

Parameters:
- `n`
  Unit-cell length of the trial states used in the estimate.
- `d`
  Local Hilbert-space dimension.
- `h`
  Local Hamiltonian density.

Keyword arguments:
- `dτ=0.1`
  Imaginary-time step used to build `exp(∓dτ * h)`.
- `Nτ=1000`
  Number of imaginary-time steps used for the low- and high-energy estimates.
- `maxdim=32`
  Maximum temporary bond dimension used during those imaginary-time updates.

Returns:
- `(Emin, Emax, (ψmin, ψmax))` where `Emin` and `Emax` are the estimated energy
  densities and `ψmin`, `ψmax` are the corresponding trial states.

Notes:
- This is a heuristic helper, not a rigorous variational bound.
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

"""
    _energy_fix!(ψ, h, χ; span, target, tol=1e-6, α=0.1, maxstep=50)

Apply the internal ScarFinder energy-correction step.

Parameters:
- `ψ`
  State updated in place.
- `h`
  Local Hamiltonian density used to measure the energy drift.
- `χ`
  Bond dimension used during the correction steps.

Keyword arguments:
- `span`
  Number of sites acted on by `h`.
- `target`
  Target energy density.
- `tol=1e-6`
  Desired absolute tolerance on the energy density.
- `α=0.1`
  Step-size parameter for the imaginary-time correction.
- `maxstep=50`
  Maximum number of substeps in the correction loop.

Returns:
- The same object `ψ`, mutated in place.

Notes:
- This is an internal helper. Most users should control energy fixing through
  `target`, `tol`, `α`, and `maxstep` on the public ScarFinder interfaces.
"""
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

"""
    _minimize_on_trajectory!(f, step!, ψ, samples)

Advance along a trial trajectory and keep the sampled point minimizing `f`.

Parameters:
- `f`
  Scalar objective evaluated on trial states.
- `step!`
  In-place evolution function used to advance the trial state by one step.
- `ψ`
  State updated in place to the minimizing sampled point.
- `samples`
  Number of sampled points along the trajectory.

Returns:
- The same object `ψ`, advanced to the minimizing sampled point.

Notes:
- This is the internal refinement kernel used by the public `scarfinder!`
  routines.
"""
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

When `dt` is a microscopic gate time but each ScarFinder iteration is intended to
represent a larger physical interval `Δt`, set `nstep ≈ Δt / dt`. For example,
`dt = 0.01` and `Δt = 0.1` should typically use `nstep = 10`.

As a practical rule, `nstep = 1` is usually too small for ScarFinder because it
reduces one iteration to a single microscopic evolution step before projection.
The implementation therefore warns when `nstep == 1`, even though that setting
is still accepted for backwards compatibility.

Parameters:
- `ψ`
  Trial state updated in place.
- `h`
  Local Hamiltonian density used both for real-time evolution and, if enabled,
  energy fixing.
- `dt`
  Microscopic real-time step used to build `exp(-1im * dt * h)`.
- `χ`
  Target bond dimension after the projection step.

Keyword arguments:
- `nstep=10`
  Number of microscopic evolution steps before the projection back to `χ`.
- `maxdim=MAXDIM`
  Maximum temporary bond dimension during the real-time evolution stage.
- `span=operator_span(ψ, h)`
  Number of sites acted on by `h`.
- `target=nothing`
  Target energy density. If `nothing`, no energy-fixing step is performed.
- `tol=1e-6`, `α=0.1`, `maxstep=50`
  Parameters controlling the energy-fixing loop.
- `cutoff=SVDTOL`
  Singular-value cutoff used when truncating back to bond dimension `χ`.

Returns:
- The same object `ψ`, mutated in place.
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
    _warn_scarfinder_nstep(nstep, :hamiltonian)
    G = exp(-1im * dt * h)
    for _ in 1:nstep
        _evolve_uniform!(ψ, G; span, maxdim)
    end
    _truncate_unitcell!(ψ, χ; cutoff)
    isnothing(target) || _energy_fix!(ψ, h, χ; span, target, tol, α, maxstep)
    return ψ
end

"""
    scarfinder_step!(ψ, G, χ; keywords...)

Perform one gate-based ScarFinder step.

This variant is useful when the driving object is already given as a local gate,
for example a projected Floquet-like step such as `G = P * U`. The routine:

1. applies the gate `G`,
2. truncates the state back to bond dimension `χ`.

Unlike the Hamiltonian-based method, this variant does not perform energy fixing.

Keyword arguments:
- `nstep=10`: number of times to apply `G` before truncation.
- `maxdim=MAXDIM`: temporary bond dimension used during the gate evolution.
- `span=operator_span(ψ, G)`: number of sites acted on by `G`.
- `cutoff=SVDTOL`: truncation cutoff used when compressing back to bond dimension `χ`.

This gate-only variant does not perform energy fixing. If the intended workflow
uses a custom gate for evolution but still needs Hamiltonian-based energy
correction, prefer `scarfinder_step!(ψ, G, h, χ; ...)`.

As a practical rule, `nstep = 1` is usually too small for ScarFinder because it
collapses one iteration into a single microscopic gate application followed by
projection. The implementation therefore warns when `nstep == 1`, even though
that setting is still accepted for backwards compatibility.

Parameters:
- `ψ`
  Trial state updated in place.
- `G`
  Local gate applied during the real-time stage.
- `χ`
  Target bond dimension after truncation.

Keyword arguments:
- `nstep=10`
  Number of gate applications before truncation.
- `maxdim=MAXDIM`
  Maximum temporary bond dimension during those gate applications.
- `span=operator_span(ψ, G)`
  Number of sites acted on by `G`.
- `cutoff=SVDTOL`
  Singular-value cutoff used when truncating back to bond dimension `χ`.

Returns:
- The same object `ψ`, mutated in place.
"""
function scarfinder_step!(
    ψ::iMPS,
    G::AbstractMatrix,
    χ::Integer;
    nstep::Integer=10,
    maxdim::Integer=MAXDIM,
    span::Integer=operator_span(ψ, G),
    cutoff::Real=SVDTOL
)
    _warn_scarfinder_nstep(nstep, :gate)
    for _ in 1:nstep
        _evolve_uniform!(ψ, G; span, maxdim)
    end
    _truncate_unitcell!(ψ, χ; cutoff)
    return ψ
end

"""
    scarfinder_step!(ψ, G, h, χ; keywords...)

Perform one gate-based ScarFinder step with Hamiltonian-based energy fixing.

This variant is designed for cases where the update rule is given by a local gate
`G`, but the target constraint should still be imposed with respect to a local
Hamiltonian `h`. A typical example is a projected update such as `G = P * U`
in the PXP model.

The routine:

1. applies the gate `G`,
2. truncates the state back to bond dimension `χ`,
3. optionally applies the same energy-fixing step used by the Hamiltonian-based
   ScarFinder, now measured with `h`.

Keyword arguments:
- `nstep=10`: number of times to apply `G` before truncation.
- `maxdim=MAXDIM`: temporary bond dimension used during the gate evolution.
- `span=operator_span(ψ, G)`: number of sites acted on by `G`.
- `hspan=operator_span(ψ, h)`: number of sites acted on by `h`.
- `target=nothing`: target energy density. If not `nothing`, an energy correction is applied.
- `tol=1e-6`, `α=0.1`, `maxstep=50`: parameters for the energy correction.
- `cutoff=SVDTOL`: truncation cutoff used when compressing back to bond dimension `χ`.

For constrained models such as PXP, it is common to choose `G` as a projected
update that restores the physical subspace after truncation, while `h` remains
the unprojected local Hamiltonian density used for energy fixing. These two
objects should generally not be conflated.

As with the Hamiltonian-based interface, if `G` is built from a microscopic time
step `dt` but one ScarFinder iteration should represent a larger interval `Δt`,
set `nstep ≈ Δt / dt`.

As a practical rule, `nstep = 1` is usually too small for ScarFinder because it
reduces each iteration to a single microscopic gate application before
projection. The implementation therefore warns when `nstep == 1`, even though
that setting is still accepted for backwards compatibility.

Parameters:
- `ψ`
  Trial state updated in place.
- `G`
  Local gate used for the real-time evolution stage.
- `h`
  Local Hamiltonian density used only for energy fixing.
- `χ`
  Target bond dimension after the projection step.

Keyword arguments:
- `nstep=10`
  Number of gate applications before truncation.
- `maxdim=MAXDIM`
  Maximum temporary bond dimension during the gate evolution.
- `span=operator_span(ψ, G)`
  Number of sites acted on by `G`.
- `hspan=operator_span(ψ, h)`
  Number of sites acted on by `h` when measuring and correcting the energy.
- `target=nothing`
  Target energy density. If `nothing`, no energy correction is applied.
- `tol=1e-6`, `α=0.1`, `maxstep=50`
  Parameters controlling the energy-fixing loop.
- `cutoff=SVDTOL`
  Singular-value cutoff used during projection back to bond dimension `χ`.

Returns:
- The same object `ψ`, mutated in place.
"""
function scarfinder_step!(
    ψ::iMPS,
    G::AbstractMatrix,
    h::AbstractMatrix,
    χ::Integer;
    nstep::Integer=10,
    maxdim::Integer=MAXDIM,
    span::Integer=operator_span(ψ, G),
    hspan::Integer=operator_span(ψ, h),
    target::Union{Real,Nothing}=nothing,
    tol::Real=1e-6,
    α::Real=0.1,
    maxstep::Integer=50,
    cutoff::Real=SVDTOL
)
    _warn_scarfinder_nstep(nstep, :gate)
    for _ in 1:nstep
        _evolve_uniform!(ψ, G; span, maxdim)
    end
    _truncate_unitcell!(ψ, χ; cutoff)
    isnothing(target) || _energy_fix!(ψ, h, χ; span=hspan, target, tol, α, maxstep)
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

Parameters:
- `ψ`
  Trial state updated in place.
- `h`
  Local Hamiltonian density.
- `dt`
  Microscopic real-time step used in each hybrid ScarFinder iteration.
- `χ`
  Target bond dimension of the projected manifold.
- `N`
  Number of ScarFinder iterations.

Keyword arguments:
- `refine=true`
  Whether to perform the post-processing scan for a minimum-entanglement point.
- `refine_dt=dt / 10`
  Microscopic step used during that refinement scan.
- `refine_step=1000`
  Number of trial points considered in the refinement scan.
- `kwargs...`
  Additional keyword arguments forwarded to [`scarfinder_step!`](@ref).

Returns:
- The same object `ψ`, mutated in place.
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
    scarfinder!(ψ, G, χ, N; keywords...)

Run `N` gate-based ScarFinder iterations in place.

This is the high-level companion of `scarfinder_step!(ψ, G, χ; ...)` and is
intended for cases where the update rule is already provided as a local gate,
for example `G = P * U` in constrained or projected dynamics.

Keyword arguments:
- `refine=true`: enable the minimum-entanglement trajectory scan.
- `refine_step=1000`: number of trial points in the refinement scan.

All keyword arguments accepted by `scarfinder_step!(ψ, G, χ; ...)` are also supported.
The return value is `ψ`, mutated in place.

Parameters:
- `ψ`
  Trial state updated in place.
- `G`
  Local gate used in the real-time stage.
- `χ`
  Target bond dimension after each projection.
- `N`
  Number of ScarFinder iterations.

Keyword arguments:
- `refine=true`
  Whether to perform the minimum-entanglement refinement scan.
- `refine_step=1000`
  Number of trial points used in that scan.
- `kwargs...`
  Additional keyword arguments forwarded to `scarfinder_step!(ψ, G, χ; ...)`.

Returns:
- The same object `ψ`, mutated in place.
"""
function scarfinder!(
    ψ::iMPS,
    G::AbstractMatrix,
    χ::Integer,
    N::Integer;
    refine::Bool=true,
    refine_step::Integer=1000,
    kwargs...
)
    for _ in 1:N
        scarfinder_step!(ψ, G, χ; kwargs...)
    end
    if refine
        step! = ψ0 -> scarfinder_step!(ψ0, G, χ; kwargs...)
        _minimize_on_trajectory!(x -> ent_S(x, x.n), step!, ψ, refine_step)
    end
    return ψ
end

"""
    scarfinder!(ψ, G, h, χ, N; keywords...)

Run `N` gate-based ScarFinder iterations in place while using `h` for energy fixing.

This is the high-level interface for projected or constrained dynamics where the
update rule is already encoded in a gate `G`, but the energy target should still
be measured using a Hamiltonian density `h`.

Keyword arguments:
- `refine=true`: enable the minimum-entanglement trajectory scan.
- `refine_step=1000`: number of trial points in the refinement scan.

All keyword arguments accepted by `scarfinder_step!(ψ, G, h, χ; ...)` are also supported.
The return value is `ψ`, mutated in place.

Parameters:
- `ψ`
  Trial state updated in place.
- `G`
  Local gate used for the real-time evolution stage.
- `h`
  Local Hamiltonian density used for energy fixing.
- `χ`
  Target bond dimension after each projection.
- `N`
  Number of ScarFinder iterations.

Keyword arguments:
- `refine=true`
  Whether to perform the minimum-entanglement refinement scan.
- `refine_step=1000`
  Number of trial points used in that scan.
- `kwargs...`
  Additional keyword arguments forwarded to
  `scarfinder_step!(ψ, G, h, χ; ...)`.

Returns:
- The same object `ψ`, mutated in place.
"""
function scarfinder!(
    ψ::iMPS,
    G::AbstractMatrix,
    h::AbstractMatrix,
    χ::Integer,
    N::Integer;
    refine::Bool=true,
    refine_step::Integer=1000,
    kwargs...
)
    for _ in 1:N
        scarfinder_step!(ψ, G, h, χ; kwargs...)
    end
    if refine
        step! = ψ0 -> scarfinder_step!(ψ0, G, h, χ; kwargs...)
        _minimize_on_trajectory!(x -> ent_S(x, x.n), step!, ψ, refine_step)
    end
    return ψ
end
