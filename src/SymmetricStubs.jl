#---------------------------------------------------------------------------------------------------
# Symmetric stubs
#
# Names declared here are real-implemented inside `ext/iTEBDTensorKitExt.jl`. In
# the base package they throw a clear message telling the user to `using
# TensorKit` first. The extension shadows them with real methods automatically
# when TensorKit is loaded.
#---------------------------------------------------------------------------------------------------
export graded_space, spin_half_ops, schmidt_values, SymmetricIMPS

# `SymmetricIMPS` — the TensorKit-backed variant of `iMPS`. Declared here so
# that the name is in the `iTEBD` namespace and accessible as `iTEBD.SymmetricIMPS`
# in tests/user code even before TensorKit is loaded.
#
# In the base package, we give it the widest possible type parameters
# (`Any, Any`) so that `ψ isa SymmetricIMPS` is still meaningful: it matches
# any `iMPS`, acting as a no-op sentinel. The TensorKit extension narrows this
# definition to `iMPS{<:AbstractTensorMap, <:DiagonalTensorMap}` at load time
# by re-exporting the tighter alias.
#
# The practical effect: `ψ isa iTEBD.SymmetricIMPS` works correctly in both the
# "no TensorKit" case (true for all iMPS, acceptable) and the "TensorKit loaded"
# case (the extension's export shadows this stub in the local scope via `using`).
const SymmetricIMPS = iMPS

const _NEEDS_TENSORKIT = """
This entry point requires the TensorKit backend. Run `using TensorKit` in your \
session (or add it to your project) to load the iTEBD symmetric extension.\
"""

graded_space(args...; kwargs...) = error(_NEEDS_TENSORKIT)
spin_half_ops(args...; kwargs...) = error(_NEEDS_TENSORKIT)

"""
    schmidt_values(ψ, i)

Return the Schmidt spectrum on bond `i` as a `Vector{Float64}`, independent of
whether `ψ` is dense or symmetric.

For dense `iMPS`, this is a thin wrapper around `ψ.λ[i]`. For the symmetric
backend (loaded via `using TensorKit`), the extension's specialisation
flattens the per-sector diagonal blocks of `ψ.λ[i]::DiagonalTensorMap` into a
single descending-sorted `Vector{Float64}`.
"""
schmidt_values(ψ::DenseIMPS, i::Integer) = convert(Vector{Float64}, ψ.λ[i])

# Symbol-dispatch stubs for symmetric constructors. The non-symbol methods of
# `rand_iMPS` / `product_iMPS` are defined in `iMPS.jl` and remain dense-only;
# only the `(sym::Symbol, …)` overloads are added here as fail-fast errors.
rand_iMPS(::Symbol, args...; kwargs...) = error(_NEEDS_TENSORKIT)
product_iMPS(::Symbol, args...; kwargs...) = error(_NEEDS_TENSORKIT)
