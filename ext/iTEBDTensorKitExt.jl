module iTEBDTensorKitExt

using iTEBD
using TensorKit
using LinearAlgebra

# Names from the base package that this extension will specialise. Using
# `import` (not `using`) so that adding methods to these names is unambiguous
# to the compiler.
import iTEBD: graded_space, spin_half_ops, schmidt_values
import iTEBD: rand_iMPS, product_iMPS
import iTEBD: iMPS, _validate_iMPS_bonds

# `SymmetricIMPS` — the TensorKit-backed variant of `iMPS`. The exact
# `AbstractTensorMap` parameters are deliberately unconstrained here so the
# alias matches any U(1)/Z_N/product-sector graded-space tensors that later
# constructors produce. The wraparound bond-space check (Chunk 4) and all
# algorithm specialisations (Chunks 5-7) dispatch on this alias.
const SymmetricIMPS = iMPS{<:AbstractTensorMap, <:DiagonalTensorMap}

# This module is currently a skeleton. Method bodies are populated by
# subsequent chunks of the implementation plan:
#   Chunk 3 — graded_space, spin_half_ops, schmidt_values
#   Chunk 4 — rand_iMPS / product_iMPS symmetric constructors,
#             _validate_iMPS_bonds specialisation
#   Chunks 5-7 — canonical!, applygate!, expect, energy_density, ent_S

end # module
