#---------------------------------------------------------------------------------------------------
# QUANTUM GATE
#---------------------------------------------------------------------------------------------------
"""
    tensor_applygate!(G, Γs, λl; keywords...)

Apply Gate:
    |
    G
    |              |   |        |
  --Γ--  ==> --λl--Γ₁--Γ₂-- ⋯ --Γₙ--,
where:
    |          |
  --Γₙ--  =  --Aₙ--λₙ-- 

Return list tensor list [Γ₁,⋯,Γₙ], and values list [λ₁,⋯,λₙ₋₁].
"""
function tensor_applygate!(
    G::AbstractMatrix{<:Number}, Γs::AbstractVector{<:AbstractArray{<:Number, 3}},
    λl::AbstractVector{<:Number};
    maxdim=MAXDIM, cutoff=SVDTOL, renormalize=false
)
    n = length(Γs)
    isone(n) && return ([GΓ], [])
    Γ = tensor_group(Γs)
    tensor_lmul!(λl, Γ)
    GΓ = tensor_umul(G, Γ)
    tensor_decomp!(GΓ, λl, n; maxdim, cutoff, renormalize)
end
#---------------------------------------------------------------------------------------------------
export applygate!
function applygate!(
    ψ::iMPS, G::AbstractMatrix,
    i::Integer, j::Integer;
    maxdim::Integer=MAXDIM, 
    cutoff::Real=SVDTOL, 
    renormalize::Bool=true
)
    if isequal(i, j)
        ψ.Γ[i] = tensor_umul(G, ψ.Γ[i])
        return ψ
    end
    inds = j>i ? collect(i:j) : [i:ψ.n; 1:j]
    Γs = ψ.Γ[inds]
    λl = ψ.λ[mod(i-2,ψ.n)+1]
    Γs, λs = tensor_applygate!(G, Γs, λl; maxdim, cutoff, renormalize)
    push!(λs, ψ.λ[j])
    for i in eachindex(inds) 
        ψ.Γ[inds[i]] = Γs[i]
        ψ.λ[inds[i]] = λs[i]
    end
    return ψ
end

function _gate_indices(ψ::iMPS, i::Integer, j::Integer)
    j > i ? collect(i:j) : [i:ψ.n; 1:j]
end

export evolve!
"""
    evolve!(ψ, gates, steps; chi_policy=:fixed, maxdim=MAXDIM, mindim=1, q=1.5, alpha=0.5, cutoff=SVDTOL, renormalize=true)

Apply a sequence of local gates repeatedly for `steps` sweeps.

Each element of `gates` must be a tuple `(G, i, j)` consisting of the local
operator `G` and the support `i:j` inside the unit cell.

With `chi_policy = :fixed`, each update is applied with the supplied `maxdim`.
With `chi_policy = :adaptive`, each update is first probed up to `maxdim`, then
the state is compressed back to a non-decreasing bond dimension chosen from the
updated Schmidt spectra.
"""
function evolve!(
    ψ::iMPS,
    gates,
    steps::Integer;
    chi_policy::Symbol=:fixed,
    maxdim::Integer=MAXDIM,
    mindim::Integer=1,
    q::Real=1.5,
    alpha::Real=0.5,
    cutoff::Real=SVDTOL,
    renormalize::Bool=true
)
    steps >= 0 || throw(ArgumentError("steps must be non-negative"))
    maxdim > 0 || throw(ArgumentError("maxdim must be positive"))
    mindim > 0 || throw(ArgumentError("mindim must be positive"))
    maxdim >= mindim || throw(ArgumentError("maxdim must be at least mindim"))

    χ = min(maxdim, max(mindim, maximum(length.(ψ.λ))))

    for _ in 1:steps
        for gate in gates
            G, i, j = gate
            if chi_policy === :fixed
                applygate!(ψ, G, i, j; maxdim, cutoff, renormalize)
            elseif chi_policy === :adaptive
                applygate!(ψ, G, i, j; maxdim, cutoff, renormalize)
                for k in _gate_indices(ψ, i, j)
                    χ = adaptive_bonddim(χ, ψ.λ[k]; mindim, maxdim, q, alpha, cutoff)
                end
                canonical!(ψ; maxdim=χ, cutoff, renormalize)
            else
                throw(ArgumentError("unknown chi_policy $(repr(chi_policy)); use :fixed or :adaptive"))
            end
        end
    end

    ψ
end

#---------------------------------------------------------------------------------------------------
# Multi-Site Operators
#---------------------------------------------------------------------------------------------------
"""
    convert_operator(mat, d, n)

Convert dⁿ×dⁿ matrix that's compatible to the column-major convention.
"""
function convert_operator(mat::AbstractMatrix, d::Integer, n::Integer)
    tensor = reshape(mat, fill(d, 2n)...)
    perm = [n:-1:1; 2n:-1:n+1]
    tensor = permutedims(tensor, perm)
    reshape(tensor, size(mat))
end
