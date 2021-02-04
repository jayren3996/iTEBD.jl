#---------------------------------------------------------------------------------------------------
# QUANTUM GATE
#---------------------------------------------------------------------------------------------------
struct GATE{T<:Number}
    mat::Matrix{T}
    inds::Vector{Int64}
    bound::Int64
    cutoff::Float64
    renormalize::Bool
end

eltype(::GATE{T}) where T = T
get_data(g::GATE) = g.mat, g.inds, g.bound, g.cutoff, g.renormalize
#---------------------------------------------------------------------------------------------------
export gate
function gate(
    mat::AbstractMatrix,
    ind::AbstractVector{<:Integer};
    bound::Integer=50,
    cutoff::Real=1e-7,
    renormalize::Bool=true
)
    GATE(
        Array(mat), 
        Int64.(ind), 
        Int64(bound), 
        Float64(cutoff), 
        renormalize
    )
end
#---------------------------------------------------------------------------------------------------
export applygate
function applygate(mps::iMPS, gate::GATE)
    mps_type = eltype(mps)
    gate_type = eltype(gate)
    ctype = promote_type(mps_type, gate_type)
    mps_out = if mps_type != ctype
        mps_promote_type(ctype, mps)
    else
        deepcopy(mps)
    end
    applygate!(mps_out, gate)
end
#---------------------------------------------------------------------------------------------------
function applygate!(mps::iMPS, gate::GATE)
    mat, inds, bound, tol, renorm = get_data(gate)
    applygate!(mps, mat, inds, renormalize=renorm, bound=bound, tol=tol)
end
