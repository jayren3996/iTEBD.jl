#---------------------------------------------------------------------------------------------------
# iTEBD object
#---------------------------------------------------------------------------------------------------
struct iTEBD_Engine{T<:AbstractMatrix}
    gate ::T
    renormalize::Bool
    bound::Int64
    tol  ::Float64
end
#---------------------------------------------------------------------------------------------------
export itebd
function itebd(
    H::AbstractMatrix{<:Number},
    dt::Real;
    mode::Symbol=:r,
    renormalize::Bool=true,
    bound::Int64=BOUND,
    tol::Float64=SVDTOL
)
    gate = if mode == :r
        exp(-1im * dt * H)
    elseif mode == :i
        exp(-dt * H)
    elseif mode == :g
        H
    else
        error("Invalid mode: $mode.")
    end
    iTEBD_Engine(gate, renormalize, bound, tol)
end
#---------------------------------------------------------------------------------------------------
# Run iTEBD
#---------------------------------------------------------------------------------------------------
function (engin::iTEBD_Engine)(mps::iMPS)
    gate = engin.gate
    renormalize = engin.renormalize
    bound = engin.bound
    tol = engin.tol

    qdim = size(mps.Γ[1], 2)
    gdim = size(gate, 1)
    nsite = round(Int64, log(qdim, gdim))
    gtype = eltype(engin.gate)
    mps_in = if gtype <: Complex
        mps_promote_type(gtype, mps)
    else
        deepcopy(mps)
    end
    for i = 1:mps.n
        inds = i:i+nsite-1
        applygate!(gate, mps_in, inds, renormalize=renormalize, bound=bound, tol=tol)
    end
    mps_in
end
