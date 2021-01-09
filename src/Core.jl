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
function cycle(
    mps::iMPS, 
    i::Integer=2
)
    n = mps.n
    j = mod(i-1, n) + 1
    pos_cycled = [j:n..., 1:(j-1)...]
    Γ_cycled = mps.Γ[pos_cycled]
    λ_cycled = mps.λ[pos_cycled]
    iMPS(Γ_cycled, λ_cycled, n)
end
#---------------------------------------------------------------------------------------------------
export itebd
function itebd(
    H::AbstractMatrix{<:Number},
    dt::AbstractFloat;
    mode::String="r",
    renormalize::Bool=true,
    bound::Int64=BOUND,
    tol::Float64=SVDTOL
)
    expH = mode=="i" ? exp(-dt * H) : exp(-1im * dt * H)
    iTEBD_Engine(expH, renormalize, bound, tol)
end
#---------------------------------------------------------------------------------------------------
# Run iTEBD
#---------------------------------------------------------------------------------------------------
function (engin::iTEBD_Engine)(mps::iMPS)
    gate, renormalize, bound, tol = engin.gate, engin.renormalize, engin.bound, engin.tol
    mps = applygate!(gate, mps, renormalize=renormalize, bound=bound, tol=tol)
    for i=2:mps.n
        mps = applygate!(gate, cycle(mps, 2), renormalize=renormalize, bound=bound, tol=tol)
    end
    cycle(mps, 2)
end
