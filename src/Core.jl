export tebd
function pindex(L::Vector, i::Integer)
    j = mod(i-1, length(L)) + 1
    vcat(L[j:end],L[1:j-1])
end
#--- TEBD type
struct TEBD{T<:AbstractMatrix}
    gate ::T
    bound::Int64
    tol  ::Float64
end
function tebd(H::AbstractMatrix,
              dt::AbstractFloat;
              mode::String="r",
              bound::Int64=BOUND,
              tol::Float64=SVDTOL)
    if mode == "r" || mode == "real"
        expH = exp(-1im * dt * H)
    elseif mode == "i" || mode == "imag"
        expH = exp(-dt * H)
    end
    TEBD(expH, bound, tol)
end
#--- Run TEBD
function run_tebd!(T::TensorArray,
                   V::ValuesArray,
                   gate::AbstractMatrix,
                   site::Int64,
                   bound::Int64,
                   tol::Float64)
    T,V = applygate!(gate, V[end], T, bound, tol)
    for i=2:site
        T,V = applygate!(gate, V[1], pindex(T,2), bound, tol)
    end
    pindex(T,2), pindex(V,2)
end
function (tebd_system::TEBD)(T::TensorArray,
                             V::ValuesArray,
                             n::Int64=1)
    gate = tebd_system.gate
    bound = tebd_system.bound
    tol = tebd_system.tol
    site = length(T)
    for i=1:n
        T,V = run_tebd!(T, V, gate, site, bound, tol)
    end
    T, V
end
