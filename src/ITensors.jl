
function imps2mps(ψ::iMPS, s; L::Integer=length(s))
    L >= 0 || throw(ArgumentError("L must be nonnegative"))
    length(s) >= L || throw(ArgumentError("site index collection must contain at least L indices"))

    sites = collect(Iterators.take(s, L))
    links = [Index(size(ψ.Γ[mod1(i, ψ.n)], 3), "Link,l=$i") for i in 1:L]
    tensors = Vector{ITensor}(undef, L)

    for i in 1:L
        Γ = ψ.Γ[mod1(i, ψ.n)]
        size(Γ, 2) == dim(sites[i]) ||
            throw(ArgumentError("site index dimension does not match iMPS physical dimension at site $i"))
        left = links[mod1(i - 1, L)]
        right = links[i]
        size(Γ, 1) == dim(left) && size(Γ, 3) == dim(right) ||
            throw(ArgumentError("iMPS bond dimensions are incompatible at site $i"))
        tensors[i] = ITensor(Γ, left, sites[i], right)
    end

    return MPS(tensors)
end
