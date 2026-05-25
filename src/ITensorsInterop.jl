
function imps2mps(ψ::iMPS, s; L::Integer=length(s))
    L >= 0 || throw(ArgumentError("L must be nonnegative"))
    length(s) >= L || throw(ArgumentError("site index collection must contain at least L indices"))
    L == 0 && return MPS(ITensor[])

    sites = collect(Iterators.take(s, L))
    links = [Index(size(ψ.Γ[mod1(i, ψ.n)], 3), "Link,l=$i") for i in 1:L]
    # The wrap-around convention reuses `links[L]` as the left bond of site 1.
    # For `L == 1` that would collapse the left and right legs of the only
    # tensor onto the same Index object, producing a self-contracted ITensor.
    # Allocate a distinct boundary Index in that case.
    boundary_left = if L == 1
        Index(size(ψ.Γ[1], 1), "Link,l=0")
    else
        links[L]
    end
    tensors = Vector{ITensor}(undef, L)

    for i in 1:L
        Γ = ψ.Γ[mod1(i, ψ.n)]
        size(Γ, 2) == dim(sites[i]) ||
            throw(ArgumentError("site index dimension does not match iMPS physical dimension at site $i"))
        left = i == 1 ? boundary_left : links[i - 1]
        right = links[i]
        size(Γ, 1) == dim(left) && size(Γ, 3) == dim(right) ||
            throw(ArgumentError("iMPS bond dimensions are incompatible at site $i"))
        tensors[i] = ITensor(Γ, left, sites[i], right)
    end

    return MPS(tensors)
end
