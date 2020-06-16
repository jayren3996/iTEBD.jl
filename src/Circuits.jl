#--- Inplace
function TL!(T,L)
    rT = reshape(T, size(T,1), :)
    lmul!(Diagonal(L),rT)
end
function TLiR!(T,L,R)
    n1 = length(L)
    n2 = length(R)
    for j=1:n2, i=1:n1
        T[i,:,j] *= R[j]/L[i]
    end
end
#--- Multiplication
function TT(T1,T2)
    sh1 = size(T1)
    sh2 = size(T2)
    rT1 = reshape(T1, :,sh1[end])
    rT2 = reshape(T2, sh2[1],:)
    T1T2 = rT1 * rT2
    reshape(T1T2, sh1[1],:,sh2[end])
end
function GT(G,T)
    nT = Array{commontype(G,T)}(undef, size(T))
    @tensor nT[:] = T[-1,1,-3] * G[-2,1]
    nT
end
#--- Tensor split
function tsvd(T, bound=BOUND, tol=SVDTOL)
    χ1,d1,d2,χ2 = size(T)
    mat = reshape(T, χ1*d1, d2*χ2)
    U, S, V = svd!(mat)
    len = bound==0 ? sum(S .> tol) : min(sum(S .> tol), bound)
    s = S[1:len]
    s /= norm(s)
    u = reshape(U[:,1:len], χ1,d1,:)
    v = reshape(transpose(V[:,1:len]), :,d2,χ2)
    u,s, Array(v)
end
function tsplit!(T, λ, bound=BOUND, tol=SVDTOL)
    TL!(T, λ)
    u,s,v = tsvd(T, bound, tol)
    TLiR!(u,λ,s)
    u,s,v
end
#--- Combination & decomposition
function combination(λ, Ts...)
    n = length(Ts)
    T = Ts[1]
    TL!(T,λ)
    for i=2:n
        T = TT(T,Ts[i])
    end
    T
end
function comb(Ts...)
    n = length(Ts)
    T = Ts[1]
    for i=2:n
        T = TT(T,Ts[i])
    end
    T
end
function decomposition(λ, T, d, n; bound=BOUND, tol=SVDTOL)
    χ = length(λ)
    Ts = Array{Array{eltype(T),3}}(undef,n)
    λs = Array{Array{Float64,1}}(undef, n)
    Ti = reshape(T, size(T,1),d,:,χ)
    Ai, λi, Ti = tsvd(Ti, bound, tol)
    TLiR!(Ai,λ,λi)
    Ts[1] = Ai
    λs[1] = λi
    for i=2:n-1
        Ti = reshape(Ti, size(Ti,1),d,:,χ)
        Ai, λi, Ti = tsplit!(Ti, λi, bound, tol)
        Ts[i] = Ai
        λs[i] = λi
    end
    Ts[n] = Ti
    λs[n] = λ
    Ts, λs
end
#--- Gate
function applygate!(G, λ, A...; bound=BOUND, tol=SVDTOL)
    n = length(A)
    d = size(A[1],2)
    B = combination(λ, A...)
    C = GT(G,B)
    decomposition(λ,C,d,n, bound=bound, tol=tol)
end
#--- Deprecated
function glab!(T,G,L,A,B)
    dL = Diagonal(L)
    @tensor T[:] = dL[-1,1]*A[1,3,2]*B[2,4,-4]*G[-2,-3,3,4]
end
function applygate(G,λ2,A,B; bound=BOUND, tol=SVDTOL)
    χ, d = size(A)[1:2]
    block = Array{commontype(G,A,B)}(undef, χ,d,d,χ)
    glab!(block,G,λ2,A,B)
    nA, λ1, nB = tsvd(block, bound, tol)
    TLiR!(nA,λ2,λ1)
    nA, λ1, nB
end
function applygate_new(G,λ2,A,B; bound=BOUND, tol=SVDTOL)
    χ, d = size(A)[1:2]
    block = Array{commontype(G,A,B)}(undef, χ,d,d,χ)
    glab!(block,G,λ2,A,B)
    T,λ = decomposition(λ2,block,d,2, bound=bound, tol=tol)
    T[1], λ[1], T[2]
end
