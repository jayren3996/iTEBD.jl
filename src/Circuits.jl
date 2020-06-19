#--- Inplace
function TL!(T::Array,
             L::Vector)
    rT = reshape(T, size(T,1), :)
    lmul!(Diagonal(L),rT)
end
function TLiR!(T::Array,
               L::Vector,
               R::Vector)
    n1 = length(L)
    n2 = length(R)
    for j=1:n2, i=1:n1
        T[i,:,j] *= R[j]/L[i]
    end
end
#--- Multiplication
function TT(T1::Tensor,
            T2::Tensor)
    i1,i2,i3 = size(T1)
    j1,j2,j3 = size(T2)
    rT1 = reshape(T1, :,i3)
    rT2 = reshape(T2, j1,:)
    reshape(rT1*rT2, i1,:,j3)
end
function TTT(Ts::TensorArray)
    T = Ts[1]
    for i=2:length(Ts)
        T = TT(T,Ts[i])
    end
    T
end
function λTTT!(λ::Vector,
               Ts::TensorArray)
    TL!(Ts[1],λ)
    TTT(Ts)
end
function GT(G::Matrix,
            T::Tensor)
    i1,i2,i3 = size(T)
    M = reshape(PermutedDimsArray(T, (2,1,3)), i2,:)
    GM = G*M
    rGM = reshape(GM, i2,i1,i3)
    permutedims(rGM, (2,1,3))
end
#--- Tensor split
function tsvd(T::GTensor,
              bound::Integer=BOUND,
              tol::AbstractFloat=SVDTOL)
    χ1,d1,d2,χ2 = size(T)
    U, S, V = svd!(reshape(T, χ1*d1,:))
    len = bound==0 ? sum(S .> tol) : min(sum(S .> tol), bound)
    s = S[1:len]
    s /= norm(s)
    u = reshape(U[:,1:len], χ1,d1,:)
    v = reshape(transpose(V[:,1:len]), :,d2,χ2)
    u,s, Array(v)
end
function tsplit!(T::GTensor,
                 λ::Vector,
                 bound::Integer=BOUND,
                 tol::AbstractFloat=SVDTOL)
    TL!(T, λ)
    u,s,v = tsvd(T, bound, tol)
    TLiR!(u,λ,s)
    u,s,v
end
function decomposition(λ::Vector,
                       T::Tensor{T1},
                       d::Integer,
                       n::Integer,
                       bound::Integer=BOUND,
                       tol::AbstractFloat=SVDTOL) where T1<:Number
    χ = length(λ)
    Ts = Array{Tensor{T1}}(undef,n)
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
function applygate!(G::AbstractMatrix,
                    λ::Vector,
                    A::TensorArray,
                    bound::Integer=BOUND,
                    tol::AbstractFloat=SVDTOL)
    n = length(A)
    d = size(A[1],2)
    B = λTTT!(λ, A)
    C = GT(G,B)
    decomposition(λ,C,d,n, bound, tol)
end
