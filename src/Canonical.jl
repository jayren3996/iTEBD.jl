module Canonical
#--- Import
using LinearAlgebra
using TensorOperations
#--- CONSTANT
const BOUND = 50
#--- Helper functions
commontype(T...) = promote_type(eltype.(T)...)
tlr!(T,L,R) = for j=1:length(R),i=1:length(L) T[i,:,j]*=L[i]*R[j] end
function transfermat!(T,trm)
    cT = conj(T)
    @tensor trm[:] = T[-1,1,-3] * cT[-2,1,-4]
end
function transfermat(T)
    χ = size(T, 1)
    trm = Array{eltype(T)}(undef, χ,χ,χ,χ)
    transfermat!(T, trm)
    reshape(trm, χ^2, χ^2)
end
#--- Factorization
function matsqrt(matrix; tol=1e-3)
    vals, vecs = eigen!(Hermitian(matrix))
    if all(vals .< tol)
        vals *= -1
    end
    pos = vals .> tol
    sqrtvals = Diagonal(sqrt.(vals[pos]))
    X = vecs[:,pos] * sqrtvals
    Xi = sqrtvals * vecs'[pos,:]
    X, Xi
end
function tsvd(T; tol=1e-7)
    χ,d = size(T)[1:2]
    rT = reshape(T, χ*d, d*χ)
    U, S, V = svd!(rT)
    len = sum(S .> tol)
    s = S[1:len]
    s /= norm(s)
    u = reshape(U[:,1:len], χ,d,:)
    v = reshape(transpose(V[:,1:len]), :,d,χ)
    u, s, v
end
function lab!(T,L,AB)
    dL = Diagonal(L)
    @tensor T[:] = dL[-1,1]*AB[1,-2,-3,-4]
end
function tl2tlt(T,λ2)
    χ, d2 = size(T)[1:2]
    d = Int(sqrt(d2))
    rT = reshape(T, χ,d,d,χ)
    T2 = Array{commontype(T,λ2)}(undef, χ,d,d,χ)
    lab!(T2,λ2,rT)
    A, λ1, B = tsvd(T2)
    λ2i = 1 ./ λ2
    tlr!(A,λ2i,λ1)
    A, λ1, B
end
#--- Dominent eigen vector
function dominentvec(matrix)
    spec, vecs = eigen(matrix)
    pos = argmax(abs.(spec))
    spec[pos], vecs[:,pos], inv(vecs)[pos,:]
end
function dominentvecs(matrix; tol=1e-3)
    spec, vecs = eigen(matrix)
    absspec = abs.(spec)
    maxspec = maximum(absspec)
    pos = absspec.>maxspec - tol
    maxspec, vecs[:,pos], transpose(inv(vecs)[pos,:])
end
#--- Trim redundancy in dominent eigenvecs
function trim(lvecs, rvecs, lstate::Vector, rstate::Vector)
    rvec = rvecs * (rvecs' * reshape(rstate*rstate', :))
    lvec = lvecs * (lvecs' * reshape(lstate*lstate', :))
    return rvec, lvec
end
function trim(lvecs, rvecs)
    i = Int(sqrt(size(lvecs,1)))
    rstate = ones(i)
    lstate = ones(i)
    trim(lvecs, rvecs, lstate, rstate)
end
#--- Schmidt form
function schmidtform(tensor,eigmax,rvec,lvec)
    i1,i2,i3 = size(tensor)
    X, Xi = matsqrt(reshape(rvec,i1,:))
    Y, Yi = transpose.(matsqrt(reshape(lvec,i3,:)))
    U, S, V = svd!(Y * X)
    normalization = norm(S)
    S /= normalization
    dS = Diagonal(S)
    lmat = V * Xi
    rmat = Yi * U
    j1 = size(lmat,1)
    j2 = size(rmat,2)
    canonicalT = Array{commontype(tensor,lmat,rmat)}(undef,j1,i2,j2)
    @tensor canonicalT[:] = lmat[-1,3]*tensor[3,-2,2]*rmat[2,1]*dS[1,-3]
    canonicalT /= sqrt(eigmax)
    canonicalT, S
end
function schmidtform(tensor; check=true)
    if check
        eigmax, rvecs, lvecs = dominentvecs(transfermat(tensor))
        rvec, lvec = trim(rvecs,lvecs)
    else
        eigmax, rvec, lvec = dominentvecs(transfermat(tensor))
    end
    schmidtform(tensor, eigmax, rvec, lvec)
end
#--- canonical form
function canonical(T; check=true)
    A, λ = schmidtform(T, check=check)
    (A, A, λ, λ)
end
function canonical(T1, T2; check=true)
    i,d = size(T1)[1:2]
    tensor = Array{commontype(T1,T2)}(undef, i,d,d,i)
    @tensor tensor[i,j,k,l] = T1[i,j,1]*T2[1,k,l]
    tensor = reshape(tensor, i,:,i)
    tensor, λ2 = schmidtform(tensor, check=check)
    A, λ1, B = tl2tlt(tensor, λ2)
    (A,B,λ1,λ2)
end

end # module Canonical
