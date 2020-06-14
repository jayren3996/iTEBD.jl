module Canonical
#--- Import
using LinearAlgebra
using TensorOperations
#--- Factorization
function matsqrt(matrix; tol=1e-5)
    vals, vecs = eigen(Hermitian(matrix))
    if all(vals.<tol) vals*=-1 end
    pos = vals .> tol
    # @assert any(pos) "Invalid eigvals:\n $(vals)"
    sqrtvals = Diagonal(sqrt.(vals[pos]))
    X = vecs[:,pos] * sqrtvals
    Xi = sqrtvals * vecs'[pos,:]
    X, Xi
end

function tensorsplit(tensor; bound=0, tol=1e-7, renormalize::Bool=false)
    χ,d = size(tensor)[1:2]
    tensor2 = reshape(tensor, χ*d, d*χ)
    U, S, V = svd(tensor2)
    len = sum(S .> tol)
    if bound>0 len=min(len,bound) end
    S = S[1:len]
    if renormalize S/=norm(S) end
    U = reshape(U[:,1:len], χ,d,:)
    V = reshape(transpose(V[:,1:len]), :,d,χ)
    U, S, V
end
#--- Tranfer matrix-like object
function transfermat(tensor)
    χ = size(tensor, 1)
    conjt = conj(tensor)
    trm = Array{eltype(tensor)}(undef,χ,χ,χ,χ)
    @tensor trm[i,j,k,l] = tensor[i,1,k] * conjt[j,1,l]
    reshape(trm, χ^2, χ^2)
end

function transfermat(T1,T2)
    χ = size(T1)
    cT1 = conj(T1)
    cT2 = conj(T2)
    trm = Array{promote_type(eltype(T1),eltype(T2))}(undef,χ,χ,χ,χ)
    @tensor trm[:] = T1[-1,2,1] * T2[1,4,-3] * cT1[-2,2,3] * cT2[3,4,-4]
    reshape(trm, χ^2, χ^2)
end

function transfermat(A1,B1,A2,B2)
    χ1 = size(A1,1)
    χ2 = size(A2,1)
    cA2 = conj(A2)
    cB2 = conj(B2)
    CommonType = promote_type(eltype(A1),eltype(A2),eltype(B1),eltype(B2))
    trm = Array{CommonType}(undef,χ1,χ2,χ1,χ2)
    @tensor trm[:] = A1[-1,2,1] * B1[1,4,-3] * cA2[-2,2,3] * cB2[3,4,-4]
    reshape(trm, χ1*χ2, χ1*χ2)
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

function dominentvector(matrix)
    spec, vecs = eigen(matrix)
    pos = argmax(abs.(spec))
    vecs[:,pos]
end
#--- Trim redundancy in dominent eigenvecs
function trim(lvecs, rvecs, lstate::Vector, rstate::Vector)
    rvec = rvecs * (rvecs' * reshape(rstate*rstate', :))
    lvec = lvecs * (lvecs' * reshape(lstate*lstate', :))
    return rvec, lvec
end

function trim(lvecs, rvecs)
    i = Int(sqrt(size(lvecs,1)))
    #lstate = rand(i)
    rstate = ones(i)
    lstate = ones(i)
    trim(lvecs, rvecs, lstate, rstate)
end
#--- Schmidt form
function schmidtform(tensor, eigmax, rvec::Vector, lvec::Vector)
    i1,i2,i3 = size(tensor)
    X, Xi = matsqrt(reshape(rvec,i1,:))
    Y, Yi = transpose.(matsqrt(reshape(lvec,i3,:)))
    U, S, V = svd(Y * X)
    normalization = norm(S)
    S /= normalization
    dS = Diagonal(S)
    lmat = V * Xi
    rmat = Yi * U
    j1 = size(lmat,1)
    j2 = size(rmat,2)
    CommonType = promote_type(eltype(tensor), eltype(lmat), eltype(rmat))
    canonicalT = Array{CommonType}(undef,j1,i2,j2)
    @tensor canonicalT[:] = lmat[-1,3]*tensor[3,-2,2]*rmat[2,1]*dS[1,-3]
    canonicalT *= normalization/sqrt(eigmax)
    canonicalT, S
end

function schmidtform(tensor; check::Bool=true)
    if check
        eigmax, rvecs, lvecs = dominentvecs(transfermat(tensor))
        rvec, lvec = trim(rvecs,lvecs)
    else
        eigmax, rvec, lvec = dominentvecs(transfermat(tensor))
    end
    schmidtform(tensor, eigmax, rvec, lvec)
end
#--- canonical form
function λTλ2TλT(tensor, λ2; bound=0, tol=1e-7, renormalize::Bool=true)
    j,d2 = size(tensor)[1:2]
    d = Int(sqrt(d2))
    T = reshape(tensor, j,d,d,j)
    CommonType = promote_type(eltype(T), eltype(λ2))
    T2 = Array{CommonType}(undef, j,d,d,j)
    dλ2 = Diagonal(λ2)
    @tensor T2[i,j,k,l] = dλ2[i,1] * T[1,j,k,l]
    U, λ1, B = tensorsplit(T2, renormalize = true)
    dλ1 = Diagonal(λ1)
    λ2i = Diagonal(1 ./ λ2)
    A = Array{eltype(U)}(undef, size(U))
    @tensor A[i,j,k] = λ2i[i,1] * U[1,j,2] * dλ1[2,k]
    A, λ1, B
end

function canonical(T; check::Bool=true)
    A, λ = schmidtform(T, check=check)
    (A, A, λ, λ)
end

function canonical(T1, T2; check::Bool=true)
    i,d = size(T1)[1:2]
    CommonType = promote_type(eltype(T1),eltype(T2))
    tensor = Array{CommonType}(undef, i,d,d,i)
    @tensor tensor[i,j,k,l] = T1[i,j,1]*T2[1,k,l]
    tensor = reshape(tensor, i,:,i)
    tensor, λ2 = schmidtform(tensor, check=check)
    A, λ1, B = λTλ2TλT(tensor, λ2)
    (A,B,λ1,λ2)
end
#--- Apply gates
function applygate(G,A,B,λ2; bound=50, tol=1e-7)
    χ, d = size(A)[1:2]
    dλ2 = Diagonal(λ2)
    CommonType = promote_type(eltype(G),eltype(A),eltype(B))
    block = Array{CommonType}(undef, χ,d,d,χ)
    @tensor block[:] = dλ2[-1,1]*A[1,3,2]*B[2,4,-4]*G[-2,-3,3,4]
    U, λ1, Bp = tensorsplit(block, bound=bound,tol=tol,renormalize=true)
    len = length(λ1)
    λ2i = inv(dλ2)
    dλ1 = Diagonal(λ1)
    Ap = Array{eltype(U)}(undef, χ,d,len)
    @tensor Ap[i,j,k] = λ2i[i,1]*U[1,j,2]*dλ1[2,k]
    Ap, λ1, Bp
end
#--- MPS functions
function overlap(A1,B1,A2,B2)
    trm = transfermat(A1,B1,A2,B2)
    vals = eigvals!(trm)
    maximum(abs.(vals))
end

end
