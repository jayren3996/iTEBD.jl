module Canonical
#--- Import
using LinearAlgebra
using TensorOperations
#--- Helper function
function matsqrt(matrix; tol=1e-5)
    vals, vecs = eigen(Hermitian(matrix))
    if all(vals .< tol)
        vals *= -1
    end
    pos = vals .> tol
    @assert any(pos) "Invalid eigvals:\n $(vals)"
    sqrtvals = Diagonal(sqrt.(vals[pos]))
    X = vecs[:,pos] * sqrtvals
    Xi = sqrtvals * vecs'[pos,:]
    X, Xi
end

function transfermat(tensor)
    χ = size(tensor, 1)
    conjt = conj(tensor)
    trm = Array{eltype(tensor)}(undef,χ,χ,χ,χ)
    @tensor trm[i,j,k,l] = tensor[i,1,k] * conjt[j,1,l]
    reshape(trm, χ^2, χ^2)
end

function dominentvec(matrix)
    spec, vecs = eigen(matrix)
    pos = argmax(abs.(spec))
    spec[pos], vecs[:,pos], inv(vecs)[pos,:]
end

function dominentvecs(matrix; tol=1e-3)
    spec, vecs = eigen(matrix)
    absspec = abs.(spec)
    maxspec = maximum(absspec)
    pos = absspec .> maxspec - tol
    maxspec, vecs[:,pos], transpose(inv(vecs)[pos,:])
end

function tensorsplit(tensor; bound=0, tol=1e-7, renormalize::Bool=false)
    χ,d = size(tensor)[1:2]
    tensor2 = reshape(tensor, χ*d, d*χ)
    U, S, V = svd(tensor2)
    len = sum(S .> tol)
    if bound>0 len=min(len, bound) end
    S = S[1:len]
    if renormalize
        S /= norm(S)
    end
    U = reshape(U[:,1:len], χ,d,:)
    S = Diagonal(S)
    V = reshape(transpose(V[:,1:len]), :,d,χ)
    U,S,V
end
#--- Trim redundancy in dominent eigenvecs
function trim(lvecs, rvecs, lstate::Vector, rstate::Vector)
    rvec = rvecs * (rvecs' * reshape(rstate*rstate', :))
    lvec = lvecs * (lvecs' * reshape(lstate*lstate', :))
    return rvec, lvec
end

function trim(lvecs, rvecs)
    i = Int(sqrt(size(lvecs,1)))
    lstate = rand(i)
    rstate = ones(i)
    trim(lvecs, rvecs, lstate, rstate)
end
#--- Schmidt form
function schmidtform(tensor, eigmax, rvec::Vector, lvec::Vector)
    i1,i2,i3 = size(tensor)
    X, Xi = matsqrt(reshape(rvec,i1,:))
    Y, Yi = transpose.(matsqrt(reshape(lvec,i3,:)))
    U, S, V = svd(Y * X)
    lmat = V * Xi
    rmat = Yi * U
    CommonType = promote_type(eltype(tensor), eltype(lmat), eltype(rmat))
    j1 = size(lmat,1)
    j2 = size(rmat,2)
    canonicaltensor = Array{CommonType}(undef,j1,i2,j2)
    @tensor canonicaltensor[i,k,m] = lmat[i,j] * tensor[j,k,l] * rmat[l,m]
    # renormalization
    normalization = norm(S)
    S /= normalization
    canonicaltensor *= normalization/sqrt(eigmax)
    canonicaltensor, Diagonal(S)
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
    @tensor T2[i,j,k,l] = λ2[i,1] * T[1,j,k,2] * λ2[2,l]
    U, λ1, V = tensorsplit(T2, renormalize = true)
    λ2i = inv(λ2)
    A = Array{eltype(U)}(undef, size(U))
    B = Array{eltype(V)}(undef, size(V))
    @tensor A[i,j,k] = λ2i[i,1] * U[1,j,k]
    @tensor B[i,j,k] = V[i,j,1] * λ2i[1,k]
    A, λ1, B
end

function canonical(T; check::Bool=true)
    A, λ = schmidtform(T, check=check)
    (A, A, λ, λ)
end

function canonical(T1, T2; check::Bool=true)
    i,d = size(T1)[1:2]
    CommonType = promote_type(eltype(T1), eltype(T2))
    tensor = Array{CommonType}(undef, i,d,d,i)
    @tensor tensor[i,j,k,l] = T1[i,j,1]*T2[1,k,l]
    tensor = reshape(tensor, i,:,i)
    tensor, λ2 = schmidtform(tensor, check=check)
    A, λ1, B = λTλ2TλT(tensor, λ2)
    (A,B,λ1,λ2)
end

function canonical(T1,T2,l1,l2; check::Bool=true)
    i,d = size(T1)[1:2]
    CommonType = promote_type(eltype(T1), eltype(T2), eltype(l1), eltype(l2))
    tensor = Array{CommonType}(undef, i,d,d,i)
    @tensor tensor[i,j,k,l] = T1[i,j,2]*l1[2,3]*T2[3,k,4]*l2[4,l]
    tensor = reshape(tensor, i,:,i)
    tensor, λ2 = schmidtform(tensor, check=check)
    A, λ1, B = λTλ2TλT(tensor, λ2)
    (A,B,λ1,λ2)
end
#--- test
"""
aklt = zeros(4,3,4)
aklt[1,1,2] = sqrt(2/3)
aklt[1,2,1] = -sqrt(1/3)
aklt[2,2,2] = sqrt(1/3)
aklt[2,3,1] = -sqrt(2/3)

aklt[3,1,4] = sqrt(2/3)
aklt[3,2,3] = -sqrt(1/3)
aklt[4,2,4] = sqrt(1/3)
aklt[4,3,3] = -sqrt(2/3)
res = canonical(aklt,aklt)
println(res[3],res[4])
res2 = canonical(aklt,aklt,Diagonal(ones(4)),Diagonal(ones(4)))
println(res2[3],res2[4])
"""

end
