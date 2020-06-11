module iTEBD
using LinearAlgebra
using TensorOperations
export canonical, applygate
const BOUND = 50
const THRESHOLD = 1e-7
#--- infinite MPS
struct iMPS{T1,T2}
    A::Array{T2,3}
    B::Array{T2,3}
    λ1::Diagonal{T2,Vector{T2}}
    λ2::Diagonal{T2,Vector{T2}}
end
iMPS(A,B,λ1,λ2) = iMPS{eltype(A),eltype(λ1)}(A,B,Diagonal(λ1),Diagonal(λ2))
#--- SVD with truncation
function tsvd(tensor; bound=BOUND,threshold=THRESHOLD)
    i1,i2,i3,i4 = size(tensor)
    matrix = reshape(tensor, i1*i2, i3*i4)
    U, S ,V = svd(matrix)
    S = S[S.>threshold]
    truncation = min(length(S),bound)
    U = U[:,1:truncation]
    V = transpose(V[:,1:truncation])
    S = S[1:truncation]
    return reshape(U,i1,i2,:), S, reshape(V,:,i3,i4)
end
#--- Apply Gates
function applygate(Gate,A,B,λ1,λ2; bound=BOUND,threshold=THRESHOLD)
    i1,i2 = size(A)[1:2]
    i3,i4 = size(B)[2:3]
    block = Array{promote_type(A,Gate)}(undef,i1,i2,i3,i4)
    @tensor block[α,β,γ,τ] = λ2[α,1]*A[1,2,3]*λ1[3,4]*B[4,5,6]*λ2[6,τ]*G[β,γ,2,5]
    U,λ1p,V = tsvd(block,bound,threshold)
    len = length(λ1p)
    Ap = Array{ComplexF64}(undef,i1,i2,len)
    Bp = Array{ComplexF64}(undef,len,i3,i4)
    λ2i = inv(λ2)
    @tensor Ap[i,j,k] = λ2i[i,1]*U[1,j,k]
    @tensor Bp[i,j,k] = V[i,j,1]*λ2i[1,k]
    Ap, Bp, Diagonal(λ1p), λ2
end

function applygate(Gate,mps::iMPS; bound=BOUND, threshold=THRESHOLD)
    A,B,λ1,λ2 = mps.A, mps.B, mps.λ1, mps.λ2
    A,B,λ1,λ2 = applygate(G,A,B,λ1,λ2)
    B,A,λ2,λ1 = applygate(G,B,A,λ2,λ1)
    iMPS(A,B,λ1,λ2)
end
#--- canonical form
function dominentvec(matrix)
    spec, vecs = eigen(matrix)
    pos = argmax(abs.(spec))
    spec[pos], vecs[:,pos], inv(vecs)[pos,:]
end

function transmat(tensor, χ)
    conjtensor = conj(tensor)
    transfermatrix = Array{eltype(tensor)}(undef,χ,χ,χ,χ)
    @tensor transfermatrix[i,j,k,l] = tensor[i,1,k] * conjtensor[j,1,l]
    reshape(transfermatrix, χ^2, χ^2)
end

function matsqrt(matrix)
    spectrum, vecmat = eigen(Hermitian(matrix))
    if all(spectrum.<0)
        spectrum *= -1
    end
    sqrtspec = sqrt.(spectrum)
    diagsqrtspec = Diagonal(sqrtspec)
    mat = vecmat*diagsqrtspec
    imat = inv(mat)
    return mat, imat
end

function canonical(tensor)
    i1,i2,i3 = size(tensor)
    eigmax, rvec, lvec = dominentvec(transmat(tensor,i1))
    X, Xi = matsqrt(reshape(rvec,i1,:))
    Y, Yi = transpose.(matsqrt(reshape(lvec,i1,:)))
    U, S, V = svd(Y*X)
    lmat = V * Xi
    rmat = Yi * U
    canonicaltensor = Array{eltype(tensor)}(undef,i1,i2,i3)
    @tensor canonicaltensor[i,k,m] = lmat[i,j] * tensor[j,k,l] * rmat[l,m]
    # renormalization
    normalization = norm(S)
    S /= normalization
    canonicaltensor *= normalization/sqrt(eigmax)
    canonicaltensor, Diagonal(S)
end

function canonical(A,B; bound=BOUND,threshold=THRESHOLD)
    i1,i2,i3 = size(A)
    j1,j2,j3 = size(B)
    tensor = Array{eltype(A)}(undef,i1,i2,j2,j3)
    @tensor tensor[i,j,l,m] = A[i,j,k]*B[k,l,m]
    tensor = reshape(tensor, i1,:,j3)
    tensor2, λ2 = canonical(tensor)
    tensor3 = Array{eltype(A)}(undef, size(tensor2))
    @tensor tensor3[i,k,m] = λ2[i,j] * tensor2[j,k,l] * λ2[l,m]
    tensor3 = reshape(tensor3, i1,i2,i2,i1)
    U,λ1,V = tsvd(tensor3)
    invλ2 = inv(λ2)
    C = Array{eltype(U)}(undef,size(U))
    D = Array{eltype(V)}(undef,size(V))
    @tensor C[i,j,k] = invλ2[i,1] * U[1,j,k]
    @tensor D[i,j,k] = V[i,j,1] * invλ2[1,k]
    iMPS(C,D,Diagonal(λ1),λ2)
end

end # module
