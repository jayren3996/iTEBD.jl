module iTEBD
using LinearAlgebra
using TensorOperations
export canonical, applygate
const BOUND = 50
const THRESHOLD = 1e-7
#--- infinite MPS
const CanonicalForm{T1,T2} = Tuple{Array{T1,3}, Diagonal{T2,Vector{T2}}}
const iMPS{T1,T2} = Tuple{CanonicalForm{T1,T2}, CanonicalForm{T1,T2}}
imps(A,B,λ1,λ2) = ((A,λ1),(B,λ2))
#--- SVD with truncation
function tsvd(tensor; bound::Int64=BOUND,threshold::Float64=THRESHOLD)
    i1,i2,i3,i4 = size(tensor)
    matrix = reshape(tensor, i1*i2, i3*i4)
    U, S ,V = svd(matrix)
    S = S[S.>threshold]
    truncation = bound == 0 ? length(S) : min(length(S),bound)
    U = U[:,1:truncation]
    V = transpose(V[:,1:truncation])
    S = S[1:truncation]
    return reshape(U,i1,i2,:), S, reshape(V,:,i3,i4)
end
#--- Apply Gates
function applygate(Gate,A,B,λ1,λ2; bound::Int64=BOUND=BOUND,threshold::Float64=THRESHOLD)
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

function applygate(Gate,mps::iMPS; bound::Int64=BOUND=BOUND,threshold::Float64=THRESHOLD)
    (A,λ1),(B,λ2) = mps
    A,B,λ1,λ2 = applygate(G,A,B,λ1,λ2)
    B,A,λ2,λ1 = applygate(G,B,A,λ2,λ1)
    imps(A,B,λ1,λ2)
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
function tensorsplit(tensor, χ, d)
    tensor = reshape(tensor, χ,:,χ)
    tensor2, λ2 = canonical(tensor)
    tensor3 = Array{eltype(A)}(undef, size(tensor2))
    @tensor tensor3[i,k,m] = λ2[i,j] * tensor2[j,k,l] * λ2[l,m]
    tensor3 = reshape(tensor3, χ,d,d,χ)
    U,λ1,V = tsvd(tensor3,bound=0,threshold=0.0)
    invλ2 = inv(λ2)
    C = Array{eltype(U)}(undef,size(U))
    D = Array{eltype(V)}(undef,size(V))
    @tensor C[i,j,k] = invλ2[i,1] * U[1,j,k]
    @tensor D[i,j,k] = V[i,j,1] * invλ2[1,k]
    imps(C,D,Diagonal(λ1),λ2)
end

function canonical(tensor::Tensor)
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

function canonical(A,B)
    χ,d = size(A)[1:2]
    tensor = Array{eltype(A)}(undef, χ,d,d,χ)
    @tensor tensor[i,j,l,m] = A[i,j,k]*B[k,l,m]
    tensorsplit(tensor, χ,d)
end

function canonical(mps::iMPS)
    (A,λ1),(B,λ2) = mps
    χ,d = size(A)[1:2]
    tensor = Array{eltype(A)}(undef, χ,d,d,χ)
    @tensor tensor[i,j,k,l] = A[i,j,1]*λ1[1,2]*B[2,k,3]*λ2[3,l]
    tensorsplit(tensor, χ,d)
end
#--- TEBD
mutable struct TEBD{T1,T2,T3}
    mps::iMPS{T1,T2}
    gate::Array{T3,4}
    dt::Float64
    T::Float64
end

function run!(tebd::TEBD)
    tebd.mps = applygate(tebd.gate,tebd.mps)
    tebd.T += tebd.dt
end

end # module
