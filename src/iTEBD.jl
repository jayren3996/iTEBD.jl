module iTEBD
using LinearAlgebra
using TensorOperations
export canonical, applygate, tebd, run
const BOUND = 50
const THRESHOLD = 1e-7
#--- Basic types
const Tensor{T} = Array{T,3}
const Gate{T} = Array{T,4}
const SchmidtVals{T} = Diagonal{T,Vector{T}
struct CMPS{T1,T2}
    tensor::Tensor{T1}
    schmidtvals::SchmidtVals{T2}
end
struct IMPS{T1,T2}
    A::CMPS{T1,T2}
    B::CMPS{T1,T2}
end
SchmidtVals(V::Vector) = Diagonal(V)
CMPS(T::Tensor, λ::Vector) = CMPS(T, SchmidtVals(λ))
IMPS(T1,T2,λ1,λ2) = IMPS(CMPS(T1,λ1),CMPS(T2,λ2))
function getdata(mps::IMPS)
    T1, T2 = mps.A, MPS.B
    A, λ1 = T1.tensor, T1.schmidtvals
    B, λ2 = T2.tensor, T2.schmidtvals
    return A,B,λ1,λ2
end
#--- tensor SVD with truncation
function svd(
    tensor::Array{T,4} where T;
    bound::Int64=BOUND,
    threshold::Float64=THRESHOLD)

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
function applygate(
    G::Gate,
    A::Tensor,
    B::Tensor,
    λ1::SchmidtVals,
    λ2::SchmidtVals;
    bound::Int64=BOUND,
    threshold::Float64=THRESHOLD)

    i1,i2 = size(A)[1:2]
    i3,i4 = size(B)[2:3]
    block = Array{promote_type(eltype(A),eltype(G))}(undef,i1,i2,i3,i4)
    @tensor block[α,β,γ,τ] = λ2[α,1]*A[1,2,3]*λ1[3,4]*B[4,5,6]*λ2[6,τ]*G[β,γ,2,5]
    U,λ1p,V = svd(block, bound, threshold)
    len = length(λ1p)
    Ap = Array{promote_type(eltype(U),eltype(λ2i))}(undef,i1,i2,len)
    Bp = Array{promote_type(eltype(V),eltype(λ2i))}(undef,len,i3,i4)
    λ2i = inv(λ2)
    @tensor Ap[i,j,k] = λ2i[i,1]*U[1,j,k]
    @tensor Bp[i,j,k] = V[i,j,1]*λ2i[1,k]
    Ap, Bp, Diagonal(λ1p), λ2
end

function applygate(
    G::Gate,
    mps::iMPS;
    bound::Int64=BOUND,
    threshold::Float64=THRESHOLD)

    A,B,λ1,λ2 = getdata(mps)
    A,B,λ1,λ2 = applygate(G,A,B,λ1,λ2)
    B,A,λ2,λ1 = applygate(G,B,A,λ2,λ1)
    IMPS(A,B,λ1,λ2)
end
#--- canonical form
function dominentvec(matrix)
    spec, vecs = eigen(matrix)
    pos = argmax(abs.(spec))
    spec[pos], vecs[:,pos], inv(vecs)[pos,:]
end
function transmat(tensor, χ::Int64)
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
function tensorsplit(tensor, χ::Int64, d::Int64)
    tensor = reshape(tensor, χ,:,χ)
    tensor2, λ2 = canonical(tensor)
    tensor3 = Array{eltype(A)}(undef, size(tensor2))
    @tensor tensor3[i,k,m] = λ2[i,j] * tensor2[j,k,l] * λ2[l,m]
    tensor3 = reshape(tensor3, χ,d,d,χ)
    U,λ1,V = svd(tensor3,bound=0,threshold=0.0)
    invλ2 = inv(λ2)
    C = Array{eltype(U)}(undef,size(U))
    D = Array{eltype(V)}(undef,size(V))
    @tensor C[i,j,k] = invλ2[i,1] * U[1,j,k]
    @tensor D[i,j,k] = V[i,j,1] * invλ2[1,k]
    IMPS(C,D,Diagonal(λ1),λ2)
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
    CMPS(canonicaltensor, S)
end

function canonical(A::Tensor, B::Tensor)
    χ,d = size(A)[1:2]
    tensor = Array{eltype(A)}(undef, χ,d,d,χ)
    @tensor tensor[i,j,l,m] = A[i,j,k]*B[k,l,m]
    tensorsplit(tensor, χ, d)
end

function canonical(mps::iMPS)
    A,B,λ1,λ2 = getdata(mps)
    χ,d = size(A)[1:2]
    tensor = Array{eltype(A)}(undef, χ,d,d,χ)
    @tensor tensor[i,j,k,l] = A[i,j,1]*λ1[1,2]*B[2,k,3]*λ2[3,l]
    tensorsplit(tensor, χ, d)
end
#--- TEBD
mutable struct TEBD{T1,T2,T3}
    mps::iMPS{T1,T2}
    gate::Array{T3,4}
    dt::Float64
    T::Float64
end

function tebd(mps::iMPS,H::Matrix,dt)
    expH = exp(-dt*im * H)
    d = Int(sqrt(size(H,1)))
    gate = reshape(expH,d,d,d,d)
    TEBD(mps,gate,dt,0.0)
end

function run!(tebd::TEBD)
    tebd.mps = applygate(tebd.gate,tebd.mps)
    tebd.T += tebd.dt
end

end # module
