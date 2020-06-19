export canonical
#--- Factorization
function matsqrt!(matrix::AbstractMatrix;
                  tol::AbstractFloat=SQRTTOL)
    vals, vecs = eigen!(Hermitian(matrix))
    if all(vals .< tol)
        vals *= -1
    end
    pos = vals .> tol
    sqrtvals = Diagonal(sqrt.(vals[pos]))
    vecs[:,pos] * sqrtvals
end
#--- Schmidt form
function canonical(tensor::Tensor{T};
                   check::Bool=true,
                   tol::AbstractFloat=SORTTOL) where T
    eigmax, rvec, lvec = dominent_eigvecs!(trm(tensor), check=check, tol=tol)
    i1,i2,i3 = size(tensor)
    X = matsqrt!(reshape(rvec,i1,:))
    Y = transpose(matsqrt!(reshape(lvec,i3,:)))
    U, S, V = svd(Y * X)
    dS = Diagonal(S)
    S /= norm(S)
    lmat = V * X'
    rmat = Y' * U
    j1 = size(lmat,1)
    j2 = size(rmat,2)
    canonicalT = Array{T}(undef,j1,i2,j2)
    for i=1:i2
        canonicalT[:,i,:] .= lmat * tensor[:,i,:] * rmat * dS
    end
    canonicalT /= sqrt(eigmax)
    canonicalT, S
end
#--- canonical form
function canonical(Ts::TensorArray;
                   check::Bool=true,
                   tol::AbstractFloat=SORTTOL)
    n = length(Ts)
    d = size(Ts[1], 2)
    T = TTT(Ts)
    A, λ = canonical(T, check=check, tol=tol)
    TL!(A, λ)
    decomposition(λ, A, d, n)
end
