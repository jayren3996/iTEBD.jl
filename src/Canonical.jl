#--- Factorization
function matsqrt!(matrix; tol=SQRTTOL)
    vals, vecs = eigen!(Hermitian(matrix))
    if all(vals .< tol)
        vals *= -1
    end
    pos = vals .> tol
    sqrtvals = Diagonal(sqrt.(vals[pos]))
    vecs[:,pos] * sqrtvals
end
#--- Schmidt form
function schmidtform(tensor; check=true, tol=SORTTOL)
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
    canonicalT = Array{commontype(tensor,lmat,rmat)}(undef,j1,i2,j2)
    @tensor canonicalT[:] = lmat[-1,3]*tensor[3,-2,2]*rmat[2,1]*dS[1,-3]
    canonicalT /= sqrt(eigmax)
    canonicalT, S
end
#--- canonical form
function canonical(Ts...; check=true)
    n = length(Ts)
    d = size(Ts[1], 2)
    if n == 1
        schmidtform(Ts[1], check=check)
    end
    T = comb(Ts...)
    A, λ = schmidtform(T, check=check)
    TL!(A, λ)
    decomposition(λ, A, d, n)
end
