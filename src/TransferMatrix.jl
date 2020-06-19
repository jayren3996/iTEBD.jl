export trm, inner, symrep
#--- Generalized transfer matrix
function gtrm(T1::Tensor,
              T2::Tensor)
    i1,j1,k1 = size(T1)
    i2,j2,k2 = size(T2)
    T1 = conj(T1)
    M1 = reshape(PermutedDimsArray(T1,(1,3,2)), i1*k1,:)
    M2 = reshape(PermutedDimsArray(T2,(2,1,3)), :,i2*k2)
    M = M1 * M2
    tM = reshape(M, i1,k1,i2,k2)
    reshape(permutedims(tM,(1,3,2,4)), i1*i2,:)
end
function gtrm(T1s::TensorArray,
              T2s::TensorArray)
    n = length(T1s)
    M = gtrm(T1s[1], T2s[1])
    for i=2:n
        M = M * gtrm(T1s[i], T2s[i])
    end
    M
end
trm(T::Union{Tensor, TensorArray}) = gtrm(T,T)
#--- Operator transfermatrix
function otrm(T1::Tensor,
              T2::Tensor,
              O::AbstractMatrix)
    i1,j1,k1 = size(T1)
    i2,j2,k2 = size(T2)
    T1 = conj(T1)
    M1 = reshape(PermutedDimsArray(T1,(1,3,2)), i1*k1,:)
    M2 = reshape(PermutedDimsArray(T2,(2,1,3)), :,i2*k2)
    M = M1 * O * M2
    tM = reshape(M, i1,k1,i2,k2)
    reshape(permutedims(tM,(1,3,2,4)), i1*i2,:)
end
function otrm(T1s::TensorArray,
              T2s::TensorArray,
              O::AbstractMatrix)
    n = length(T1s)
    M = otrm(T1s[1], T2s[1], O)
    for i=2:n
        M = M * otrm(T1s[i], T2s[i], O)
    end
    M
end
function otrm(T1s::TensorArray,
              T2s::TensorArray,
              O::Vector{T}) where T<:AbstractMatrix
    n = length(T1s)
    M = otrm(T1s[1], T2s[1], O[1])
    for i=2:n
        M = M * otrm(T1s[i], T2s[i], O[i])
    end
    M
end
#--- Dominent eigensystem
function dominent_eigen(matrix::AbstractMatrix)
    spec, vecs = eigen(matrix)
    vals = abs.(spec)
    pos = argmax(vals)
    vals[pos], vecs[:,pos]
end
function dominent_eigen!(matrix::AbstractMatrix)
    spec, vecs = eigen!(matrix)
    vals = abs.(spec)
    pos = argmax(vals)
    vals[pos], vecs[:,pos]
end
dominent_eigval(matrix::AbstractMatrix) = maximum(abs.(eigvals(matrix)))
dominent_eigval!(matrix::AbstractMatrix) = maximum(abs.(eigvals!(matrix)))
function lrvec(lvecs::AbstractMatrix, rvecs::AbstractMatrix)
    n = Int(sqrt(size(lvecs,1)))
    rstate = ones(n)
    lstate = ones(n)
    rvec = rvecs * (rvecs' * reshape(rstate*rstate', :))
    lvec = lvecs * (lvecs' * reshape(lstate*lstate', :))
    return rvec, lvec
end
function dominent_eigvecs!(matrix::AbstractMatrix;
                           check::Bool=true,
                           tol::AbstractFloat=SORTTOL)
    spec, vecs = eigen!(matrix)
    absspec = abs.(spec)
    if check
        maxspec = maximum(absspec)
        pos = absspec.>maxspec - tol
        if sum(pos) > 1
            rvec, lvec = lrvec(vecs[:,pos], transpose(inv(vecs)[pos,:]))
            return maxspec, rvec, lvec
        end
    end
    pos = argmax(absspec)
    absspec[pos], vecs[:,pos], inv(vecs)[pos,:]
end
#--- Overlap
inner(T::Union{Tensor, TensorArray}) = dominent_eigval!(trm(T))
inner(T1::Tensor, T2::Tensor) = dominent_eigval!(gtrm(T1,T2))
inner(T1::TensorArray, T2::TensorArray) = dominent_eigval!(gtrm(T1,T2))
#--- Energy
function energy(H::AbstractMatrix,
                Ts::TensorArray)
    T = TTT(Ts)
    TM = trm(T)
    em, rv, lv = dominent_eigvecs!(TM)
    HTM = otrm(T,H,T)
    E = (transpose(lv) * HTM * rv) / em
    real(E)
end
#--- Symmetry representation
function symrep(T::Tensor,
                U::AbstractMatrix;
                TR::Bool=false)
    tT = TR ? conj(T) : T
    M = otrm(T,U,tT)
    de, dv = dominent_eigen!(M)
    reshape(dv,χ,χ)
end
function symrep(Ts::TensorArray,
                U::AbstractMatrix;
                TR::Bool=false)
    tT = TR ? conj.(Ts) : Ts
    M = otrm(T,U,tT)
    de, dv = dominent_eigen!(M)
    reshape(dv,χ,χ)
end
function symrep(Ts::TensorArray,
                U::Vector{T};
                TR::Bool=false) where T<:AbstractMatrix
    tT = TR ? conj.(Ts) : Ts
    M = otrm(T,U,tT)
    de, dv = dominent_eigen!(M)
    reshape(dv,χ,χ)
end
