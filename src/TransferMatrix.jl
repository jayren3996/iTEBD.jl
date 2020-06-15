module TransferMatrix
using LinearAlgebra
using TensorOperations
#--- Helper functions
commontype(T...) = promote_type(eltype.(T)...)
#--- Overwrite functions
function gtrm!(M, A1, A2)
    cA2 = conj(A2)
    @tensor M[:] = A1[-1,1,-3] * cA2[-2,1,-4]
end
function gtrm!(M, A1, B1, A2, B2)
    cA2 = conj(A2)
    cB2 = conj(B2)
    @tensor M[:] = A1[-1,2,1] * B1[1,3,-3] * cA2[-2,2,4] * cB2[4,3,-4]
end
function utrm!(M, A, U, B)
    cB = conj!(B)
    @tensor M[:] = A[-1,1,-3] * U[2,1] * cB[-2,2,-4]
end
#--- Transfer matrix
function trm(A)
    χ = size(A, 1)
    M = Array{eltype(A)}(undef, χ,χ,χ,χ)
    gtrm!(M, A,A)
    reshape(M, χ^2, χ^2)
end
function trm(A,B)
    χ = size(A, 1)
    M = Array{commontype(A,B)}(undef, χ,χ,χ,χ)
    gtrm!(M, A,B,A,B)
    reshape(M, χ^2, χ^2)
end
#--- Generalized transfer matrix
function gtrm(A1, A2)
    χ1 = size(A1, 1)
    χ2 = size(A2, 1)
    M = Array{commontype(A1,A2)}(undef, χ1,χ2,χ1,χ2)
    gtrm!(M, A1,A2)
    reshape(M, χ1*χ2, χ1*χ2)
end
function gtrm(A1, B1, A2, B2)
    χ1 = size(A1, 1)
    χ2 = size(A2, 1)
    M = Array{commontype(A1,A2,B1,B2)}(undef, χ1,χ2,χ1,χ2)
    gtrm!(M, A1,B1,A2,B2)
    reshape(M, χ1*χ2, χ1*χ2)
end
#--- Operator transfermatrix
function utrm(A,U,B)
    χ = size(A, 1)
    M = Array{commontype(A,U,B)}(undef, χ,χ,χ,χ)
    utrm!(M,A,U,B)
    reshape(M, χ^2, χ^2)
end
#--- Dominent eigensystem
function dominent_eigen(matrix)
    spec, vecs = eigen(matrix)
    vals = abs.(spec)
    pos = argmax(vals)
    vals[pos], vecs[:,pos]
end
function dominent_eigen!(matrix)
    spec, vecs = eigen!(matrix)
    vals = abs.(spec)
    pos = argmax(vals)
    vals[pos], vecs[:,pos]
end
dominent_eigval(matrix) = maximum(abs.(eigvals(matrix)))
dominent_eigval!(matrix) = maximum(abs.(eigvals!(matrix)))
#--- Overlap
normalization(A) = dominent_eigval!(trm(A))
normalization(A,B) = dominent_eigval!(trm(A,B))
inner(A1,A2) = dominent_eigval!(gtrm(A1,A2))
inner(A1,B1,A2,B2) = dominent_eigval!(gtrm(A1,B1,A2,B2))
#--- Symmetry representation
function symrep(A,U; TR=false)
    cA = TR ? conj(A) : A
    de, dv = dominent_eigen!(utrm(A,U,cA))
    reshape(dv,χ,χ)
end
function symrep(A,B,U; TR=false)
    χ = size(A,1)
    cA = TR ? conj(A) : A
    cB = TR ? conj(B) : B
    de, dv = dominent_eigen!(utrm(A,U,cA) * utrm(B,U,cB))
    reshape(dv,χ,χ)
end
function symrep(A,B,Ua,Ub; TR=false)
    cA = TR ? conj(A) : A
    cB = TR ? conj(B) : B
    de, dv = dominent_eigen!(utrm(A,Ua,cA) * utrm(B,Ub,cB))
    reshape(dv,χ,χ)
end

end # module TransferMatrix
