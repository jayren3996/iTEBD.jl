module Gate
using TensorOperations
using LinearAlgebra
#--- CONSTANT
const BOUND = 50
const TOL = 1e-7
#--- Helper functions
commontype(T...) = promote_type(eltype.(T)...)
function tlr!(T,L,R)
    n1 = length(L)
    n2 = length(R)
    for j=1:n2, i=1:n1
        T[i,:,j] *= L[i]*R[j]
    end
end
function glab!(T,G,L,A,B)
    dL = Diagonal(L)
    @tensor T[:] = dL[-1,1]*A[1,3,2]*B[2,4,-4]*G[-2,-3,3,4]
end
function tsvd(T, bound, tol)
    χ,d = size(T)[1:2]
    mat = reshape(T, χ*d, d*χ)
    U, S, V = svd!(mat)
    len = min(sum(S .> tol), bound)
    s = S[1:len]
    s /= norm(s)
    u = reshape(U[:,1:len], χ,d,:)
    v = reshape(transpose(V[:,1:len]), :,d,χ)
    u,s,v
end
#--- Gate
function applygate(G,A,B,λ2; bound=BOUND, tol=TOL)
    χ, d = size(A)[1:2]
    block = Array{commontype(G,A,B)}(undef, χ,d,d,χ)
    glab!(block,G,λ2,A,B)
    nA, λ1, nB = tsvd(block, bound, tol)
    λ2i = 1 ./ λ2
    tlr!(nA,λ2i,λ1)
    nA, λ1, nB
end

end  # module Gate
