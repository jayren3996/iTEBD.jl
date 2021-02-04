#-----------------------------------------------------------------------------------------------------
# Spin Operators
#-----------------------------------------------------------------------------------------------------
# Dictionary
function spin_dict(D::Integer)
    J = (D-1)/2
    coeff = [sqrt(i*(D-i)) for i = 1:D-1]
    sp = sparse(1:D-1, 2:D, coeff, D, D)
    sm = sp'
    sx = (sp+sm)/2
    sy = (sp-sm)/2
    sz = sparse(1:D, 1:D, J:-1:-J)
    s0 = sparse(1.0I, D, D)
    Dict([('+', sp), ('-', sm), ('x', sx), ('y', sy), ('z', sz), ('1', s0)])
end
# Atomic spin matrix
function spin_atom(
    s::String, 
    dic::Dict
)
    n = length(s)
    ny = sum(si == 'y' for si in s)
    temp = n == 1 ? dic[s[1]] : kron([dic[si] for si in s]...)
    P = mod(ny, 2) == 0 ? (-1)^(ny÷2) : (1im)^ny
    P * temp
end
# General spin matrix
function spin(
    tup::Tuple{<:Number, String}...;
    D::Integer=2
)
    dic = spin_dict(D)
    res = sum(ci * spin_atom(si, dic) for (ci, si) in tup)
    Array(res)
end

#---------------------------------------------------------------------------------------------------
# Entropy
#---------------------------------------------------------------------------------------------------
function entanglement_entropy(
    S::AbstractVector;
    cutoff::AbstractFloat=1e-10
)
    EE = 0.0
    for si in S
        if si > cutoff
            EE -= si * log(si)
        end
    end
    EE
end

#---------------------------------------------------------------------------------------------------
# Inner Product
#---------------------------------------------------------------------------------------------------
export inner_product
function inner_product(T)
    trmat = trm(T)
    val, vec = eigsolve(trmat)
    abs(val[1])
end

function inner_product(T1, T2)
    trmat = gtrm(T1, T2)
    val, vec = eigsolve(trmat)
    abs(val[1])
end

#---------------------------------------------------------------------------------------------------
# Symmetry representation
#---------------------------------------------------------------------------------------------------
function symrep(T, U; tr::Bool=false)
    tT = tr ? conj.(T) : T
    M = otrm(T, U, tT)
    de, dv = dominent_eigen(M)
    χ = round(Int, sqrt(length(dv)))
    reshape(dv, χ, χ)
end

