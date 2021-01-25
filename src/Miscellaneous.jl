#-----------------------------------------------------------------------------------------------------
# Spin Operators
#-----------------------------------------------------------------------------------------------------
coefficient(D::Integer) = [sqrt(i*(D-i)) for i = 1:D-1]
Sp(D::Integer) = sparse(1:D-1, 2:D,coefficient(D), D,D)
Sm(D::Integer) = Sp(D)'
Sx(D::Integer) = (sp=Sp(D); (sp+sp')/2)
iSy(D::Integer) = (sp=Sp(D); (sp-sp')/2)
Sy(D::Integer) = (sp=Sp(D); 0.5im*(sp'-sp))
Sz(D::Integer) = (J=(D-1)/2; sparse(1:D,1:D, J:-1:-J))
# Dictionary
function spin(
    s::Char, 
    D::Integer
)
    if s == '+' return Sp(D)
    elseif s == '-' return Sm(D)
    elseif s == 'x' return Sx(D)
    elseif s == 'Y' return iSy(D)
    elseif s == 'y' return Sy(D)
    elseif s == 'z' return Sz(D)
    elseif s == '1' return sparse(I,D,D)
    end
end
#-----------------------------------------------------------------------------------------------------
# General spin matrix
function spinmat(
    s::String, 
    D::Integer
)
    if length(s) == 1
        return spin(s[1],D)
    end
    mat = kron([spin(si,D) for si in s]...)
    Array(mat)
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

