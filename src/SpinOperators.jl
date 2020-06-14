module SpinOperators
using LinearAlgebra
export spinop

function Sp(J)
    D = Int(2*J)+1
    M = zeros(D,D)
    for i = 1:D-1
        ele = sqrt(i*(D-i))
        M[i,i+1] += ele
    end
    M
end
function Sm(J)
    D = Int(2*J)+1
    M = zeros(D,D)
    for i = 1:D-1
        ele = sqrt(i*(D-i))
        M[i+1,i] += ele
    end
    M
end
function Sx(J)
    D = Int(2*J)+1
    M = zeros(D,D)
    for i = 1:D-1
        ele = sqrt(i*(D-i)) / 2
        M[i,i+1] += ele
        M[i+1,i] += ele
    end
    M
end
function iSy(J)
    D = Int(2*J)+1
    M = zeros(D,D)
    for i = 1:D-1
        ele = sqrt(i*(D-i)) / 2
        M[i,i+1] += ele
        M[i+1,i] -= ele
    end
    M
end
Sy(J) = -1im * iSy(J)
Sz(J) = Diagonal(J:-1:-J)
S0(J) = Diagonal(ones(Int(2*J)+1))

function spinmap(s,J)
    if s=='1' || s=='i' || s=='0'
        return S0(J)
    elseif s == 'x'
        return Sx(J)
    elseif s == 'y'
        return Sy(J)
    elseif s == 'z'
        return Sz(J)
    elseif s == '+'
        return Sp(J)
    elseif s == '-'
        return Sm(J)
    end
end

function spinop(op::String, S=1/2)
    if op=="yy"
        operator = -kron(iSy(S),iSy(S))
    elseif length(op) == 1
        operator = spinmap(op[1], S)
    else
        s1 = spinmap(op[1], S)
        s2 = spinmap(op[2], S)
        operator = kron(s1, s2)
    end
    Array(operator)
end

end
