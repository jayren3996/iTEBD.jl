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
#---------------------------------------------------------------------------------------------------
function inner_product(T1, T2)
    trmat = gtrm(T1, T2)
    val, vec = eigsolve(trmat)
    abs(val[1])
end



