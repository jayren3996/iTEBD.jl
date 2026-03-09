#---------------------------------------------------------------------------------------------------
# Entropy
#---------------------------------------------------------------------------------------------------
"""
    entanglement_entropy(p; cutoff=1e-10)

Compute the von Neumann entropy `-sum(p .* log.(p))` of a probability vector
or Schmidt spectrum `p`, ignoring entries smaller than `cutoff`.
"""
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
"""
    inner_product(T)
    inner_product(T1, T2)

Return the dominant transfer-matrix overlap.

For `inner_product(T)`, this is the norm of a single tensor network transfer
matrix. For `inner_product(T1, T2)`, it is the overlap per unit cell between
two tensor networks or `iMPS` objects.
"""
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


