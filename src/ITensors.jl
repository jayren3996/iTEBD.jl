
function imps2mps(ψ::iMPS, s)
    Γ = [permutedims(a, [1,3,2]) for a in ψ.Γ]
    Γs = vcat(fill(Γ, L ÷ ψ.n)...)
    return pbcmps(s, Γs)
end
