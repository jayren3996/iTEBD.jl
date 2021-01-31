var documenterSearchIndex = {"docs":
[{"location":"man/Core/#Core-iTEBD-Algorithm","page":"Core iTEBD Algorithm","title":"Core iTEBD Algorithm","text":"","category":"section"},{"location":"man/Core/","page":"Core iTEBD Algorithm","title":"Core iTEBD Algorithm","text":"Main iTEBD algorithm.","category":"page"},{"location":"man/TensorAlgebra/#TensorAlgebra-Tools","page":"TensorAlgebra Tools","title":"TensorAlgebra Tools","text":"","category":"section"},{"location":"man/TensorAlgebra/","page":"TensorAlgebra Tools","title":"TensorAlgebra Tools","text":"Low-level tensor manipulation routines.","category":"page"},{"location":"man/iMPS/#iMPS-Type","page":"iMPS Type","title":"iMPS Type","text":"","category":"section"},{"location":"manual/#Manual","page":"Manual","title":"Manual","text":"","category":"section"},{"location":"manual/#iMPS-Type","page":"Manual","title":"iMPS Type","text":"","category":"section"},{"location":"manual/#iTEBD-Algorithm","page":"Manual","title":"iTEBD Algorithm","text":"","category":"section"},{"location":"manual/#Canonical-Form","page":"Manual","title":"Canonical Form","text":"","category":"section"},{"location":"manual/","page":"Manual","title":"Manual","text":"Compute the canonical form of infinite MPS.","category":"page"},{"location":"man/Miscellaneous/#Miscellaneous","page":"Miscellaneous","title":"Miscellaneous","text":"","category":"section"},{"location":"man/Miscellaneous/","page":"Miscellaneous","title":"Miscellaneous","text":"Other useful functions for MPS calculation.","category":"page"},{"location":"lowlevel/#Low-level-Functions","page":"Low-level Functions","title":"Low-level Functions","text":"","category":"section"},{"location":"lowlevel/#TensorAlgebra-Tools","page":"Low-level Functions","title":"TensorAlgebra Tools","text":"","category":"section"},{"location":"lowlevel/","page":"Low-level Functions","title":"Low-level Functions","text":"Low-level tensor manipulation routines.","category":"page"},{"location":"#iTEBD.jl","page":"Home","title":"iTEBD.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Julia package for infinite time-evolution block-decimation (iTEBD) calculation.","category":"page"},{"location":"","page":"Home","title":"Home","text":"This julia package is for iTEBD algorithms, introduced by G. Vidal (PRL.91.147902, PRL.98.070201, PRB.78.155117), to simulate time evolution of 1D infinite size systems. The iTEBD algorithm relies on a Trotter-Suzuki and subsequent approximation of the time-evolution operator. It provides an extremely efficient method to study both the short time evolution and the ground state (using imaginary time evolution) of 1-D gapped system.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The package can be installed in julia REPL:","category":"page"},{"location":"","page":"Home","title":"Home","text":"pkg> add https://github.com/jayren3996/iTEBD.jl","category":"page"},{"location":"#Code-Examples","page":"Home","title":"Code Examples","text":"","category":"section"},{"location":"#iMPS-objects","page":"Home","title":"iMPS objects","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The states in iTEBD algorithm are represented by infinite matrix-product-states (iMPS):","category":"page"},{"location":"","page":"Home","title":"Home","text":"struct iMPS{TΓ<:Number, Tλ<:Number}\n    Γ::Vector{Array{TΓ, 3}}\n    λ::Vector{Vector{Tλ}}\n    n::Integer\nend","category":"page"},{"location":"","page":"Home","title":"Home","text":"Here we use a slightly different representation with that of Vidal's. The tensor Γ[i] already contain the Schmidt spectrum λ[i], which means when brought to canonical form, each Γ[i] is right-canonical, while λ[i] contains the entanglement information.","category":"page"},{"location":"","page":"Home","title":"Home","text":"There is a function rand_iMPS(n::Integer,d::Integer,dim::Integer) that generates a random iMPS with n periodic sites, d local degrees of freedom, and bond dimension dim.","category":"page"},{"location":"#Hamiltonian","page":"Home","title":"Hamiltonian","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"An Hamiltonian is just an  Array{T,2}. There is also a helper function spinmat for constructing spin Hamiltonian. For example, AKLT Hamiltoniancan be constructed by:","category":"page"},{"location":"","page":"Home","title":"Home","text":"hamiltonian = begin\n    SS = spinmat(\"xx\", 3) + spinmat(\"yy\", 3) + spinmat(\"zz\", 3)\n    SS + 1/3 * SS^2 - 2/3 * spinmat(\"11\", 3)\nend","category":"page"},{"location":"#Setup-and-run-iTEBD","page":"Home","title":"Setup and run iTEBD","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The iTEBD algorithm is generated by an iTEBD_Engine object:","category":"page"},{"location":"","page":"Home","title":"Home","text":"struct iTEBD_Engine{T<:AbstractMatrix}\n    gate ::T\n    renormalize::Bool\n    bound::Int64\n    tol  ::Float64\nend","category":"page"},{"location":"","page":"Home","title":"Home","text":"where gate is the local time-evolving operator, renormalize controls whether to renorm the Schmidt spectrum in the simulation (which is necessary in the non-unitary evolution, such as imaginary-time iTEBD), bound controls the truncation of the singular values, and tol controls the minimal value of the singular value below witch the singular value will be discarded.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The iTEBD_Eigine can be constructed by the function itebd:","category":"page"},{"location":"","page":"Home","title":"Home","text":"function itebd(\n    H::AbstractMatrix{<:Number},\n    dt::AbstractFloat;\n    mode::String=\"r\",\n    renormalize::Bool=true,\n    bound::Int64=BOUND,\n    tol::Float64=SVDTOL\n)","category":"page"},{"location":"","page":"Home","title":"Home","text":"We show an explict example to solve the ground state of AKLT model:","category":"page"},{"location":"","page":"Home","title":"Home","text":"using iTEBD\n\n# Create random iMPS\nimps = begin\n    dim_num = 50\n    rand_iMPS(2, 3, dim_num)\nend\n\n# Create AKLT Hamiltonian and iTEBD engine\nhamiltonian = begin\n    SS = spinmat(\"xx\", 3) + spinmat(\"yy\", 3) + spinmat(\"zz\", 3)\n    SS + 1/3 * SS^2 + 2/3 * spinmat(\"11\", 3)\nend\n\nengine = begin\n    time_step = 0.01\n    itebd(hamiltonian, time_step, mode=\"i\")\nend\n\n# Exact AKLT ground state\naklt = begin\n    aklt_tensor = zeros(2,3,2)\n    aklt_tensor[1,1,2] = +sqrt(2/3)\n    aklt_tensor[1,2,1] = -sqrt(1/3)\n    aklt_tensor[2,2,2] = +sqrt(1/3)\n    aklt_tensor[2,3,1] = -sqrt(2/3)\n    aklt_tensor\n    iMPS([aklt_tensor, aklt_tensor])\nend\n\n# Setup TEBD\nfor i=1:2000\n    global imps, aklt, engine\n    imps = engine(imps)\n    if mod(i, 100) == 0\n        println(\"Overlap: \", inner_product(aklt, imps))\n    end\nend","category":"page"},{"location":"","page":"Home","title":"Home","text":"Here we calculate the inner product of intermediate state and the exact AKLT ground state. We see the overlap quickly converges to 1.","category":"page"},{"location":"#Canonical-form","page":"Home","title":"Canonical form","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"In many cases, it is much simpler to work on the canonical form of MPS. Here, the canonical form is the right-canonical form. However, we keep track of the Schmidt values so that it can easily transformed to Schmidt canonical form.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The canonical form is obtained using the function canonical(imps::iMPS). Note that this function only works when the transfer matrix has single fixed points. Otherwise we should first block diagonalized the tensor using the function block_canonical(Γ::AbstractArray{<:Number, 3}). Note that currently the function block_canonical is NOT numerically stable, though it gives correct result in most of time.","category":"page"},{"location":"man/Canonical/#Canonical-Form","page":"Canonical Form","title":"Canonical Form","text":"","category":"section"},{"location":"man/Canonical/","page":"Canonical Form","title":"Canonical Form","text":"Compute the canonical form of infinite MPS.","category":"page"}]
}