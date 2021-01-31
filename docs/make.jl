using Documenter
using iTEBD

makedocs(
    modules = [iTEBD],
    sitename = "iTEBD.jl",
    authors = "Jie Ren",
    pages = [
        "Home" => "index.md",
        "Manual" => "manual.md",
        "Low-level Functions" => "lowlevel.md"
    ],
)

deploydocs(repo = "github.com/jayren3996/iTEBD.jl.git")