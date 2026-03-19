using Documenter
using iTEBD
using LinearAlgebra

DocMeta.setdocmeta!(iTEBD, :DocTestSetup, :(using iTEBD, LinearAlgebra); recursive=true)

makedocs(
    sitename="iTEBD.jl",
    modules=[iTEBD],
    authors="jayren3996",
    format=Documenter.HTML(prettyurls=get(ENV, "CI", "false") == "true"),
    pages=[
        "Overview" => "index.md",
        "Guide" => [
            "Getting Started" => "getting-started.md",
            "States and Canonical Form" => "imps.md",
            "Time Evolution" => "time-evolution.md",
            "Observables" => "observables.md",
            "ScarFinder Workflow" => "scarfinder.md",
        ],
        "Reference" => [
            "API Reference" => "api.md",
        ],
    ],
)

deploydocs(
    repo="github.com/jayren3996/iTEBD.jl.git",
    devbranch="master",
)
