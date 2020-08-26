using Documenter
using LinearMaps
using LinearAlgebra
using SparseArrays

makedocs(
    sitename = "LinearMaps",
    format = Documenter.HTML(),
    modules = [LinearMaps],
    pages = Any[
        "Home" => "index.md",
        "Version history" => "history.md",
        "Types and methods" => "types.md",
        "Custom maps" => "custom.md",
        "Related packages" => "related.md"
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/Jutho/LinearMaps.jl.git",
    push_preview=true
)
