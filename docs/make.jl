using Documenter
using Literate
using LinearMaps
using LinearAlgebra
using SparseArrays

Literate.markdown(joinpath(@__DIR__, "src", "custom.jl"), joinpath(@__DIR__, "src/generated"))

makedocs(
    sitename = "LinearMaps.jl",
    format = Documenter.HTML(),
    modules = [LinearMaps],
    pages = Any[
        "Home" => "index.md",
        "Version history" => "history.md",
        "Types and methods" => "types.md",
        "Custom maps" => "generated/custom.md",
        "Related packages" => "related.md"
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/JuliaLinearAlgebra/LinearMaps.jl.git",
    push_preview=true
)
