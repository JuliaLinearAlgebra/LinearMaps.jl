if "revise" in ARGS
    using Revise
    Revise.revise()
end
const is_draft = "draft" in ARGS

using Documenter
using Literate
using LinearMaps
using LinearAlgebra
using SparseArrays

Literate.markdown(joinpath(@__DIR__, "literate", "custom.jl"), joinpath(@__DIR__, "src"))
Literate.markdown(joinpath(@__DIR__, "literate", "advanced_applications.jl"), joinpath(@__DIR__, "src"))

makedocs(
    sitename = "LinearMaps.jl",
    format = Documenter.HTML(),
    modules = [LinearMaps],
    pages = Any[
        "Home" => "index.md",
        "Version history" => "history.md",
        "Types and methods" => "types.md",
        "Custom maps" => "custom.md",
        "Advanced applications" => "advanced_applications.md",
        "Related packages" => "related.md"
    ],
    draft = is_draft,
)

if !is_draft
deploydocs(
    repo = "github.com/JuliaLinearAlgebra/LinearMaps.jl.git",
    push_preview=true
)
end
