using Documenter
using LinearMaps

makedocs(
    sitename = "LinearMaps",
    format = Documenter.HTML(),
    modules = [LinearMaps],
    pages = Any[
        "Home" => "index.md",
        "Version history" => "history.md",
        "Methods" => "methods.md",
        "Types" => "types.md",
        "Custom maps" => "custom.md"
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/CoherentStructures/CoherentStructures.jl.git",
    push_preview=true
)
