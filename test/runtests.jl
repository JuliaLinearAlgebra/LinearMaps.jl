using Test, Documenter, LinearMaps, Aqua

@testset "code quality" begin
    Aqua.test_all(LinearMaps, piracies = (broken=true,))
end

# doctest(LinearMaps)

include("linearmaps.jl")

include("transpose.jl")

include("functionmap.jl")

include("scaledmap.jl")

include("composition.jl")

include("linearcombination.jl")

include("wrappedmap.jl")

include("uniformscalingmap.jl")

include("numbertypes.jl")

include("blockmap.jl")

include("kronecker.jl")

include("conversion.jl")

include("left.jl")

include("fillmap.jl")

include("nontradaxes.jl")

include("embeddedmap.jl")

include("getindex.jl")

include("inversemap.jl")

include("rrules.jl")

include("khatrirao.jl")

include("trace.jl")
