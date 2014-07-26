using LinearMaps
using Base.Test

if VERSION.minor < 3
    include("tests2.jl")
else
    include("tests3.jl")
end
