# Related open-source packages

The following open-source packages provide similar or even extended functionality as
`LinearMaps.jl`.

*   [`Spot`: A linear-operator toolbox for Matlab](https://github.com/mpf/spot),
    which seems to have heavily inspired the Julia package
    [`LinearOperators.jl`](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl)
    and the Python package [`PyLops`](https://github.com/equinor/pylops)

*   [`fastmat`: fast linear transforms in Python](https://pypi.org/project/fastmat/)

*   [`FunctionOperators.jl`](https://github.com/hakkelt/FunctionOperators.jl)
    and [`LinearMapsAA.jl`](https://github.com/JeffFessler/LinearMapsAA.jl)
    also support mappings between `Array`s, inspired by the `fatrix` object type in the
    [Matlab version of the Michigan Image Reconstruction Toolbox (MIRT)](https://github.com/JeffFessler/mirt).

As for lazy array manipulation (like addition, composition, Kronecker products and concatenation),
there exist further related packages in the Julia ecosystem:

*   [`LazyArrays.jl`](https://github.com/JuliaArrays/LazyArrays.jl)

*   [`BlockArrays.jl`](https://github.com/JuliaArrays/BlockArrays.jl)

*   [`BlockDiagonals.jl`](https://github.com/invenia/BlockDiagonals.jl)

*   [`Kronecker.jl`](https://github.com/MichielStock/Kronecker.jl)

Since these packages provide types that are subtypes of Julia `Base`'s `AbstractArray` type,
objects of those types can be wrapped by a `LinearMap` and freely mixed with, for instance,
function-based linear maps.