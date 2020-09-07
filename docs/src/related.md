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

*   [`FillArrays.jl`](https://github.com/JuliaArrays/FillArrays.jl)
Since these packages provide types that are subtypes of Julia `Base`'s `AbstractMatrix` type,
objects of those types can be wrapped by a `LinearMap` and freely mixed with, for instance,
function-based linear maps. The same applies to custom matrix types as provided, for instance,
by packages maintained by the [`JuliaArrays`](https://github.com/JuliaArrays) github organization.
For any `CustomMatrix{T} <: AbstractMatrix{T}` type, you only need to provide a
`mul!(::AbstractVecOrMat, ::CustomMatrix, ::AbstractVector[, ::Number, ::Number])` method for
seamless integration with `LinearMaps.jl`.
