# LinearMaps

*A Julia package for defining and working with linear maps.*

Linear maps are also known as linear transformations or linear operators acting on vectors.
The only requirement for a `LinearMap` is that it can act on a vector (by multiplication) efficiently.

| **Documentation**                                                               | **Build Status**                                                                                |
|:-------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| [![stable docs][docs-stable-img]][docs-stable-url] [![dev docs][docs-dev-img]][docs-dev-url] | [![build status][build-img]][build-url] [![coverage][codecov-img]][codecov-url] [![Aqua QA][aqua-img]][aqua-url] [![license][license-img]][license-url] |

## Installation

The package can be installed with the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```julia
pkg> add LinearMaps
```

Or, equivalently, via the `Pkg` API:

```julia
julia> import Pkg; Pkg.add("LinearMaps")
```

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://JuliaLinearAlgebra.github.io/LinearMaps.jl/dev

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://JuliaLinearAlgebra.github.io/LinearMaps.jl/stable

[build-img]: https://github.com/JuliaLinearAlgebra/LinearMaps.jl/workflows/CI/badge.svg
[build-url]: https://github.com/JuliaLinearAlgebra/LinearMaps.jl/actions

[codecov-img]: https://codecov.io/gh/JuliaLinearAlgebra/LinearMaps.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/JuliaLinearAlgebra/LinearMaps.jl

[license-img]: http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat
[license-url]: LICENSE.md

[aqua-img]: https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg
[aqua-url]: https://github.com/JuliaTesting/Aqua.jl
