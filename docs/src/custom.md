# Defining custom `LinearMap` types

In this section, we want to demonstrate on a simple, actually built-in, linear map type
how to define custom `LinearMap` subtypes. First of all, `LinearMap{T}` is an extendable
abstract type, where `T` denotes the `eltype`.

## Basics

As an example, we want to define a map type whose objects correspond to lazy analogues
of `fill`ed matrices. Naturally, we need to store the filled value `λ` and the `size`
of the linear map.

```julia
using LinearMaps, LinearAlgebra

struct MyFillMap{T} <: LinearMaps.LinearMap{T}
    λ::T
    size::Dims{2}
    function MyFillMap(λ::T, dims::Dims{2}) where {T}
        all(d -> d >= 0, dims) || throw(ArgumentError("dims of MyFillMap must be non-negative"))
        promote_type(T, typeof(λ)) == T || throw(InexactError())
        return new{T}(λ, dims)
    end
end
```

By default, for any `A::MyFillMap{T}`, `eltype(A)` returns `T`. Upon application to a
vector `x` and/or interaction with other `LinearMap` objects, we need to check consistent
sizes.

```julia
Base.size(A::MyFillMap) = A.size
```

By a couple of defaults provided for all subtypes of `LinearMap`, we only need to define
a `LinearAlgebra.mul!` method to have minimal, operational type.

```julia
function LinearAlgebra.mul!(y::AbstractVecOrMat, A::MyFillMap, x::AbstractVector)
    LinearMaps.check_dim_mul(y, A, x)
    return fill!(y, iszero(A.λ) ? zero(eltype(y)) : A.λ*sum(x))
end
```

Again, due to generic fallbacks the following now "just works":
* out-of-place multiplication `A*x`,
* in-place multiplication with vectors `mul!(y, A, x)`,
* in-place multiply-and-add with vectors `mul!(y, A, x, α, β)`,
* in-place multiplication and multiply-and-add with matrices `mul!(Y, A, X, α, β)`,
* conversion to a (sparse) matrix `Matrix(A)` and `sparse(A)`.

```julia
julia> A = MyFillMap(5.0, (3, 3)); x = ones(3); sum(x)
3.0

julia> A * x
3-element Array{Float64,1}:
 15.0
 15.0
 15.0

julia> mul!(zeros(3), A, x)
3-element Array{Float64,1}:
 15.0
 15.0
 15.0

julia> mul!(ones(3), A, x, 2, 2)
3-element Array{Float64,1}:
 32.0
 32.0
 32.0

julia> mul!(ones(3,3), A, reshape(collect(1:9), 3, 3), 2, 2)
3×3 Array{Float64,2}:
 62.0  152.0  242.0
 62.0  152.0  242.0
 62.0  152.0  242.0
```

## Multiply-and-add and the `MulStyle` trait

While the above function calls work out of the box due to generic fallbacks, the latter
may be suboptimally implemented for your custom map type. Let's see some benchmarks.

```julia
using BenchmarkTools

julia> @btime mul!($(zeros(3)), $A, $x);
  8.788 ns (0 allocations: 0 bytes)

julia> @btime mul!($(zeros(3)), $A, $x, $(rand()), $(rand()));
  58.695 ns (1 allocation: 112 bytes)
```

The second benchmark indicates the allocation of an intermediate vector `z``
which stores the result of `A*x` before it gets scaled and added to (the scaled)
`y = zeros(3)`. For that reason, it is beneficial to provide a custom "5-arg `mul!`"
if you can avoid the allocation of an intermediate vector. To indicate that there
exists an allocation-free implementation, you should set the `MulStyle` trait,
whose default is `ThreeArg()`.

```julia
LinearMaps.MulStyle(A::MyFillMap) = FiveArg()

function LinearAlgebra.mul!(y::AbstractVecOrMat, A::MyFillMap, x::AbstractVector, α::Number, β::Number)
    if iszero(α)
        !isone(β) && rmul!(y, β)
        return y
    else
        temp = A.λ * sum(x) * α
        if iszero(β)
            y .= temp
        elseif isone(β)
            y .+= temp
        else
            y .= y .* β .+ temp
        end
    end
    return y
end
```

With this function at hand, let's redo the benchmark.

```julia
julia> @btime mul!($(zeros(3)), $A, $x, $(rand()), $(rand()));
  11.156 ns (0 allocations: 0 bytes)
```

There you go, the allocation is gone and the computation time is significantly reduced.

## Adjoints and transposes

The following functions are obviously helpful additions.

```
LinearAlgebra.issymmetric(A::MyFillMap) = A.size[1] == A.size[2]
LinearAlgebra.ishermitian(A::MyFillMap) = isreal(A) && A.size[1] == A.size[2]
LinearAlgebra.isposdef(A::MyFillMap) = (size(A, 1) == size(A, 2) == 1 && isposdef(A.λ))
Base.:(==)(A::FillMap, B::MyFillMap) = A.λ == B.λ && A.size == B.size
```

These are used, for instance, in checking symmetry or positive definiteness of
higher-order `LinearMap`s, like products or linear combinations of linear maps, or signal
to iterative eigenproblem solvers that real eigenvalues are to be computed.
Without these definitions, the first three functions would return `false` (by default),
and the last one would fall back to `===`.

As for the multiplication of adjoints/transposes of `MyFillMap`s with vectors, there exist
two *distinct* paths.

### Path 1: Generic, non-invariant `LinearMap` subtypes

Generically, taking the transpose (or the adjoint) of a (real, resp.) map wraps the
linear map by a `TransposeMap`.

```julia
julia> typeof(A')
LinearMaps.TransposeMap{Float64,MyFillMap{Float64}}
```

Not surprisingly, without further definitions, multipliying `A'` by `x` yields an error.

```julia
julia> A'x
ERROR: transpose not implemented for MyFillMap{Float64}(5.0, (3, 3))
```

This can be fixed by providing `LinearAlgebra.mul!` methods for the corresponding wrapped
map types; for instance,

```julia
function LinearAlgebra.mul!(y::AbstractVecOrMat, transA::LinearMaps.TransposeMap{<:Any,<:MyFillMap}, x::AbstractVector)
    LinearMaps.check_dim_mul(y, transA, x)
    λ = transA.lmap.λ
    return fill!(y, iszero(λ) ? zero(eltype(y)) : transpose(λ)*sum(x))
end
```

If you have set the `MulStyle` trait to `FiveArg()`, you should provide a corresponding
5-arg `mul!` method for `LinearMaps.TransposeMap{<:Any,<:MyFillMap}` and
`LinearMaps.AdjointMap{<:Any,<:MyFillMap}`.

### Path 2: Invariant `LinearMap` subtypes

The easier option is when your class of linear maps that are modelled by your custom
`LinearMap` subtype are invariant under taking adjoints and transposes.

```julia
LinearAlgebra.adjoint(A::MyFillMap) = MyFillMap(adjoint(A.λ), reverse(A.size))
LinearAlgebra.transpose(A::MyFillMap) = MyFillMap(transpose(A.λ), reverse(A.size))
```

With such invariant definitions, i.e., the adjoint/transpose of a `MyFillMap` is again
a `MyFillMap`, no further method definitions are required, and the entire functionality
listed above just works for adjoints/transposes of your custom map type.

```julia
julia> mul!(ones(3), A', x, 2, 2)
3-element Array{Float64,1}:
 32.0
 32.0
 32.0
```

## Application to matrices

By default, applying a `LinearMap` `A` to a matrix `X` via `A*X` does
*not* aplly `A` to each column of `X` viewed as a vector, but interprets
`X` as a linear map, wraps it as such and returns `(A*X)::CompositeMap`.
Calling the in-place multiplication function `mul!(Y, A, X)` for matrices,
however, does compute the columnwise action of `A` on `X` and stores the
result in `Y`. In case there is a more efficient implementation for the
matrix application, you can provide `mul!` methods with signature
`mul!(Y::AbstractMatrix, A::MyFillMap, X::AbstractMatrix)`, and, depending
on the chosen path to handle adjoints/transposes, corresponding methods
for wrapped maps of type `AdjointMap` or `TransposeMap`, plus potentially
corresponding 5-arg `mul!` methods. This may seem like a lot of methods to
be implemented, but note that adding such methods is only necessary/recommended
for performance.