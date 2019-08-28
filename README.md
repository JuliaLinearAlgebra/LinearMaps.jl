# LinearMaps

[![Build Status](https://travis-ci.org/Jutho/LinearMaps.jl.svg?branch=master)](https://travis-ci.org/Jutho/LinearMaps.jl)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)
[![Coverage Status](https://coveralls.io/repos/github/Jutho/LinearMaps.jl/badge.svg?branch=master)](https://coveralls.io/github/Jutho/LinearMaps.jl?branch=master)
[![codecov.io](http://codecov.io/github/Jutho/LinearMaps.jl/coverage.svg?branch=master)](http://codecov.io/github/Jutho/LinearMaps.jl?branch=master)

A Julia package for defining and working with linear maps, also known as linear transformations or linear operators acting on vectors. The only requirement for a LinearMap is that it can act on a vector (by multiplication) efficiently.

## What's new in v2.6
*   New feature: "lazy" Kronecker product of `LinearMap`s. `AbstractMatrix`es are
    promoted to `LinearMap`s if one of the first 8 Kronecker factors is a
    `LinearMap` object.

## What's new in v2.5.0
*   New feature: concatenation of `LinearMap`s objects with `UniformScaling`s,
    consistent with (h-, v-, and hc-)concatenation of matrices. Note, matrices
    `A` must be wrapped as `LinearMap(A)`, `UniformScaling`s are promoted to
    `LinearMap`s automatically.

## What's new in v2.4.0
*   Support restricted to Julia v1.0+.

## What's new in v2.3.0
*   Fully Julia v0.7/v1.0/v1.1 compatible.
*   Full support of noncommutative number types such as quaternions.

## What's new in v2.2.0
*   Fully Julia v0.7/v1.0 compatible.
*   A `convert(SparseMatrixCSC, A::LinearMap)` function, that calls the `sparse` matrix generating function.

## What's new in v2.1.0
*   Fully Julia v0.7 compatible; dropped compatibility for previous versions of Julia from LinearMaps.jl v2.0.0 on.
*   A 5-argument version for `mul!(y, A::LinearMap, x, α=1, β=0)`, which computes `y := α * A * x + β * y` and implements the usual 3-argument `mul!(y, A, x)` for the default `α` and `β`.
*   Synonymous `convert(Matrix, A::LinearMap)` and `convert(Array, A::LinearMap)` functions, that call the `Matrix` constructor and return the matrix representation of `A`.
*   Multiplication with matrices, interpreted as a block row vector of vectors:
    * `mul!(Y::AbstractArray, A::LinearMap, X::AbstractArray, α=1, β=0)`: applies `A` to each column of `X` and stores the result in-place in the corresponding column of `Y`;
    * for the out-of-place multiplication, the approach is to compute `convert(Matrix, A * X)`; this is equivalent to applying `A` to each column of `X`. In generic code which handles both `A::AbstractMatrix` and `A::LinearMap`, the additional call to `convert` is a noop when `A` is a matrix.
*   Full compatibility with [Arpack.jl](https://github.com/JuliaLinearAlgebra/Arpack.jl)'s `eigs` and `svds`; previously only `eigs` was working. For more, nicely collaborating packages see the [Example](#example) section.

## Installation

Install with the package manager, i.e., `]add LinearMaps` (or `Pkg.add("LinearMaps")` in Julia versions below 0.7).

## Philosophy

Several iterative linear algebra methods such as linear solvers or eigensolvers only require an efficient evaluation of the matrix vector product, where the concept of a matrix can be formalized / generalized to a linear map (or linear operator in the special case of a square matrix).

The LinearMaps package provides the following functionality:

1.  A `LinearMap` type that shares with the `AbstractMatrix` type that it responds to the functions `size`, `eltype`, `isreal`, `issymmetric`, `ishermitian` and `isposdef`, `transpose` and `adjoint` and multiplication with a vector using both `*` or the in-place version `mul!`. Linear algebra functions that use duck-typing for their arguments can handle `LinearMap` objects similar to `AbstractMatrix` objects, provided that they can be written using the above methods. Unlike `AbstractMatrix` types, `LinearMap` objects cannot be indexed, neither using `getindex` or `setindex!`.

2.  A single method `LinearMap` function that acts as a general purpose constructor (though it is only an abstract type) and allows to construct linear map objects from functions, or to wrap objects of type `AbstractMatrix` or `LinearMap`. The latter functionality is useful to (re)define the properties (`isreal`, `issymmetric`, `ishermitian`, `isposdef`) of the existing matrix or linear map.

3.  A framework for combining objects of type `LinearMap` and of type `AbstractMatrix` using linear combinations, transposition and composition, where the linear map resulting from these operations is never explicitly evaluated but only its matrix vector product is defined (i.e. lazy evaluation). The matrix vector product is written to minimize memory allocation by using a minimal number of temporary vectors. There is full support for the in-place version `mul!`, which should be preferred for higher efficiency in critical algorithms. In addition, it tries to recognize the properties of combinations of linear maps. In particular, compositions such as `A'*A` for arbitrary `A` or even `A'*B*C*B'*A` with arbitrary `A` and `B` and positive definite `C` are recognized as being positive definite and hermitian. In case a certain property of the resulting `LinearMap` object is not correctly inferred, the `LinearMap` method can be called to redefine the properties.

## Methods

*   `LinearMap`

    General purpose method to construct `LinearMap` objects of specific types, as described in the [Types](#types) section below

    ```
    LinearMap{T}(A::AbstractMatrix[; issymmetric::Bool, ishermitian::Bool, isposdef::Bool])
    LinearMap{T}(A::LinearMap[; issymmetric::Bool, ishermitian::Bool, isposdef::Bool])
    ```

    Create a `WrappedMap` object that will respond to the methods `issymmetric`, `ishermitian`, `isposdef` with the values provided by the keyword arguments, and to `eltype` with the value `T`. The default values correspond to the result of calling these methods on the argument `A`; in particular `{T}` does not need to be specified and is set as `eltype(A)`. This allows to use an `AbstractMatrix` within the `LinearMap` framework and to redefine the properties of an existing `LinearMap`.

    ```
    LinearMap{T}(f, [fc = nothing], M::Int, [N::Int = M]; ismutating::Bool, issymmetric::Bool, ishermitian::Bool, isposdef::Bool])
    ```

    Create a `FunctionMap` instance that wraps an object describing the action of the linear map on a vector as a function call. Here, `f` can be a function or any other object for which either the call `f(src::AbstractVector) -> dest::AbstractVector` (when `ismutating = false`) or `f(dest::AbstractVector,src::AbstractVector) -> dest` (when `ismutating = true`) is supported. The value of `ismutating` can be spefified, by default its value is guessed by looking at the number of arguments of the first method in the method list of `f`.

    A second function or object can optionally be provided that implements the action of the adjoint (transposed) linear map. Here, it is always assumed that this represents the adjoint/conjugate transpose, though this is of course equivalent to the normal transpose for real linear maps. Furthermore, the conjugate transpose also enables the use of `mul!(y, transpose(A), x)` using some extra conjugation calls on the input and output vector. If no second function is provided, than `mul!(y, transpose(A), x)` and `mul!(y, adjoint(A), x)` cannot be used with this linear map, unless it is symmetric or hermitian.

    `M` is the number of rows (length of the output vectors) and `N` the number of columns (length of the input vectors). When the latter is not specified, `N = M`.

    Finally, one can specify the `eltype` of the resulting linear map using the type parameter `T`. If not specified, a default value of `Float64` is assumed. Use a complex type `T` if the function represents a complex linear map.

    In summary, the keyword arguments and their default values are:

    *   `ismutating`: `false` if the function `f` accepts a single vector argument corresponding to the input, and `true` if it accepts two vector arguments where the first will be mutated so as to contain the result. In both cases, the resulting `A::FunctionMap` will support both the mutating and non-mutating matrix vector multiplication. Default value is guessed based on the number of arguments for the first method in the method list of `f`; it is not possible to use `f` and `fc` where only one of the two is mutating and the other is not.
    *   `issymmetric [=false]`: whether the function represents the multiplication with a symmetric matrix. If `true`, this will automatically enable `A' * x` and `transpose(A) * x`.
    *   `ishermitian [=T<:Real && issymmetric]`: whether the function represents the multiplication with a hermitian matrix. If `true`, this will automatically enable `A' * x` and `transpose(A) * x`.
    *   `isposdef [=false]`: whether the function represents the multiplication with a positive definite matrix.

*   `Array(A::LinearMap)`, `Matrix(A::LinearMap)`, `convert(Matrix, A::LinearMap)` and `convert(Array, A::LinearMap)`

    Create a dense matrix representation of the `LinearMap` object, by multiplying it with the successive basis vectors. This is mostly for testing purposes or if you want to have the explicit matrix representation of a linear map for which you only have a function definition (e.g. to be able to use its `transpose` or `adjoint`). This way, one may conveniently make `A` act on the columns of a matrix `X`, instead of interpreting `A * X` as a composed linear map: `Matrix(A * X)`. For generic code, that is supposed to handle both `A::AbstractMatrix` and `A::LinearMap`, it is recommended to use `convert(Matrix, A*X)`.

*   `SparseArrays.sparse(A::LinearMap)` and `convert(SparseMatrixCSC, A::LinearMap)`

    Create a sparse matrix representation of the `LinearMap` object, by multiplying it with the successive basis vectors. This is mostly for testing purposes or if you want to have the explicit sparse matrix representation of a linear map for which you only have a function definition (e.g. to be able to use its `transpose` or `adjoint`).

*   The following matrix multiplication methods.

    * `A * x`: applies `A` to `x` and returns the result;
    * `mul!(y::AbstractVector, A::LinearMap, x::AbstractVector)`: applies `A` to `x` and stores the result in `y`;
    * `mul!(Y::AbstractMatrix, A::LinearMap, X::AbstractMatrix)`: applies `A` to each column of `X` and stores the results in the corresponding columns of `Y`;
    * `mul!(y::AbstractVector, A::LinearMap, x::AbstractVector, α::Number=1, β::Number=0)`: computes `α * A * x + β * y` and stores the result in `y`. Analogously for `X,Y::AbstractMatrix`.

    Applying the adjoint or transpose of `A` (if defined) to `x` works exactly as in the usual matrix case: `transpose(A) * x` and `mul!(y, A', x)`, for instance.

*   Horizontal, vertical, and hv-concatenation as known for regular matrices,
    where `UniformScaling`s are automatically promoted to `LinearMap`s and their
    sizes are inferred, if possible.

## Types

None of the types below need to be constructed directly; they arise from performing
operations between `LinearMap` objects or by calling the `LinearMap` constructor
described above.

*   `LinearMap`

    Abstract supertype

*   `FunctionMap`

    Type for wrapping an arbitrary function that is supposed to implement the
    matrix vector product as a `LinearMap`.

*   `WrappedMap`

    Type for wrapping an `AbstractMatrix` or `LinearMap` and to possible redefine
    the properties `isreal`, `issymmetric`, `ishermitian` and `isposdef`. An
    `AbstractMatrix` will automatically be converted to a `WrappedMap` when it is
    combined with other `AbstractLinearMap` objects via linear combination or
    composition (multiplication). Note that `WrappedMap(mat1)*WrappedMap(mat2)`
    will never evaluate `mat1*mat2`, since this is more costly then evaluating
    `mat1*(mat2*x)` and the latter is the only operation that needs to be performed
    by `LinearMap` objects anyway. While the cost of matrix addition is comparable
    to matrix vector multiplication, this too is not performed explicitly since
    this would require new storage of the same amount as of the original matrices.

*   `UniformScalingMap`

    Type for representing a scalar multiple of the identity map (a.k.a. uniform
    scaling) of a certain size `M=N`, obtained simply as `UniformScalingMap(λ, M)`.
    The type `T` of the resulting `LinearMap` object is inferred from the type of
    `λ`. A `UniformScalingMap` of the correct size will be automatically created
    if `LinearMap` objects are multiplied by scalars from the left or from the right,
    respecting the order of multiplication.

*   `LinearCombination`, `CompositeMap`, `TransposeMap` and `AdjointMap`

    Used to add and multiply `LinearMap` objects, don't need to be constructed explicitly.

## Example

The `LinearMap` object combines well with methods of the following packages:
* [Arpack.jl](https://github.com/JuliaLinearAlgebra/Arpack.jl): iterative eigensolver
  `eigs` and SVD `svds`;
* [IterativeSolvers.jl](https://github.com/JuliaMath/IterativeSolvers.jl): iterative
  solvers, eigensolvers, and SVD;
* [TSVD.jl](https://github.com/andreasnoack/TSVD.jl): truncated SVD `tsvd`.

```
using LinearMaps
import Arpack, IterativeSolvers, TSVD

function leftdiff!(y::AbstractVector, x::AbstractVector) # left difference assuming periodic boundary conditions
    N = length(x)
    length(y) == N || throw(DimensionMismatch())
    @inbounds for i=1:N
        y[i] = x[i] - x[mod1(i-1, N)]
    end
    return y
end

function mrightdiff!(y::AbstractVector, x::AbstractVector) # minus right difference
    N = length(x)
    length(y) == N || throw(DimensionMismatch())
    @inbounds for i=1:N
        y[i] = x[i] - x[mod1(i+1, N)]
    end
    return y
end

D = LinearMap(leftdiff!, mrightdiff!, 100; ismutating=true) # by default has eltype(D) = Float64

Arpack.eigs(D'D; nev=3, which=:SR) # note that D'D is recognized as symmetric => real eigenfact
Arpack.svds(D; nsv=3)

Σ, L = IterativeSolvers.svdl(D; nsv=3)

TSVD.tsvd(D, 3)
```
