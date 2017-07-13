# LinearMaps

[![LinearMaps](http://pkg.julialang.org/badges/LinearMaps_0.6.svg)](http://pkg.julialang.org/?pkg=LinearMaps)
[![Build Status](https://travis-ci.org/Jutho/LinearMaps.jl.svg?branch=master)](https://travis-ci.org/Jutho/LinearMaps.jl)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)
[![Coverage Status](https://coveralls.io/repos/github/Jutho/LinearMaps.jl/badge.svg?branch=master)](https://coveralls.io/github/Jutho/LinearMaps.jl?branch=master)
[![codecov.io](http://codecov.io/github/Jutho/LinearMaps.jl/coverage.svg?branch=master)](http://codecov.io/github/Jutho/LinearMaps.jl?branch=master)
A Julia package for defining and working with linear maps, also known as linear transformations or linear operators acting on vectors. The only requirement for a LinearMap is that it can act on a vector (by multiplication) efficiently.

## What's new.
*   Fully julia v0.6 compatible; dropped compatibility for previous versions of Julia.

*   `LinearMap` is now the name of the abstract type on top (used to be `AbstractLinearMap` because you could not have a function/constructor with the same name as an abstract type in older julia versions)

*   Specifying the `eltype` of a function to be used as linear map should now use the constructor `LinearMap{T}(f, ...)`.

## Installation

Install with the package manager, i.e. `Pkg.add("LinearMaps")`.

## Philosophy

Several iterative linear algebra methods such as linear solvers or eigensolvers only require an efficient evaluation of the matrix vector product, where the concept of a matrix can be formalized / generalized to a linear map (or linear operator in the special case of a square matrix).

The LinearMaps package provides the following functionality:

1.  A `LinearMap` type that shares with the `AbstractMatrix` type that it responds to the functions `size`, `eltype`, `isreal`, `issymmetric`, `ishermitian` and `isposdef`, `transpose` and `ctranspose` and multiplication with a vector using both `*` or the in-place version `A_mul_B!`. Depending on the subtype, also `At_mul_B`, `At_mul_B!`, `Ac_mul_B` and `Ac_mul_B!` are supported. Linear algebra functions that uses duck-typing for its arguments can handle `LinearMap` objects similar to `AbstractMatrix` objects, provided that they can be written using the above methods. Unlike `AbstractMatrix` types, `LinearMap` objects cannot be indexed, neither using `getindex` or `setindex!`.

2.  A single method `LinearMap` function that acts as a general purpose constructor (though it only an abstract type) and allows to construct linear map objects from functions, or to wrap objects of type `AbstractMatrix` or `LinearMap`. The latter functionality is useful to (re)define the properties (`isreal`, `issymmetric`, `ishermitian`, `isposdef`) of the existing matrix or linear map.

3.  A framework for combining objects of type `LinearMap` and of type `AbstractMatrix` using linear combinations, transposition and composition, where the  linear map resulting from these operations is never explicitly evaluated but only its matrix vector product is defined (i.e. lazy evaluation). The matrix vector product is written to minimize memory allocation by using a minimal number of temporary vectors. There is full support for the in-place version `A_mul_B!`, which should be preferred for higher efficiency in critical algorithms. In addition, it tries to recognize the properties of combinations of linear maps. In particular, compositions such as `A'*A` for arbitrary `A` or even `A'*B*C*B'*A` with arbitrary `A` and `B` and positive definite `C` are recognized as being positive definite and hermitian. In case a certain property of the resulting `LinearMap` object is not correctly inferred, the `LinearMap` method can be called to redefine the properties.

## Methods

*   `LinearMap`

    General purpose method to construct `LinearMap` objects of specific types, as described in the Types section below

    ```
    LinearMap{T}(A::AbstractMatrix[; isreal::Bool, issymmetric::Bool, ishermitian::Bool, isposdef::Bool])
    LinearMap{T}(A::LinearMap[; isreal::Bool, issym::Bool, ishermitian::Bool, isposdef::Bool])
    ```

    Create a `WrappedMap` object that will respond to the methods `isreal`, `issymmetric`, `ishermitian`, `isposdef` with the values provided by the keyword arguments, and to `eltype` with the value `T`. The default values correspond to the result of calling these methods on the argument `A`; in particular `{T}` does not need to be specified and is set as `eltype(A)`. This allows to use an `AbstractMatrix` within the `LinearMap` framework and to redefine the properties of an existing `LinearMap`.

    ```
    LinearMap{T}(f, [fc = nothing], M::Int, [N::Int = M]; ismutating::Bool, issymmetric::Bool, ishermitian::Bool, isposdef::Bool])
    ```

    Create a `FunctionMap` instance that wraps an object describing the action of the linear map on a vector as a function call. Here, `f` can be a function or any other object for which either the call `f(src::AbstractVector) -> dest::AbstractVector` (when `ismutating = false`) or `f(dest::AbstractVector,src::AbstractVector) -> dest` (when `ismutating = true`) is supported. The value of `ismutating` can be spefified, by default its value is guessed by looking at the number of arguments of the first method in the method list of `f`.

    A second function or object can optionally be provided that implements the action of the adjoint (transposed) linear map. Here, it is always assumed that this represents the conjugate transpose, though this is of course equivalent to the normal transpose for real linear maps. Furthermore, the conjugate transpose also enables the use of `At_mul_B(!)` using some extra conjugation calls on the input and output vector. If no second function is provided, than `At_mul_B(!)` and `Ac_mul_B(!)` cannot be used with this linear map, unless it is symmetric or hermitian.

    `M` is the number of rows (length of the output vectors) and `N` the number of columns (length of the input vectors). When the latter is not specified, `N = M`.

    Finally, one can specify the `eltype` of the resulting linear map using the type parameter `T`. If not specified, a default value of `Float64` is assumed. Use a complex type `T` if the function represents a complex linear map.

    In summary, the keyword arguments and their default values are:

    *   `ismutating`: `false` if the function `f` accepts a single vector argument corresponding to the input, and `true` if they accept two vector arguments where the first will be mutated so as to contain the result. In both cases, the resulting `A::FunctionMap` will support both the mutating as non-mutating matrix vector multiplication. Default value is guessed based on the number of arguments for the first method in the method list of `f`; it is not possible to use `f` and `fc` where only one of the two is mutating and the other is not.
    *   `issymmetric [=false]`: whether the function represents the multiplication with a symmetric matrix. If `true`, this will automatically enable `A'*x` and `A.'*x`.
    *   `ishermitian [=T<:Real && issymmetric]`: whether the function represents the multiplication with a hermitian matrix. If `true`, this will automatically enable `A'*x` and `A.'*x`.
    *   `isposdef [=false]`: whether the function represents the multiplication with a positive definite matrix.

*   `Base.full(linearmap)`

    Creates a full matrix representation of the linearmap object, by multiplying it with the successive basis vectors. This is mostly for testing purposes or if you want to have the explicit matrix representation of a linear map for which you only have a function definition (e.g. to be able to use its `(c)transpose`).

*   All matrix multiplication methods and the corresponding mutating versions.

## Types

None of the types below need to be constructed directly; they arise from performing operations between `LinearMap` objects or by calling the `LinearMap` constructor described above.

*   `LinearMap`

    Abstract supertype

*   `FunctionMap`

    Type for wrapping an arbitrary function that is supposed to implement the matrix vector product as a `LinearMap`.

*   `WrappedMap`

    Type for wrapping an `AbstractMatrix` or `LinearMap` and to possible redefine the properties `isreal`, `issymmetric`, `ishermitian` and `isposdef`. An `AbstractMatrix` will automatically be converted to a `WrappedMap` when it is combined with other `AbstractLinearMap` objects via linear combination or composition (multiplication). Note that `WrappedMap(mat1)*WrappedMap(mat2)` will never evaluate `mat1*mat2`, since this is more costly then evaluating `mat1*(mat2*x)` and the latter is the only operation that needs to be performed by `LinearMap` objects anyway. While the cost of matrix addition is comparable to matrix vector multiplication, this too is not performed explicitly since this would require new storage of the same amount as of the original matrices.

*   `IdentityMap`

    Type for representing the identity map of a certain size `M=N`, obtained simply as `IdentityMap{T}(M)`, `IdentityMap(T,M)=IdentityMap(T,M,N)=IdentityMap(T,(M,N))` or even `IdentityMap(M)=IdentityMap(M,N)=IdentityMap((M,N))`. If `T` is not specified, `Bool` is assumed, since operations between `Bool` and any other `Number` will always be converted to the type of the other `Number`. If `M!=N`, an error is returned. An `IdentityMap` of the correct size and element type will automatically be created if `LinearMap` objects are combined with `I`, Julia's built in identity (`UniformScaling`).

*   `LinearCombination`, `CompositeMap`, `TransposeMap` and `CTransposeMap`

    Used to add and multiply `LinearMap` objects, don't need to be constructed explicitly.

## Examples

The `LinearMap` object combines well with the iterative eigensolver `eigs`, which is the Julia wrapper for Arpack.

```
using LinearMaps

function leftdiff!(y::AbstractVector, x::AbstractVector) # left difference assuming periodic boundary conditions
    N=length(x)
    length(y)==N || throw(DimensionMismatch())
        @inbounds for i=1:N
    y[i]=x[i]-x[mod1(i-1,N)]
    end
    return y
end

function mrightdiff!(y::AbstractVector, x::AbstractVector) # minus right difference
    N=length(x)
    length(y)==N || throw(DimensionMismatch())
    @inbounds for i=1:N
        y[i]=x[i]-x[mod1(i+1,N)]
    end
    return y
end

D=LinearMap(leftdiff!, mrightdiff!, 100; ismutating=true)
eigs(D'*D;nev=3,which=:SR)
```
