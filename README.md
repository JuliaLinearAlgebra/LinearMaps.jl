# LinearMaps

[![LinearMaps](http://pkg.julialang.org/badges/LinearMaps_0.5.svg)](http://pkg.julialang.org/?pkg=LinearMaps)
[![LinearMaps](http://pkg.julialang.org/badges/LinearMaps_0.6.svg)](http://pkg.julialang.org/?pkg=LinearMaps)
[![Build Status](https://travis-ci.org/Jutho/LinearMaps.jl.svg?branch=master)](https://travis-ci.org/Jutho/LinearMaps.jl)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)
[![Coverage Status](https://coveralls.io/repos/github/Jutho/LinearMaps.jl/badge.svg?branch=master)](https://coveralls.io/github/Jutho/LinearMaps.jl?branch=master)
[![codecov.io](http://codecov.io/github/Jutho/LinearMaps.jl/coverage.svg?branch=master)](http://codecov.io/github/Jutho/LinearMaps.jl?branch=master)

A Julia package for defining and working with linear maps, also known as linear transformations or linear operators acting on vectors. The only requirement for a LinearMap is that it can act on a vector (by multiplication) efficiently.

## What's new.
*   Updated to the new terminology `issymmetric` instead of `issym`. Note that the corresponding keyword argument for the `LinearMap` constructor has been modified accordingly.

*   Internal changes to better ensure type stability, especially for `FunctionMap` objects, but also for linear combinations and compositions.

## Installation

Install with the package manager, i.e. `Pkg.add("LinearMaps")`.

## Philosophy

Several iterative linear algebra methods such as linear solvers or eigensolvers only require an efficient evaluation of the matrix vector product, where the concept of a matrix can be formalized / generalized to a linear map (or linear operator in the special case of a square matrix).

The LinearMaps package provides the following functionality:

1.  An `AbstractLinearMap` type that shares with the `AbstractMatrix` type that it responds to the functions `size`, `eltype`, `isreal`, `issymmetric`, `ishermitian` and `isposdef`, `transpose` and `ctranspose` and multiplication with a vector using both `*` or the in-place version `A_mul_B!`. Depending on the subtype, also `At_mul_B`, `At_mul_B!`, `Ac_mul_B` and `Ac_mul_B!` are supported. Linear algebra functions that uses duck-typing for its arguments can handle `AbstractLinearMap` objects similar to `AbstractMatrix` objects, provided that they can be written using the above methods. Unlike `AbstractMatrix` types, `AbstractLinearMap` objects cannot be indexed, neither using `getindex` or `setindex!`.

2.  A single method `LinearMap` function that acts as a general purpose constructor (though it is not a real type) and allows to construct `AbstractLinearMap` objects from functions, or to wrap objects of type `AbstractMatrix` or `AbstractLinearMap`. This method thus can also be used to (re)define the properties (`isreal`, `issymmetric`, `ishermitian`, `isposdef`) of the corresponding linear map.

3.  A framework for combining objects of type `AbstractLinearMap` and of type `AbstractMatrix` using linear combinations, transposition and composition, where the  linear map resulting from these operations is never explicitly evaluated but only its matrix vector product is defined (i.e. lazy evaluation). The matrix vector product is written to minimize memory allocation by using a minimal number of temporary vectors. There is full support for the in-place version `A_mul_B!`, which should be preferred for higher efficiency in critical algorithms. In addition, it tries to recognize the properties of combinations of linear maps. In particular, compositions such as `A'*A` for arbitrary `A` or even `A'*B*C*B'*A` with arbitrary `A` and `B` and positive definite `C` are recognized as being positive definite and hermitian. In case a certain property of the resulting `AbstractLinearMap` object is not correctly inferred, the `LinearMap` method can be called to redefine the properties.

## Methods

*   `LinearMap`

    General purpose method to construct AbstractLinearMap objects of specific types, as described in the Types section below

    ```
    LinearMap(A::AbstractMatrix[; isreal::Bool, issymmetric::Bool, ishermitian::Bool, isposdef::Bool])
    LinearMap(A::AbstractLinearMap[; isreal::Bool, issym::Bool, ishermitian::Bool, isposdef::Bool])
    ```

    Create a `WrappedMap` object that will respond to the methods `isreal`, `issymmetric`, `ishermitian`, `isposdef` with the values provided by the keyword arguments. The default values correspond to the result of calling these methods on the argument `A`. This allows to use an `AbstractMatrix` within the `AbstractLinearMap` framework and to redefine the properties of an existing `AbstractLinearMap`.

    ```
    LinearMap(f, [fc = nothing], M::Int, [N::Int = M, eltype::Type = Float64]; ismutating::Bool, issymmetric::Bool, ishermitian::Bool, isposdef::Bool])
    ```

    Create `FunctionMap` object that wraps a function describing the action of the linear map on a vector. The corresponding properties of the linear map can also be specified. Here, `f` represents the function implementing the action of the linear map on a vector, either as returning the result (i.e. `f(src::AbstractVector) -> dest::AbstractVector`) when `ismutating = false` (default) or as a mutating function that accepts a vector for the destination (i.e. `f(dest::AbstractVector,src::AbstractVector) -> dest`).

    A second function can optionally be provided that implements the action of the adjoint (transposed) linear map. Here, it is always assumed that this represents the conjugate transpose, though this is of course equivalent to the normal transpose for real linear maps. Furthermore, the conjugate transpose also enables the use of `At_mul_B(!)` using some extra conjugation calls on the input and output vector. If no second function is provided, than `At_mul_B(!)` and `Ac_mul_B(!)` cannot be used with this linear map, unless it is symmetric or hermitian.

    `M` is the number of rows (length of the output vectors) and `N` the number of columns (length of the input vectors). When the latter is not specified, `N = M`.

    Finally, one can specify the `eltype` of the resulting linear map as final normal argument, where a default value of `Float64` is assumed. If the function acts as a  complex linear map, than one should provide a complex type such as `Complex128`.

    The keyword arguments and their default values are:

    *   `ismutating [=false]`: `false` if the function `f` accepts a single vector argument corresponding to the input, and `true` if they accept two vector arguments where the first will be mutated so as to contain the result. In both cases, the resulting `A::FunctionMap` will support both the mutating as nonmutating matrix vector multiplication.
    *   `issymmetric [=false]`: whether the function represents the multiplication with a symmetric matrix. If `true`, this will automatically enable `A'*x` and `A.'*x`.
    *   `ishermitian [=T<:Real && issymmetric]`: whether the function represents the multiplication with a hermitian matrix. If `true`, this will automatically enable `A'*x` and `A.'*x`.
    *   `isposdef [=false]`: whether the function represents the multiplication with a positive definite matrix.

*   `Base.full(linearmap)`

    Creates a full matrix representation of the linearmap object, by multiplying it with the successive basis vectors. This is mostly for testing purposes

*   All matrix multiplication methods and the corresponding mutating versions.

## Types

None of the types below need to be constructed directly; they arise from performing operations between `AbstractLinearMap` objects or by calling the `LinearMap` method described above.

*   `AbstractLinearMap`

    Abstract supertype

*   `FunctionMap`

    Type for wrapping an arbitrary function that is supposed to implement the matrix vector product as an `AbstractLinearMap`.

*   `WrappedMap`

    Type for wrapping an `AbstractMatrix` or `AbstractLinearMap` and to possible redefine the properties `isreal`, `issym`, `ishermitian` and `isposdef`. An `AbstractMatrix` will automatically be converted to a `WrappedMap` when it is combined with other `AbstractLinearMap` objects via linear combination or composition (multiplication). Note that `WrappedMap(mat1)*WrappedMap(mat2)` will never evaluate `mat1*mat2`, since this is more costly then evaluating `mat1*(mat2*x)` and the latter is the only operation that needs to be performed by `AbstractLinearMap` objects anyway. While the cost of matrix addition is comparible to matrix vector multiplication, this too is not performed explicitly since this would require new storage of the same amount as of the original matrices.

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
