# Types

None of the types below need to be constructed directly; they arise from
performing operations between `LinearMap` objects or by calling the `LinearMap`
constructor described in [Methods](@ref).

## `LinearMap`

Abstract supertype

```@docs
LinearMaps.LinearMap
```

General purpose method to construct `LinearMap` objects of specific types,
as described in the [Types](#types) section below

```
LinearMap{T}(A::AbstractMatrix[; issymmetric::Bool, ishermitian::Bool, isposdef::Bool])
LinearMap{T}(A::LinearMap [; issymmetric::Bool, ishermitian::Bool, isposdef::Bool])
```
Create a `WrappedMap` object that will respond to the methods `issymmetric`,
`ishermitian`, `isposdef` with the values provided by the keyword arguments,
and to `eltype` with the value `T`. The default values correspond to the
result of calling these methods on the argument `A`; in particular `{T}`
does not need to be specified and is set as `eltype(A)`. This allows to use
an `AbstractMatrix` within the `LinearMap` framework and to redefine the
properties of an existing `LinearMap`.

```
LinearMap{T}(f, [fc = nothing], M::Int, [N::Int = M]; ismutating::Bool, issymmetric::Bool, ishermitian::Bool, isposdef::Bool])
```
Create a `FunctionMap` instance that wraps an object describing the action
of the linear map on a vector as a function call. Here, `f` can be a
function or any other object for which either the call
`f(src::AbstractVector) -> dest::AbstractVector` (when `ismutating = false`)
or `f(dest::AbstractVector,src::AbstractVector) -> dest` (when
`ismutating = true`) is supported. The value of `ismutating` can be
specified, by default its value is guessed by looking at the number of
arguments of the first method in the method list of `f`.

A second function or object `fc` can optionally be provided that implements
the action of the adjoint (transposed) linear map. Here, it is always
assumed that this represents the adjoint/conjugate transpose, though this is
of course equivalent to the normal transpose for real linear maps.
Furthermore, the conjugate transpose also enables the use of
`mul!(y, transpose(A), x)` using some extra conjugation calls on the input
and output vector. If no second function is provided, than
`mul!(y, transpose(A), x)` and `mul!(y, adjoint(A), x)` cannot be used with
this linear map, unless it is symmetric or hermitian.

`M` is the number of rows (length of the output vectors) and `N` the number
of columns (length of the input vectors). When the latter is not specified,
`N = M`.

Finally, one can specify the `eltype` of the resulting linear map using the
type parameter `T`. If not specified, a default value of `Float64` is
assumed. Use a complex type `T` if the function represents a complex linear
map.

The keyword arguments and their default values are:

*   `ismutating`: `false` if the function `f` accepts a single vector
    argument corresponding to the input, and `true` if it accepts two vector
    arguments where the first will be mutated so as to contain the result.
    In both cases, the resulting `A::FunctionMap` will support both the
    mutating and non-mutating matrix-vector multiplication. Default value is
    guessed based on the number of arguments for the first method in the
    method list of `f`; it is not possible to use `f` and `fc` where only
    one of the two is mutating and the other is not.
*   `issymmetric [=false]`: whether the function represents the
    multiplication with a symmetric matrix. If `true`, this will
    automatically enable `A' * x` and `transpose(A) * x`.
*   `ishermitian [=T<:Real && issymmetric]`: whether the function represents
    the multiplication with a hermitian matrix. If `true`, this will
    automatically enable `A' * x` and `transpose(A) * x`.
*   `isposdef [=false]`: whether the function represents the multiplication
    with a positive definite matrix.

```
LinearMap(J::UniformScaling, M::Int)
```
Create a `UniformScalingMap` instance that corresponds to a uniform scaling
map of size `M`×`M`.

## `FunctionMap`

Type for wrapping an arbitrary function that is supposed to implement the
matrix-vector product as a `LinearMap`.

## `WrappedMap`

Type for wrapping an `AbstractMatrix` or `LinearMap` and to possible redefine
the properties `isreal`, `issymmetric`, `ishermitian` and `isposdef`. An
`AbstractMatrix` will automatically be converted to a `WrappedMap` when it is
combined with other `AbstractLinearMap` objects via linear combination or
composition (multiplication). Note that `WrappedMap(mat1)*WrappedMap(mat2)`
will never evaluate `mat1*mat2`, since this is more costly than evaluating
`mat1*(mat2*x)` and the latter is the only operation that needs to be performed
by `LinearMap` objects anyway. While the cost of matrix addition is comparable
to matrix-vector multiplication, this too is not performed explicitly since
this would require new storage of the same amount as of the original matrices.

## `ScaledMap`

Type for representing a scalar multiple of any `LinearMap` type. A
`ScaledMap` will be automatically constructed if real or complex `LinearMap`
objects are multiplied by real or complex scalars from the left or from the
right.

## `UniformScalingMap`

Type for representing a scalar multiple of the identity map (a.k.a. uniform
scaling) of a certain size `M=N`, obtained simply as `UniformScalingMap(λ, M)`.
The type `T` of the resulting `LinearMap` object is inferred from the type of
`λ`. A `UniformScalingMap` of the correct size will be automatically
constructed if `LinearMap` objects are multiplied by scalars from the left
or from the right (respecting the order of multiplication), if either the
`eltype` of the `LinearMap` or the scalar are of non-commutative type, .

## `LinearCombination`. `CompositeMap`, `TransposeMap` and `AdjointMap`

Used to add and multiply `LinearMap` objects, don't need to be constructed explicitly.

```@docs
+(::LinearMap,::LinearMap)
*(::LinearMap,::LinearMap)
LinearAlgebra.transpose(::LinearMap)
LinearAlgebra.adjoint(::LinearMap)
```

## `KroneckerMap` and `KroneckerSumMap`

Types for representing Kronecker products and Kronecker sums, resp., lazily.

```@docs
Base.kron(::LinearMap,::LinearMap)
LinearMaps.:⊗
kronsum
LinearMaps.:⊕
```

## `BlockMap` and `BlockDiagonalMap`

Types for representing block (diagonal) maps lazily.

```@docs
Base.hcat
Base.vcat
Base.hvcat
```