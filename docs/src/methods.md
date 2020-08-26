# Methods

## `LinearMap`

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

## `Array(A::LinearMap)`, `Matrix(A::LinearMap)`, `convert(Matrix, A::LinearMap)` and `convert(Array, A::LinearMap)`

Create a dense matrix representation of the `LinearMap` object, by
multiplying it with the successive basis vectors. This is mostly for testing
purposes or if you want to have the explicit matrix representation of a
linear map for which you only have a function definition (e.g. to be able to
use its `transpose` or `adjoint`). This way, one may conveniently make `A`
act on the columns of a matrix `X`, instead of interpreting `A * X` as a
composed linear map: `Matrix(A * X)`. For generic code, that is supposed to
handle both `A::AbstractMatrix` and `A::LinearMap`, it is recommended to use
`convert(Matrix, A*X)`.

## `convert(AbstractMatrix, A::LinearMap)`, `convert(AbstractArray, A::LinearMap)`

Create an `AbstractMatrix` representation of the `LinearMap`. This falls
back to `Matrix(A)`, but avoids explicit construction in case the `LinearMap`
object is matrix-based.

## `SparseArrays.sparse(A::LinearMap)` and `convert(SparseMatrixCSC, A::LinearMap)`

Create a sparse matrix representation of the `LinearMap` object, by
multiplying it with the successive basis vectors. This is mostly for testing
purposes or if you want to have the explicit sparse matrix representation of
a linear map for which you only have a function definition (e.g. to be able
to use its `transpose` or `adjoint`).

## Multiplication methods

* `A * x`: applies `A` to `x` and returns the result;
* `mul!(y::AbstractVector, A::LinearMap, x::AbstractVector)`: applies `A` to
    `x` and stores the result in `y`;
* `mul!(Y::AbstractMatrix, A::LinearMap, X::AbstractMatrix)`: applies `A` to
    each column of `X` and stores the results in the corresponding columns of
    `Y`;
* `mul!(y::AbstractVector, A::LinearMap, x::AbstractVector, α::Number, β::Number)`:
    computes `A * x * α + y * β` and stores the result in `y`. Analogously for `X,Y::AbstractMatrix`.

Applying the adjoint or transpose of `A` (if defined) to `x` works exactly
as in the usual matrix case: `transpose(A) * x` and `mul!(y, A', x)`, for instance.
