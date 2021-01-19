# Types and methods

## Types and their constructors

None of the types below need to be constructed directly; they arise from
performing operations between `LinearMap` objects or by calling the `LinearMap`
constructor described next.

### `LinearMap`

Abstract supertype

```@docs
LinearMaps.LinearMap
```

### `FunctionMap`

Type for wrapping an arbitrary function that is supposed to implement the
matrix-vector product as a `LinearMap`; see above.

### `WrappedMap`

Type for wrapping an `AbstractMatrix` or `LinearMap` and to possible redefine
the properties `isreal`, `issymmetric`, `ishermitian` and `isposdef`. An
`AbstractMatrix` will automatically be converted to a `WrappedMap` when it is
combined with other `LinearMap` objects via linear combination or
composition (multiplication). Note that `WrappedMap(mat1)*WrappedMap(mat2)`
will never evaluate `mat1*mat2`, since this is more costly than evaluating
`mat1*(mat2*x)` and the latter is the only operation that needs to be performed
by `LinearMap` objects anyway. While the cost of matrix addition is comparable
to matrix-vector multiplication, this too is not performed explicitly since
this would require new storage of the same amount as of the original matrices.

### `ScaledMap`

Type for representing a scalar multiple of any `LinearMap` type. A
`ScaledMap` will be automatically constructed if real or complex `LinearMap`
objects are multiplied by real or complex scalars from the left or from the
right.

### `UniformScalingMap`

Type for representing a scalar multiple of the identity map (a.k.a. uniform
scaling) of a certain size `M=N`, obtained simply as `LinearMap(λI, M)`,
where `I` is the `LinearAlgebra.UniformScaling` object.
The type `T` of the resulting `LinearMap` object is inferred from the type of
`λ`. A `UniformScalingMap` of the correct size will be automatically
constructed if `LinearMap` objects are multiplied by scalars from the left
or from the right (respecting the order of multiplication), if the scalar `λ`
is either real or complex.

### `LinearCombination`, `CompositeMap`, `TransposeMap` and `AdjointMap`

Used to add/multiply/transpose/adjoint `LinearMap` objects lazily, don't need to be constructed explicitly.

```@docs
+(::LinearMap,::LinearMap)
*(::LinearMap,::LinearMap)
LinearAlgebra.transpose(::LinearMap)
LinearAlgebra.adjoint(::LinearMap)
```

### `KroneckerMap` and `KroneckerSumMap`

Types for representing Kronecker products and Kronecker sums, resp., lazily.

```@docs
Base.kron(::LinearMap,::LinearMap)
LinearMaps.:⊗
kronsum
LinearMaps.:⊕
```

### `BlockMap` and `BlockDiagonalMap`

Types for representing block (diagonal) maps lazily.

```@docs
Base.hcat
Base.vcat
Base.hvcat
Base.cat
SparseArrays.blockdiag
```

### `FillMap`

Type for lazily representing constantly filled matrices.

```@docs
LinearMaps.FillMap
```

## Methods

### Multiplication methods

```@docs
Base.:*(::LinearMap,::AbstractVector)
Base.:*(::LinearMap,::AbstractMatrix)
Base.:*(::AbstractMatrix,::LinearMap)
LinearAlgebra.mul!(::AbstractVecOrMat,::LinearMap,::AbstractVector)
LinearAlgebra.mul!(::AbstractVecOrMat,::LinearMap,::AbstractVector,::Number,::Number)
LinearAlgebra.mul!(::AbstractMatrix,::AbstractMatrix,::LinearMap)
*(::LinearAlgebra.AdjointAbsVec,::LinearMap)
*(::LinearAlgebra.TransposeAbsVec,::LinearMap)
```

Applying the adjoint or transpose of `A` (if defined) to `x` works exactly
as in the usual matrix case: `transpose(A) * x` and `mul!(y, A', x)`, for instance.

### Conversion methods

* `Array`, `Matrix` and associated `convert` methods

  Create a dense matrix representation of the `LinearMap` object, by
  multiplying it with the successive basis vectors. This is mostly for testing
  purposes or if you want to have the explicit matrix representation of a
  linear map for which you only have a function definition (e.g. to be able to
  use its `transpose` or `adjoint`). This way, one may conveniently make `A`
  act on the columns of a matrix `X`, instead of interpreting `A * X` as a
  composed linear map: `Matrix(A * X)`. For generic code, that is supposed to
  handle both `A::AbstractMatrix` and `A::LinearMap`, it is recommended to use
  `convert(Matrix, A*X)`.

* `convert(Abstract[Matrix/Array], A::LinearMap)`

  Create an `AbstractMatrix` representation of the `LinearMap`. This falls
  back to `Matrix(A)`, but avoids explicit construction in case the `LinearMap`
  object is matrix-based.

* `SparseArrays.sparse(A::LinearMap)` and `convert(SparseMatrixCSC, A::LinearMap)`

  Create a sparse matrix representation of the `LinearMap` object, by
  multiplying it with the successive basis vectors. This is mostly for testing
  purposes or if you want to have the explicit sparse matrix representation of
  a linear map for which you only have a function definition (e.g. to be able
  to use its `transpose` or `adjoint`).
