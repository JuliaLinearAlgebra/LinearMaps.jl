# What's new?

### What's new in v2.7
*   Potential reduction of memory allocations in multiplication of
    `LinearCombination`s, `BlockMap`s, and real- or complex-scaled `LinearMap`s.
    For the latter, a new internal type `ScaledMap` has been introduced.
*   Multiplication code for `CompositeMap`s has been refactored to facilitate to
    provide memory for storage of intermediate results by directly calling helper
    functions.

### What's new in v2.6
*   New feature: "lazy" Kronecker product, Kronecker sums, and powers thereof
    for `LinearMap`s. `AbstractMatrix` objects are promoted to `LinearMap`s if
    one of the first 8 Kronecker factors is a `LinearMap` object.
*   Compatibility with the generic multiply-and-add interface (a.k.a. 5-arg
    `mul!`) introduced in julia v1.3

### What's new in v2.5
*   New feature: concatenation of `LinearMap`s objects with `UniformScaling`s,
    consistent with (h-, v-, and hc-)concatenation of matrices. Note, matrices
    `A` must be wrapped as `LinearMap(A)`, `UniformScaling`s are promoted to
    `LinearMap`s automatically.

### What's new in v2.4
*   Support restricted to Julia v1.0+.

### What's new in v2.3
*   Fully Julia v0.7/v1.0/v1.1 compatible.
*   Full support of noncommutative number types such as quaternions.

### What's new in v2.2
*   Fully Julia v0.7/v1.0 compatible.
*   A `convert(SparseMatrixCSC, A::LinearMap)` function, that calls the `sparse`
    matrix generating function.

### What's new in v2.1
*   Fully Julia v0.7 compatible; dropped compatibility for previous versions of
    Julia from LinearMaps.jl v2.0.0 on.
*   A 5-argument version for `mul!(y, A::LinearMap, x, α=1, β=0)`, which
    computes `y := α * A * x + β * y` and implements the usual 3-argument
    `mul!(y, A, x)` for the default `α` and `β`.
*   Synonymous `convert(Matrix, A::LinearMap)` and `convert(Array, A::LinearMap)`
    functions, that call the `Matrix` constructor and return the matrix
    representation of `A`.
*   Multiplication with matrices, interpreted as a block row vector of vectors:
    * `mul!(Y::AbstractArray, A::LinearMap, X::AbstractArray, α=1, β=0)`:
      applies `A` to each column of `X` and stores the result in-place in the
      corresponding column of `Y`;
    * for the out-of-place multiplication, the approach is to compute
      `convert(Matrix, A * X)`; this is equivalent to applying `A` to each
      column of `X`. In generic code which handles both `A::AbstractMatrix` and
      `A::LinearMap`, the additional call to `convert` is a noop when `A` is a
      matrix.
*   Full compatibility with [Arpack.jl](https://github.com/JuliaLinearAlgebra/Arpack.jl)'s
    `eigs` and `svds`; previously only `eigs` was working. For more, nicely
    collaborating packages see the [Example](#example) section.
