# Version history

## What's new in v3.5

* `WrappedMap`, `ScaledMap`, and `LinearCombination`, instead of using the default `axes(A) 
  = map.(oneto, size(A))`, now forward calls to axes to the underlying wrapped linear map. 
  This allows allocating operations such as `*` to determine the appropriate storage and axes 
  type of their outputs. For example, linear maps that wrap `BlockArrays` will, upon
  multiplicative action, produce a `BlockArrays.PseudoBlockVector` with block structure
  inherited from the operator's *output* axes `axes(A,1)`.

## What's new in v3.4

* In `WrappedMap` constructors, as implicitly called in addition and mutliplication
  of `LinearMap`s and `AbstractMatrix` objects, (conjugate) symmetry and positive
  definiteness are only determined for matrix types for which these checks are expected
  to be very cheap or even known at compile time based on the concrete type. The default
  for `LinearMap` subtypes is to call, for instance, `issymmetric`, because symmetry
  properties are either stored or easily obtained from constituting maps. For custom matrix
  types, define corresponding methods `LinearMaps._issymmetric`, `LinearMaps._ishermitian`
  and `LinearMaps._isposdef` to hook into the property checking mechanism.

## What's new in v3.3

* `AbstractVector`s can now be wrapped by a `LinearMap` just like `AbstractMatrix``
  typed objects. Upon wrapping, there are not implicitly reshaped to matrices. This
  feature might be helpful, for instance, in the lazy representation of rank-1
  operators `kron(LinearMap(u), v') == ⊗(u, v') == u ⊗ v'` for vectors `u` and `v`.
  The action on vectors,`(u⊗v')*x`, is implemented optimally via `u*(v'x)`.

## What's new in v3.2

* In-place left-multiplication `mul!(Y, X, A::LinearMap)` is now allowed for
  `X::AbstractMatrix` and implemented via the adjoint equation `Y' = A'X'`.

## What's new in v3.1

* In Julia v1.3 and above, `LinearMap`-typed objects are callable on `AbstractVector`s:
  For `L::LinearMap` and `x::AbstractVector`, `L(x) = L*x`.

## What's new in v3.0

* BREAKING change: Internally, any dependence on former `A*_mul_B!` methods is abandonned.
  For custom `LinearMap` subtypes, there are now two options:
  1. In case your type is invariant under adjoint/transposition (i.e.,
     `adjoint(L::MyLinearMap)::MyLinearMap` similar to, for instance,
     `LinearCombination`s or `CompositeMap`s, `At_mul_B!` and `Ac_mul_B!` do
     not require any replacement! Rather, multiplication by `L'` is, in this case,
     handled by `mul!(y, L::MyLinearMap, x[, α, β])`.
  2. Otherwise, you will need to define `mul!` methods with the signature
     `mul!(y, L::TransposeMap{<:Any,MyLinearMap}, x[, α, β])` and
     `mul!(y, L::AdjointMap{<:Any,MyLinearMap}, x[, α, β])`.
* Left multiplying by a transpose or adjoint vector (e.g., `y'*A`)
  produces a transpose or adjoint vector output, rather than a composite `LinearMap`.
* Block concatenation now handles matrices and vectors directly by internal promotion
  to `LinearMap`s. For `[h/v/hc]cat` it suffices to have a `LinearMap` object anywhere
  in the list of arguments. For the block-diagonal concatenation via
  `SparseArrays.blockdiag`, a `LinearMap` object has to appear among the first 8 arguments.
  This restriction, however, does not apply to block-diagonal concatenation via
  `Base.cat(As...; dims=(1,2))`.
* Introduction of more expressive and visually appealing `show` methods, replacing
  the fallback to the generic `show`.

## What's new in v2.7

* Potential reduction of memory allocations in multiplication of
  `LinearCombination`s, `BlockMap`s, and real- or complex-scaled `LinearMap`s.
  For the latter, a new internal type `ScaledMap` has been introduced.
* Multiplication code for `CompositeMap`s has been refactored to facilitate to
  provide memory for storage of intermediate results by directly calling helper
  functions.

## What's new in v2.6

* New feature: "lazy" Kronecker product, Kronecker sums, and powers thereof
  for `LinearMap`s. `AbstractMatrix` objects are promoted to `LinearMap`s if
  one of the first 8 Kronecker factors is a `LinearMap` object.
* Compatibility with the generic multiply-and-add interface (a.k.a. 5-arg
  `mul!`) introduced in julia v1.3

## What's new in v2.5

* New feature: concatenation of `LinearMap`s objects with `UniformScaling`s,
  consistent with (h-, v-, and hc-)concatenation of matrices. Note, matrices
  `A` must be wrapped as `LinearMap(A)`, `UniformScaling`s are promoted to
  `LinearMap`s automatically.

## What's new in v2.4

* Support restricted to Julia v1.0+.

## What's new in v2.3

* Fully Julia v0.7/v1.0/v1.1 compatible.
* Full support of noncommutative number types such as quaternions.

## What's new in v2.2

* Fully Julia v0.7/v1.0 compatible.
* A `convert(SparseMatrixCSC, A::LinearMap)` function, that calls the `sparse`
  matrix generating function.

## What's new in v2.1

* Fully Julia v0.7 compatible; dropped compatibility for previous versions of
  Julia from LinearMaps.jl v2.0.0 on.
* A 5-argument version for `mul!(y, A::LinearMap, x, α=1, β=0)`, which
  computes `y := α * A * x + β * y` and implements the usual 3-argument
  `mul!(y, A, x)` for the default `α` and `β`.
* Synonymous `convert(Matrix, A::LinearMap)` and `convert(Array, A::LinearMap)`
  functions, that call the `Matrix` constructor and return the matrix
  representation of `A`.
* Multiplication with matrices, interpreted as a block row vector of vectors:
  * `mul!(Y::AbstractArray, A::LinearMap, X::AbstractArray, α=1, β=0)`:
    applies `A` to each column of `X` and stores the result in-place in the
    corresponding column of `Y`;
  * for the out-of-place multiplication, the approach is to compute
    `convert(Matrix, A * X)`; this is equivalent to applying `A` to each
    column of `X`. In generic code which handles both `A::AbstractMatrix` and
    `A::LinearMap`, the additional call to `convert` is a noop when `A` is a
    matrix.
* Full compatibility with [Arpack.jl](https://github.com/JuliaLinearAlgebra/Arpack.jl)'s
  `eigs` and `svds`; previously only `eigs` was working. For more, nicely
  collaborating packages see the [Example](#example) section.
