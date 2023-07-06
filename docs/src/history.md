# Version history

## What's new in v3.11

* The `tr` function from `LinearAlgebra.jl` is now overloaded both for generic `LinearMap`
  types and specialized for most provided `LinearMap` types. In the generic case, this is
  computationally as expensive as computing the whole matrix representation, though the
  latter is, of course, not stored.

## What's new in v3.10

* A new `MulStyle` trait called `TwoArg` has been added. It should be used for `LinearMap`s
  that do not admit a mutating multiplication à la (3-arg or 5-arg) `mul!`, but only
  out-of-place multiplication à la `A * x`. Products (aka `CompositeMap`s) and sums (aka
  `LinearCombination`s) of `TwoArg`-`LinearMap`s now have memory-optimized multiplication
  kernels. For instance, `A*B*C*x` for three `TwoArg`-`LinearMap`s `A`, `B` and `C` now
  allocates only `y = C*x`, `z = B*y` and the result of `A*z`.
* The construction of function-based `LinearMap`s, typed `FunctionMap`, has been rearranged.
  Additionally to the convenience constructor `LinearMap{T=Float64}(f, [fc,] M, N=M; kwargs...)`,
  the newly exported constructor `FunctionMap{T,iip}(f, [fc], M, N; kwargs...)` is readily
  available. Here, `iip` is either `true` or `false`, and encodes whether `f` (and `fc` if
  present) are mutating functions. In the convenience constructor, this is determined via the
  `Bool` keyword argument `ismutating` and may not be fully inferred.

## What's new in v3.9

* The application of `LinearMap`s to vectors operation, i.e., `(A,x) -> A*x = A(x)`, is now
  differentiable w.r.t. to the input `x` for integration with machine learning frameworks
  such as [`Flux.jl`](https://fluxml.ai/Flux.jl/stable/). The reverse differentiation rule
  makes `A::LinearMap` usable as a static, i.e., non-trainable, layer in a network, and
  requires the adjoint `A'` of `A` to be defined.
* New map types called `KhatriRaoMap` and `FaceSplittingMap` are introduced. These
  correspond to lazy representations of the [column-wise Kronecker product](https://en.wikipedia.org/wiki/Khatri%E2%80%93Rao_product#Column-wise_Kronecker_product)
  and the [row-wise Kronecker product](https://en.wikipedia.org/wiki/Khatri%E2%80%93Rao_product#Face-splitting_product)
  (or "transposed Khatri-Rao product"), respectively. They can be constructed from two
  matrices `A` and `B` via `khatrirao(A, B)` and `facesplitting(A, B)`, respectively.
  The first is particularly efficient as it makes use of the vec-trick for Kronecker
  products and computes `y = khatrirao(A, B) * x` for a vector `x` as
  `y = vec(B * Diagonal(x) * transpose(A))`. As such, the Khatri-Rao product can actually
  be built for general `LinearMap`s, including function-based types. Even for moderate
  sizes of 5 or more columns, this map-vector product is faster than first creating the
  explicit Khatri-Rao product in memory and then multiplying with the vector; not to
  mention the memory savings. Unfortunately, similar efficiency cannot be achieved for the
  face-splitting product.

## What's new in v3.8

* A new map called [`InverseMap`](@ref) is introduced. Letting an `InverseMap` act on a
  vector is equivalent to solving the linear system, i.e. `InverseMap(A) * b` is the same as
  `A \ b`. The default solver is `ldiv!`, but can be specified with the `solver` keyword
  argument to the constructor (see the docstring for details). Note that `A` must be
  compatible with the solver: `A` can, for example, be a factorization, or another
  `LinearMap` in combination with an iterative solver.
* New constructors for lazy representations of Kronecker products ([`squarekron`](@ref))
  and sums ([`sumkronsum`](@ref)) for _square_ factors and summands, respectively, are
  introduced. They target cases with 3 or more factors/summands, and benchmarking intended
  use cases for comparison with `KroneckerMap` (constructed via [`Base.kron`](@ref)) and
  `KroneckerSumMap` (constructed via [`kronsum`](@ref)) is recommended.

## What's new in v3.7

* `mul!(M::AbstractMatrix, A::LinearMap, s::Number, a, b)` methods are provided, mimicking
  similar methods in `Base.LinearAlgebra`. This version allows for the memory efficient
  implementation of in-place addition and conversion of a `LinearMap` to `Matrix`.
  Efficient specialisations for `WrappedMap`, `ScaledMap`, and `LinearCombination` are
  provided. If users supply the corresponding `_unsafe_mul!` method for their custom maps,
  conversion, construction, and inplace addition will benefit from this supplied efficient
  implementation. If no specialisation is supplied, a generic fallback is used that is based
  on feeding the canonical basis of unit vectors to the linear map.
* A new map type called `EmbeddedMap` is introduced. It is a wrapper of a "small" `LinearMap`
  (or a suitably converted `AbstractVecOrMat`) embedded into a "larger" zero map. Hence,
  the "small" map acts only on a subset of the coordinates and maps to another subset of
  the coordinates of the "large" map. The large map `L` can therefore be imagined as the
  composition of a sampling/projection map `P`, of the small map `A`, and of an embedding
  map `E`: `L = E ⋅ A ⋅ P`. It is implemented, however, by acting on a view of the vector
  `x` and storing the result into a view of the result vector `y`. Such maps can be
  constructed by the new methods:
  * `LinearMap(A::MapOrVecOrMat, dims::Dims{2}, index::NTuple{2, AbstractVector{Int}})`,
    where `dims` is the dimensions of the "large" map and index is a tuple of the `x`- and
    `y`-indices that interact with `A`, respectively;
  * `LinearMap(A::MapOrVecOrMat, dims::Dims{2}; offset::Dims{2})`, where the keyword
    argument `offset` determines the dimension of a virtual upper-left zero block, to which
    `A` gets (virtually) diagonally appended.
* An often requested new feature has been added: slicing (i.e., non-scalar indexing) any
  `LinearMap` object via `Base.getindex` overloads. Note, however, that only rather
  efficient complete slicing operations are implemented: `A[:,j]`, `A[:,J]`, and `A[:,:]`,
  where `j::Integer` and `J` is either of type `AbstractVector{<:Integer>}` or an
  `AbstractVector{Bool}` of appropriate length ("logical slicing"). Partial slicing
  operations such as `A[I,j]` and `A[I,J]` where `I` is as `J` above are disallowed.

  Scalar indexing `A[i::Integer,j::Integer]` as well as other indexing operations that fall
  back on scalar indexing such as logical indexing by some `AbstractMatrix{Bool}`, or
  indexing by vectors of (linear or Cartesian) indices are not supported; as an exception,
  `getindex` calls on wrapped `AbstractVecOrMat`s is forwarded to corresponding `getindex`
  methods from `Base` and therefore allow any type of usual indexing/slicing.
  If scalar indexing is really required, consider using `A[:,j][i]` which is as efficient
  as a reasonable generic implementation for `LinearMap`s can be.

  Furthermore, (predominantly) horizontal slicing operations require the adjoint operation
  of the `LinearMap` type to be defined, or will fail otherwise. Important note:
  `LinearMap` objects are meant to model objects that act on vectors efficiently, and are
  in general *not* backed up by storage-like types like `Array`s. Therefore, slicing of
  `LinearMap`s is potentially slow, and it may require the (repeated) allocation of
  standard unit vectors. As a consequence, generic algorithms relying heavily on indexing
  and/or slicing are likely to run much slower than expected for `AbstractArray`s. To avoid
  repeated indexing operations which may involve redundant computations, it is strongly
  recommended to consider `convert`ing `LinearMap`-typed objects to `Matrix` or
  `SparseMatrixCSC` first, if memory permits.

## What's new in v3.6

* Support for Julia versions below v1.6 has been dropped.
* `Block[Diagonal]Map`, `CompositeMap`, `KroneckerMap` and `LinearCombination` type objects
  can now be backed by a `Vector` of `LinearMap`-type elements. This can be beneficial in
  cases where these higher-order `LinearMap`s are constructed from many maps where a tuple
  backend may get inefficient or impose hard work for the compiler at construction.
  The default behavior, however, does not change, and construction of vector-based
  `LinearMap`s requires usage of the unexported constructors ("expert usage"), except for
  constructions like `sum([A, B, C])` or `prod([A, B, C])` (`== C*B*A`), where `A`, `B` and
  `C` are of some `LinearMap` type.

## What's new in v3.5

* `WrappedMap`, `ScaledMap`, `LinearCombination`, `AdjointMap`, `TransposeMap` and
  `CompositeMap`, instead of using the default `axes(A) = map(oneto, size(A))`, now forward
  calls to `axes` to the underlying wrapped linear map. This allows allocating operations
  such as `*` to determine the appropriate storage and axes type of their outputs.
  For example, linear maps that wrap `BlockArrays` will, upon multiplicative action,
  produce a `BlockArrays.PseudoBlockVector` with block structure inherited from the
  operator's *output* axes `axes(A,1)`.

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
  typed objects. Upon wrapping, they are not implicitly reshaped to matrices. This
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
     `LinearCombination`s or `CompositeMap`s), `At_mul_B!` and `Ac_mul_B!` do
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
