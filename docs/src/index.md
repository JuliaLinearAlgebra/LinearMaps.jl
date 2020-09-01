# LinearMaps.jl

*A Julia package for defining and working with linear maps, also known as linear transformations or linear operators acting on vectors. The only requirement for a LinearMap is that it can act on a vector (by multiplication) efficiently.*

## Installation

`LinearMaps.jl` is a registered package and can be installed via

    pkg> add LinearMaps

in package mode, to be entered by typing `]` in the Julia REPL.

## Examples

Let

    A = LinearMap(rand(10, 10))
    B = LinearMap(cumsum, reverse∘cumsum∘reverse, 10)
    
be a matrix- and function-based linear map, respectively. Then the following code just works,
indistinguishably from the case when `A` and `B` are both `AbstractMatrix`-typed objects.

```
3.0A + 2B
A*B'
[A B; B A]
kron(A, B)
```

The `LinearMap` type and corresponding methods combine well with the following packages:
* [Arpack.jl](https://github.com/JuliaLinearAlgebra/Arpack.jl): iterative eigensolver
  `eigs` and SVD `svds`;
* [IterativeSolvers.jl](https://github.com/JuliaMath/IterativeSolvers.jl): iterative
  solvers, eigensolvers, and SVD;
* [KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl): Krylov-based algorithms for linear problems, singular value and eigenvalue problems
* [TSVD.jl](https://github.com/andreasnoack/TSVD.jl): truncated SVD `tsvd`.

```julia
using LinearMaps
import Arpack, IterativeSolvers, KrylovKit, TSVD

# Example 1, 1-dimensional Laplacian with periodic boundary conditions
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

# Example 2, 1-dimensional Laplacian
A = LinearMap(100; issymmetric=true, ismutating=true) do C, B
    C[1] = -2B[1] + B[2]
    for i in 2:length(B)-1
        C[i] = B[i-1] - 2B[i] + B[i+1]
    end
    C[end] = B[end-1] - 2B[end]
    return C
end

Arpack.eigs(-A; nev=3, which=:SR)

# Example 3, 2-dimensional Laplacian
Δ = kronsum(A, A)

Arpack.eigs(Δ; nev=3, which=:LR)
KrylovKit.eigsolve(x -> Δ*x, size(Δ, 1), 3, :LR)
```

## Philosophy

Several iterative linear algebra methods such as linear solvers or eigensolvers
only require an efficient evaluation of the matrix-vector product, where the
concept of a matrix can be formalized / generalized to a linear map (or linear
operator in the special case of a square matrix).

The LinearMaps package provides the following functionality:

1.  A `LinearMap` type that shares with the `AbstractMatrix` type that it
    responds to the functions `size`, `eltype`, `isreal`, `issymmetric`,
    `ishermitian` and `isposdef`, `transpose` and `adjoint` and multiplication
    with a vector using both `*` or the in-place version `mul!`. Linear algebra
    functions that use duck-typing for their arguments can handle `LinearMap`
    objects similar to `AbstractMatrix` objects, provided that they can be
    written using the above methods. Unlike `AbstractMatrix` types, `LinearMap`
    objects cannot be indexed, neither using `getindex` or `setindex!`.

2.  A single function `LinearMap` that acts as a general purpose
    constructor (though it is only an abstract type) and allows to construct
    linear map objects from functions, or to wrap objects of type
    `AbstractMatrix` or `LinearMap`. The latter functionality is useful to
    (re)define the properties (`isreal`, `issymmetric`, `ishermitian`,
    `isposdef`) of the existing matrix or linear map.

3.  A framework for combining objects of type `LinearMap` and of type
    `AbstractMatrix` using linear combinations, transposition, composition,
    concatenation and Kronecker product/sums,
    where the linear map resulting from these operations is never explicitly
    evaluated but only its matrix-vector product is defined (i.e. lazy
    evaluation). The matrix-vector product is written to minimize memory
    allocation by using a minimal number of temporary vectors. There is full
    support for the in-place version `mul!`, which should be preferred for
    higher efficiency in critical algorithms. In addition, it tries to recognize
    the properties of combinations of linear maps. In particular, compositions
    such as `A'*A` for arbitrary `A` or even `A'*B*C*B'*A` with arbitrary `A`
    and `B` and positive definite `C` are recognized as being positive definite
    and hermitian. In case a certain property of the resulting `LinearMap`
    object is not correctly inferred, the `LinearMap` method can be called to
    redefine the properties.
