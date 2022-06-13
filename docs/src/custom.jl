# # Defining custom `LinearMap` types

# In this section, we want to demonstrate on a simple, actually built-in, linear map type
# how to define custom `LinearMap` subtypes. First of all, `LinearMap{T}` is an extendable
# abstract type, where `T` denotes the `eltype`.

# ## Basics

# As an example, we want to define a map type whose objects correspond to lazy analogues
# of `fill`ed matrices. Naturally, we need to store the filled value `λ` and the `size`
# of the linear map.

using LinearMaps, LinearAlgebra

struct MyFillMap{T} <: LinearMaps.LinearMap{T}
    λ::T
    size::Dims{2}
    function MyFillMap(λ::T, dims::Dims{2}) where {T}
        all(≥(0), dims) || throw(ArgumentError("dims of MyFillMap must be non-negative"))
        promote_type(T, typeof(λ)) == T || throw(InexactError())
        return new{T}(λ, dims)
    end
end

# By default, for any `A::MyFillMap{T}`, `eltype(A)` returns `T`. Upon application to a
# vector `x` and/or interaction with other `LinearMap` objects, we need to check consistent
# sizes.

Base.size(A::MyFillMap) = A.size

# By a couple of defaults provided for all subtypes of `LinearMap`, we only need to define
# a `LinearMaps._unsafe_mul!` method to have a minimal, operational type. The (internal)
# function `_unsafe_mul!` is called by `LinearAlgebra.mul!`, constructors, and conversions
# and only needs to be concerned with the bare computing kernel. Dimension checking is done
# on the level of `mul!` etc. Factoring out dimension checking is done to minimise overhead
# caused by repetitive checking.

function LinearMaps._unsafe_mul!(y::AbstractVecOrMat, A::MyFillMap, x::AbstractVector)
    return fill!(y, iszero(A.λ) ? zero(eltype(y)) : A.λ*sum(x))
end

# Again, due to generic fallbacks the following now "just work":

# * out-of-place multiplication `A*x`,
# * in-place multiplication with vectors `mul!(y, A, x)`,
# * in-place multiply-and-add with vectors `mul!(y, A, x, α, β)`,
# * in-place multiplication and multiply-and-add with matrices `mul!(Y, A, X, α, β)`,
# * conversion to a (sparse) matrix `Matrix(A)` and `sparse(A)`.

A = MyFillMap(5.0, (3, 3)); x = ones(3); sum(x)

#-

A * x

#-

mul!(zeros(3), A, x)

#-

mul!(ones(3), A, x, 2, 2)

#-

mul!(ones(3,3), A, reshape(collect(1:9), 3, 3), 2, 2)

# ## Multiply-and-add and the `MulStyle` trait

# While the above function calls work out of the box due to generic fallbacks, the latter
# may be suboptimally implemented for your custom map type. Let's see some benchmarks.

using BenchmarkTools

@benchmark mul!($(zeros(3)), $A, $x)

#-

@benchmark mul!($(zeros(3)), $A, $x, $(rand()), $(rand()))

# The second benchmark indicates the allocation of an intermediate vector `z`
# which stores the result of `A*x` before it gets scaled and added to (the scaled)
# `y = zeros(3)`. For that reason, it is beneficial to provide a custom "5-arg `mul!`"
# if you can avoid the allocation of an intermediate vector. To indicate that there
# exists an allocation-free implementation, you should set the `MulStyle` trait,
# whose default is `ThreeArg()`.

LinearMaps.MulStyle(A::MyFillMap) = FiveArg()

function LinearMaps._unsafe_mul!(
    y::AbstractVecOrMat,
    A::MyFillMap,
    x::AbstractVector,
    α::Number,
    β::Number
)
    if iszero(α)
        !isone(β) && rmul!(y, β)
        return y
    else
        temp = A.λ * sum(x) * α
        if iszero(β)
            y .= temp
        elseif isone(β)
            y .+= temp
        else
            y .= y .* β .+ temp
        end
    end
    return y
end

# With this function at hand, let's redo the benchmark.

@benchmark mul!($(zeros(3)), $A, $x, $(rand()), $(rand()))

# There you go, the allocation is gone and the computation time is significantly reduced.

# ## Adjoints and transposes

# Generically, taking the transpose (or the adjoint) of a (real, resp.) map wraps the
# linear map by a `TransposeMap`, taking the adjoint of a complex map wraps it by an
# `AdjointMap`.

typeof(A')

# Not surprisingly, without further definitions, multiplying `A'` by `x` yields an error.

try A'x catch e println(e) end

# If the operator is symmetric or Hermitian, the transpose and the adjoint, respectively,
# of the linear map `A` is given by `A` itself. So let us define corresponding checks.

LinearAlgebra.issymmetric(A::MyFillMap) = A.size[1] == A.size[2]
LinearAlgebra.ishermitian(A::MyFillMap) = isreal(A.λ) && A.size[1] == A.size[2]
LinearAlgebra.isposdef(A::MyFillMap) = (size(A, 1) == size(A, 2) == 1 && isposdef(A.λ))
Base.:(==)(A::MyFillMap, B::MyFillMap) = A.λ == B.λ && A.size == B.size

# These are used, for instance, in checking symmetry or positive definiteness of
# higher-order `LinearMap`s, like products or linear combinations of linear maps, or signal
# to iterative eigenproblem solvers that real eigenvalues are to be computed.
# Without these definitions, the first three functions would return `false` (by default),
# and the last one would fall back to `===`.

# With this at hand, we note that `A` above is symmetric, and we can compute

transpose(A)*x

# This, however, does not work for nonsquare maps

try MyFillMap(5.0, (3, 4))' * ones(3) catch e println(e) end

# which require explicit adjoint/transpose handling, for which there exist two *distinct*
# paths.

# ### Path 1: Generic, non-invariant `LinearMap` subtypes

# The first option is to write `LinearMaps._unsafe_mul!` methods for the corresponding
# wrapped map types; for instance,

function LinearMaps._unsafe_mul!(
    y::AbstractVecOrMat,
    transA::LinearMaps.TransposeMap{<:Any,<:MyFillMap},
    x::AbstractVector
)
    λ = transA.lmap.λ
    return fill!(y, iszero(λ) ? zero(eltype(y)) : transpose(λ)*sum(x))
end

# Now, the adjoint multiplication works.

MyFillMap(5.0, (3, 4))' * ones(3)

# If you have set the `MulStyle` trait to `FiveArg()`, you should provide a corresponding
# 5-arg `mul!` method for `LinearMaps.TransposeMap{<:Any,<:MyFillMap}` and
# `LinearMaps.AdjointMap{<:Any,<:MyFillMap}`.

# ### Path 2: Invariant `LinearMap` subtypes

# Before we start, let us delete the previously defined method to make sure we use the
# following definitions.

Base.delete_method(
    first(methods(
        LinearMaps._unsafe_mul!,
        (AbstractVecOrMat, LinearMaps.TransposeMap{<:Any,<:MyFillMap}, AbstractVector))
    )
)

# The seconnd option is when your class of linear maps that are modelled by your custom
# `LinearMap` subtype are invariant under taking adjoints and transposes.

LinearAlgebra.adjoint(A::MyFillMap) = MyFillMap(adjoint(A.λ), reverse(A.size))
LinearAlgebra.transpose(A::MyFillMap) = MyFillMap(transpose(A.λ), reverse(A.size))

# With such invariant definitions, i.e., the adjoint/transpose of a `MyFillMap` is again
# a `MyFillMap`, no further method definitions are required, and the entire functionality
# listed above just works for adjoints/transposes of your custom map type.

mul!(ones(3), A', x, 2, 2)

#-

MyFillMap(5.0, (3, 4))' * ones(3)

# Now that we have defined the action of adjoints/transposes, the
# following right action on vectors is automatically defined:

ones(3)' * MyFillMap(5.0, (3, 4))

# and `transpose(x) * A` correspondingly, as well as in-place multiplication

mul!(similar(x)', x', A)

# and `mul!(transpose(y), transpose(x), A)`.

# ## Application to matrices

# By default, applying a `LinearMap` `A` to a matrix `X` via `A*X` does
# *not* apply `A` to each column of `X` viewed as a vector, but interprets
# `X` as a linear map, wraps it as such and returns `(A*X)::CompositeMap`.
# Calling the in-place multiplication function `mul!(Y, A, X)` for matrices,
# however, does compute the columnwise action of `A` on `X` and stores the
# result in `Y`. In case there is a more efficient implementation for the
# matrix application, you can provide `mul!` methods with signature
# `mul!(Y::AbstractMatrix, A::MyFillMap, X::AbstractMatrix)`, and, depending
# on the chosen path to handle adjoints/transposes, corresponding methods
# for wrapped maps of type `AdjointMap` or `TransposeMap`, plus potentially
# corresponding 5-arg `mul!` methods. This may seem like a lot of methods to
# be implemented, but note that adding such methods is only necessary/recommended
# for performance.

# ## Computing a matrix representation

# In some cases, it might be necessary to compute a matrix representation of a `LinearMap`.
# This is essentially done via the
# `[LinearMaps._unsafe_mul!(::Matrix,::LinearMap,::Number)]`(@ref) method, for which a
# generic fallback exists: it applies the `LinearMap` successively to the standard unit
# vectors.

F = MyFillMap(5, (100,100))
M = Matrix{eltype(F)}(undef, size(F))
@benchmark Matrix($F)

#-

@benchmark LinearMaps._unsafe_mul!($(Matrix{Int}(undef, (100,100))), $(MyFillMap(5, (100,100))), true)

# If a more performant implementation exists, it is recommended to overwrite this method,
# for instance (as before, size checks need not be included here since they are handled by
# the corresponding `LinearAlgebra.mul!` method):

LinearMaps._unsafe_mul!(M::AbstractMatrix, A::MyFillMap, s::Number) = fill!(M, A.λ*s)
@benchmark Matrix($F)

#-

@benchmark LinearMaps._unsafe_mul!($(Matrix{Int}(undef, (100,100))), $(MyFillMap(5, (100,100))), true)

# As one can see, the above runtimes are dominated by the allocation of the output matrix,
# but still overwriting the multiplication kernel yields a speed-up of about factor 3 for
# the matrix filling part.

# ## Slicing

# As usual, generic fallbacks for `LinearMap` slicing exist and are handled by the following
# method hierarchy, where at least one of `I` and `J` has to be a `Colon`:
#
#     Base.getindex(::LinearMap, I, J)
#     -> LinearMaps._getindex(::LinearMap, I, J)
#
# The method `Base.getindex` checks the validity of the the requested indices and calls
# `LinearMaps._getindex`, which should be overloaded for custom `LinearMap`s subtypes.
# For instance:

@benchmark F[1,:]

#-

LinearMaps._getindex(A::MyFillMap, ::Integer, J::Base.Slice) = fill(A.λ, axes(J))
@benchmark F[1,:]

# Note that in `Base.getindex` `Colon`s are converted to `Base.Slice` via
# `Base.to_indices`, thus the dispatch must be on `Base.Slice` rather than on `Colon`.
