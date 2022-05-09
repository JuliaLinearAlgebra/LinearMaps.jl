module LinearMaps

export LinearMap
export ⊗, kronsum, ⊕
export FillMap

using LinearAlgebra
import LinearAlgebra: mul!
using SparseArrays

import Statistics: mean

using Base: require_one_based_indexing

abstract type LinearMap{T} end

const MapOrVecOrMat{T} = Union{LinearMap{T}, AbstractVecOrMat{T}}
const MapOrMatrix{T} = Union{LinearMap{T}, AbstractMatrix{T}}
const RealOrComplex = Union{Real, Complex}

const LinearMapTuple = Tuple{Vararg{LinearMap}}
const LinearMapVector = AbstractVector{<:LinearMap}
const LinearMapTupleOrVector = Union{LinearMapTuple,LinearMapVector}

Base.eltype(::LinearMap{T}) where {T} = T

# conversion to LinearMap
Base.convert(::Type{LinearMap}, A::LinearMap) = A
Base.convert(::Type{LinearMap}, A::AbstractVecOrMat) = LinearMap(A)

convert_to_lmaps() = ()
convert_to_lmaps(A) = (convert(LinearMap, A),)
@inline convert_to_lmaps(A, B, Cs...) =
    (convert(LinearMap, A), convert(LinearMap, B), convert_to_lmaps(Cs...)...)

abstract type MulStyle end

struct FiveArg <: MulStyle end
struct ThreeArg <: MulStyle end

MulStyle(::FiveArg, ::FiveArg) = FiveArg()
MulStyle(::ThreeArg, ::FiveArg) = ThreeArg()
MulStyle(::FiveArg, ::ThreeArg) = ThreeArg()
MulStyle(::ThreeArg, ::ThreeArg) = ThreeArg()
MulStyle(::LinearMap) = ThreeArg() # default
MulStyle(::AbstractVecOrMat) = FiveArg()
MulStyle(A::LinearMap, As::LinearMap...) = MulStyle(MulStyle(A), MulStyle(As...))

Base.isreal(A::LinearMap) = eltype(A) <: Real
LinearAlgebra.issymmetric(::LinearMap) = false # default assumptions
LinearAlgebra.ishermitian(A::LinearMap{<:Real}) = issymmetric(A)
LinearAlgebra.ishermitian(::LinearMap) = false # default assumptions
LinearAlgebra.isposdef(::LinearMap) = false # default assumptions

Base.ndims(::LinearMap) = 2
Base.size(A::LinearMap, n) =
    (n == 1 || n == 2 ? size(A)[n] : error("LinearMap objects have only 2 dimensions"))
Base.axes(A::LinearMap, n::Integer) =
    (n == 1 || n == 2 ? axes(A)[n] : error("LinearMap objects have only 2 dimensions"))
Base.length(A::LinearMap) = prod(size(A))

# check dimension consistency for multiplication A*B
_iscompatible((A, B)) = size(A, 2) == size(B, 1)
function check_dim_mul(A, B)
    _iscompatible((A, B)) ||
        throw(DimensionMismatch("second dimension of left factor, $(size(A, 2)), " *
            "does not match first dimension of right factor, $(size(B, 1))"))
    return nothing
end
# check dimension consistency for multiplication C = A*B
function check_dim_mul(C, A, B)
    mA, nA = size(A) # A always has two dimensions
    mB, nB = size(B, 1), size(B, 2)
    (mB == nA && size(C, 1) == mA && size(C, 2) == nB) ||
        throw(DimensionMismatch("A has size ($mA,$nA), B has size ($mB,$nB), C has size $(size(C))"))
    return nothing
end

_front(As::Tuple) = Base.front(As)
_front(As::AbstractVector) = @inbounds @views As[1:end-1]
_tail(As::Tuple) = Base.tail(As)
_tail(As::AbstractVector) = @inbounds @views As[2:end]

_combine(A::LinearMap, B::LinearMap) = tuple(A, B)
_combine(A::LinearMap, Bs::LinearMapTuple) = tuple(A, Bs...)
_combine(As::LinearMapTuple, B::LinearMap) = tuple(As..., B)
_combine(As::LinearMapTuple, Bs::LinearMapTuple) = tuple(As..., Bs...)
_combine(A::LinearMap, Bs::LinearMapVector) = Base.vect(A, Bs...)
_combine(As::LinearMapVector, B::LinearMap) = Base.vect(As..., B)
_combine(As::LinearMapVector, Bs::LinearMapTuple) = Base.vect(As..., Bs...)
_combine(As::LinearMapTuple, Bs::LinearMapVector) = Base.vect(As..., Bs...)
_combine(As::LinearMapVector, Bs::LinearMapVector) = Base.vect(As..., Bs...)

# The (internal) multiplication logic is as follows:
#  - `*(A, x)` calls `mul!(y, A, x)` for appropriately-sized y
#  - `mul!` checks consistency of the sizes, and calls `_unsafe_mul!`,
#    which does not check sizes, but potentially one-based indexing if necessary
#  - by default, `_unsafe_mul!` is redirected back to `mul!`
#  - custom map types only need to implement 3-arg (vector) `mul!`, and
#    everything else (5-arg multiplication, application to matrices,
#    conversion to matrices) will just work

"""
    *(A::LinearMap, x::AbstractVector)::AbstractVector

Compute the action of the linear map `A` on the vector `x`.

!!! compat "Julia 1.3"
    In Julia versions v1.3 and above, objects `L` of any subtype of `LinearMap`
    are callable in the sense that `L(x) = L*x` for `x::AbstractVector`.

## Examples
```jldoctest; setup=(using LinearAlgebra, LinearMaps)
julia> A=LinearMap([1.0 2.0; 3.0 4.0]); x=[1.0, 1.0];

julia> A*x
2-element Array{Float64,1}:
 3.0
 7.0

julia> A(x)
2-element Array{Float64,1}:
 3.0
 7.0
```
"""
function Base.:(*)(A::LinearMap, x::AbstractVector)
    check_dim_mul(A, x)
    T = promote_type(eltype(A), eltype(x))
    y = similar(x, T, axes(A)[1])
    return mul!(y, A, x)
end

(L::LinearMap)(x::AbstractVector) = L*x

"""
    mul!(Y::AbstractVecOrMat, A::LinearMap, B::AbstractVector) -> Y
    mul!(Y::AbstractMatrix, A::LinearMap, B::AbstractMatrix) -> Y

Calculates the action of the linear map `A` on the vector or matrix `B` and stores the
result in `Y`, overwriting the existing value of `Y`. Note that `Y` must not be aliased
with either `A` or `B`.

## Examples
```jldoctest; setup=(using LinearAlgebra, LinearMaps)
julia> A=LinearMap([1.0 2.0; 3.0 4.0]); B=[1.0, 1.0]; Y = similar(B); mul!(Y, A, B);

julia> Y
2-element Array{Float64,1}:
 3.0
 7.0

julia> A=LinearMap([1.0 2.0; 3.0 4.0]); B=[1.0 1.0; 1.0 1.0]; Y = similar(B); mul!(Y, A, B);

julia> Y
2×2 Array{Float64,2}:
 3.0  3.0
 7.0  7.0
```
"""
function mul!(y::AbstractVecOrMat, A::LinearMap, x::AbstractVector)
    check_dim_mul(y, A, x)
    return _unsafe_mul!(y, A, x)
end

function mul!(y::AbstractVecOrMat, A::LinearMap, s::Number)
    size(y) == size(A) ||     
        throw(
            DimensionMismatch("y has size $(size(y)), A has size $(size(A)), s has size $(size(s))"))
    return _unsafe_mul!(y, A, s)
end

"""
    mul!(C::AbstractVecOrMat, A::LinearMap, B::AbstractVector, α, β) -> C
    mul!(C::AbstractMatrix, A::LinearMap, B::AbstractMatrix, α, β) -> C

Combined inplace multiply-add ``A B α + C β``. The result is stored in `C` by overwriting
it. Note that `C` must not be aliased with either `A` or `B`.

## Examples
```jldoctest; setup=(using LinearAlgebra, LinearMaps)
julia> A=LinearMap([1.0 2.0; 3.0 4.0]); B=[1.0, 1.0]; C=[1.0, 3.0];

julia> mul!(C, A, B, 100.0, 10.0) === C
true

julia> C
2-element Array{Float64,1}:
 310.0
 730.0

julia> A=LinearMap([1.0 2.0; 3.0 4.0]); B=[1.0 1.0; 1.0 1.0]; C=[1.0 2.0; 3.0 4.0];

julia> mul!(C, A, B, 100.0, 10.0) === C
true

julia> C
2×2 Array{Float64,2}:
 310.0  320.0
 730.0  740.0
```
"""
function mul!(y::AbstractVecOrMat, A::LinearMap, x::AbstractVector, α::Number, β::Number)
    check_dim_mul(y, A, x)
    return _unsafe_mul!(y, A, x, α, β)
end

function mul!(y::AbstractVecOrMat, A::LinearMap, s::Number, α::Number, β::Number)
    size(y) == size(A) ||     
        throw(
            DimensionMismatch("y has size $(size(y)), A has size $(size(A)), s has size $(size(s))"))
    return _unsafe_mul!(y, A, s, α, β)
end

function _generic_mapvec_mul!(y, A, x, α, β)
    # this function needs to call mul! for, e.g.,  AdjointMap{...,<:CustomMap}
    if isone(α)
        iszero(β) && return mul!(y, A, x)
        z = A * x
        if isone(β)
            y .+= z
        else
            y .= y.*β .+ z
        end
        return y
    elseif iszero(α)
        iszero(β) && return fill!(y, zero(eltype(y)))
        isone(β) && return y
        # β != 0, 1
        return rmul!(y, β)
    else # α != 0, 1
        iszero(β) && return rmul!(mul!(y, A, x), α)
        z = A * x
        if isone(β)
            y .+= z .* α
        else
            y .= y .* β .+ z .* α
        end
        return y
    end
end

# the following is of interest in, e.g., subspace-iteration methods
function mul!(Y::AbstractMatrix, A::LinearMap, X::AbstractMatrix)
    check_dim_mul(Y, A, X)
    return _unsafe_mul!(Y, A, X)
end
function mul!(Y::AbstractMatrix, A::LinearMap, X::AbstractMatrix, α::Number, β::Number)
    check_dim_mul(Y, A, X)
    return _unsafe_mul!(Y, A, X, α, β)
end

function _generic_mapmat_mul!(Y, A, X, α=true, β=false)
    for (Xi, Yi) in zip(eachcol(X), eachcol(Y))
        mul!(Yi, A, Xi, α, β)
    end
    return Y
end

function _generic_mapnum_mul!(Y, A, s, α=true, β=false)
    T = typeof(s)
    ax2 = axes(A,2)
    xi = zeros(T,ax2)
    @inbounds for (i,Yi) in zip(ax2, eachcol(Y))
        xi[i] = s
        mul!(Yi, A, xi, α, β)
        xi[i] = zero(T)
    end
    return Y
end

_unsafe_mul!(y, A::MapOrVecOrMat, x) = mul!(y, A, x)
_unsafe_mul!(y, A::AbstractVecOrMat, x, α, β) = mul!(y, A, x, α, β)
function _unsafe_mul!(y::AbstractVecOrMat, A::LinearMap, x::AbstractVector, α, β)
    return _generic_mapvec_mul!(y, A, x, α, β)
end
function _unsafe_mul!(y::AbstractMatrix, A::LinearMap, x::AbstractMatrix)
    return _generic_mapmat_mul!(y, A, x)
end
function _unsafe_mul!(y::AbstractMatrix, A::LinearMap, x::AbstractMatrix, α, β)
    return _generic_mapmat_mul!(y, A, x, α, β)
end
function _unsafe_mul!(Y::AbstractMatrix, A::LinearMap, s::Number, α=true, β=false)
    return _generic_mapnum_mul!(Y, A, s, α, β)
end

include("left.jl") # left multiplication by a transpose or adjoint vector
include("transpose.jl") # transposing linear maps
include("wrappedmap.jl") # wrap a matrix of linear map in a new type, thereby allowing to alter its properties
include("uniformscalingmap.jl") # the uniform scaling map, to be able to make linear combinations of LinearMap objects and multiples of I
include("linearcombination.jl") # defining linear combinations of linear maps
include("scaledmap.jl") # multiply by a (real or complex) scalar
include("composition.jl") # composition of linear maps
include("functionmap.jl") # using a function as linear map
include("blockmap.jl") # block linear maps
include("kronecker.jl") # Kronecker product of linear maps
include("fillmap.jl") # linear maps representing constantly filled matrices
include("conversion.jl") # conversion of linear maps to matrices
include("show.jl") # show methods for LinearMap objects

"""
    LinearMap(A::LinearMap; kwargs...)::WrappedMap
    LinearMap(A::AbstractVecOrMat; kwargs...)::WrappedMap
    LinearMap(J::UniformScaling, M::Int)::UniformScalingMap
    LinearMap{T=Float64}(f, [fc,], M::Int, N::Int = M; kwargs...)::FunctionMap

Construct a linear map object, either from an existing `LinearMap` or `AbstractVecOrMat` `A`,
with the purpose of redefining its properties via the keyword arguments `kwargs`;
a `UniformScaling` object `J` with specified (square) dimension `M`; or
from a function or callable object `f`. In the latter case, one also needs to specify
the size of the equivalent matrix representation `(M, N)`, i.e., for functions `f` acting
on length `N` vectors and producing length `M` vectors (with default value `N=M`).
Preferably, also the `eltype` `T` of the corresponding matrix representation needs to be
specified, i.e., whether the action of `f` on a vector will be similar to, e.g., multiplying
by numbers of type `T`. If not specified, the devault value `T=Float64` will be assumed.
Optionally, a corresponding function `fc` can be specified that implements the adjoint
(=transpose in the real case) of `f`.

The keyword arguments and their default values for the function-based constructor are:
*   `issymmetric::Bool = false` : whether `A` or `f` act as a symmetric matrix
*   `ishermitian::Bool = issymmetric & T<:Real` : whether `A` or `f` act as a Hermitian
    matrix
*   `isposdef::Bool = false` : whether `A` or `f` act as a positive definite matrix.
For existing linear maps or matrices `A`, the default values will be taken by calling
internal functions `_issymmetric`, `_ishermitian` and `_isposdef` on the existing object `A`.
These in turn dispatch to (overloads of) `LinearAlgebra`'s `issymmetric`, `ishermitian`,
and `isposdef` methods whenever these checks are expected to be computationally cheap or even
known at compile time as for certain structured matrices, but return `false` for generic
`AbstractMatrix` types.

For the function-based constructor, there is one more keyword argument:
*   `ismutating::Bool` : flags whether the function acts as a mutating matrix multiplication
    `f(y,x)` where the result vector `y` is the first argument (in case of `true`),
    or as a normal matrix multiplication that is called as `y=f(x)` (in case of `false`).
    The default value is guessed by looking at the number of arguments of the first
    occurrence of `f` in the method table.
"""
LinearMap(A::MapOrVecOrMat; kwargs...) = WrappedMap(A; kwargs...)
LinearMap(J::UniformScaling, M::Int) = UniformScalingMap(J.λ, M)
LinearMap(f, M::Int; kwargs...) = LinearMap{Float64}(f, M; kwargs...)
LinearMap(f, M::Int, N::Int; kwargs...) = LinearMap{Float64}(f, M, N; kwargs...)
LinearMap(f, fc, M::Int; kwargs...) = LinearMap{Float64}(f, fc, M; kwargs...)
LinearMap(f, fc, M::Int, N::Int; kwargs...) = LinearMap{Float64}(f, fc, M, N; kwargs...)

LinearMap{T}(A::MapOrMatrix; kwargs...) where {T} = WrappedMap{T}(A; kwargs...)
LinearMap{T}(f, args...; kwargs...) where {T} = FunctionMap{T}(f, args...; kwargs...)

end # module
