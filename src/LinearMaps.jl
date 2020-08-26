module LinearMaps

export LinearMap
export ⊗, kronsum, ⊕

using LinearAlgebra
using SparseArrays

if VERSION < v"1.2-"
    import Base: has_offset_axes
    require_one_based_indexing(A...) = !has_offset_axes(A...) || throw(ArgumentError("offset arrays are not supported but got an array with index other than 1"))
else
    import Base: require_one_based_indexing
end

abstract type LinearMap{T} end

const MapOrMatrix{T} = Union{LinearMap{T},AbstractMatrix{T}}
const RealOrComplex = Union{Real,Complex}

Base.eltype(::LinearMap{T}) where {T} = T

abstract type MulStyle end

struct FiveArg <: MulStyle end
struct ThreeArg <: MulStyle end

MulStyle(::FiveArg, ::FiveArg) = FiveArg()
MulStyle(::ThreeArg, ::FiveArg) = ThreeArg()
MulStyle(::FiveArg, ::ThreeArg) = ThreeArg()
MulStyle(::ThreeArg, ::ThreeArg) = ThreeArg()
MulStyle(::LinearMap) = ThreeArg() # default
@static if VERSION ≥ v"1.3.0-alpha.115"
    MulStyle(::AbstractMatrix) = FiveArg()
else
    MulStyle(::AbstractMatrix) = ThreeArg()
end
MulStyle(A::LinearMap, As::LinearMap...) = MulStyle(MulStyle(A), MulStyle(As...))

Base.isreal(A::LinearMap) = eltype(A) <: Real
LinearAlgebra.issymmetric(::LinearMap) = false # default assumptions
LinearAlgebra.ishermitian(A::LinearMap{<:Real}) = issymmetric(A)
LinearAlgebra.ishermitian(::LinearMap) = false # default assumptions
LinearAlgebra.isposdef(::LinearMap) = false # default assumptions

Base.ndims(::LinearMap) = 2
Base.size(A::LinearMap, n) = (n==1 || n==2 ? size(A)[n] : error("LinearMap objects have only 2 dimensions"))
Base.length(A::LinearMap) = size(A)[1] * size(A)[2]

# check dimension consistency for multiplication C = A*B
function check_dim_mul(C, A, B)
    # @info "checked vector dimensions" # uncomment for testing
    mA, nA = size(A) # A always has two dimensions
    mB, nB = size(B, 1), size(B, 2)
    (mB == nA) ||
        throw(DimensionMismatch("left factor has dimensions ($mA,$nA), right factor has dimensions ($mB,$nB)"))
    (size(C, 1) != mA || size(C, 2) != nB) &&
        throw(DimensionMismatch("result has dimensions $(size(C)), needs ($mA,$nB)"))
    return nothing
end

# conversion of AbstractMatrix to LinearMap
convert_to_lmaps_(A::AbstractMatrix) = LinearMap(A)
convert_to_lmaps_(A::LinearMap) = A
convert_to_lmaps() = ()
convert_to_lmaps(A) = (convert_to_lmaps_(A),)
@inline convert_to_lmaps(A, B, Cs...) =
    (convert_to_lmaps_(A), convert_to_lmaps_(B), convert_to_lmaps(Cs...)...)

function Base.:(*)(A::LinearMap, x::AbstractVector)
    size(A, 2) == length(x) || throw(DimensionMismatch("mul!"))
    return @inbounds A_mul_B!(similar(x, promote_type(eltype(A), eltype(x)), size(A, 1)), A, x)
end
function LinearAlgebra.mul!(y::AbstractVector, A::LinearMap, x::AbstractVector)
    @boundscheck check_dim_mul(y, A, x)
    return @inbounds A_mul_B!(y, A, x)
end
function LinearAlgebra.mul!(y::AbstractVector, A::LinearMap, x::AbstractVector, α::Number, β::Number)
    @boundscheck check_dim_mul(y, A, x)
    if isone(α)
        iszero(β) && (A_mul_B!(y, A, x); return y)
        isone(β) && (y .+= A * x; return y)
        # β != 0, 1
        rmul!(y, β)
        y .+= A * x
        return y
    elseif iszero(α)
        iszero(β) && (fill!(y, zero(eltype(y))); return y)
        isone(β) && return y
        # β != 0, 1
        rmul!(y, β)
        return y
    else # α != 0, 1
        iszero(β) && (A_mul_B!(y, A, x); rmul!(y, α); return y)
        isone(β) && (y .+= rmul!(A * x, α); return y)
        # β != 0, 1
        rmul!(y, β)
        y .+= rmul!(A * x, α)
        return y
    end
end
# the following is of interest in, e.g., subspace-iteration methods
Base.@propagate_inbounds function LinearAlgebra.mul!(Y::AbstractMatrix, A::LinearMap, X::AbstractMatrix, α::Number=true, β::Number=false)
    @boundscheck check_dim_mul(Y, A, X)
    @inbounds @views for i = 1:size(X, 2)
        mul!(Y[:, i], A, X[:, i], α, β)
    end
    # starting from Julia v1.1, we could use the `eachcol` iterator
    # for (Xi, Yi) in zip(eachcol(X), eachcol(Y))
    #     mul!(Yi, A, Xi, α, β)
    # end
    return Y
end

A_mul_B!(y::AbstractVector, A::AbstractMatrix, x::AbstractVector)  = mul!(y, A, x)
At_mul_B!(y::AbstractVector, A::AbstractMatrix, x::AbstractVector) = mul!(y, transpose(A), x)
Ac_mul_B!(y::AbstractVector, A::AbstractMatrix, x::AbstractVector) = mul!(y, adjoint(A), x)

include("left.jl") # left multiplication by a transpose or adjoint vector
include("wrappedmap.jl") # wrap a matrix of linear map in a new type, thereby allowing to alter its properties
include("uniformscalingmap.jl") # the uniform scaling map, to be able to make linear combinations of LinearMap objects and multiples of I
include("transpose.jl") # transposing linear maps
include("linearcombination.jl") # defining linear combinations of linear maps
include("scaledmap.jl") # multiply by a (real or complex) scalar
include("composition.jl") # composition of linear maps
include("functionmap.jl") # using a function as linear map
include("blockmap.jl") # block linear maps
include("kronecker.jl") # Kronecker product of linear maps
include("conversion.jl") # conversion of linear maps to matrices

"""
    LinearMap(A::LinearMap,AbstractMatrix}; kwargs...)::WrappedMap
    LinearMap(A::AbstractMatrix; kwargs...)::WrappedMap
    LinearMap(J::UniformScaling, M::Int)::UniformScalingMap
    LinearMap{T=Float64}(f, [fc,], M::Int, N::Int = M; kwargs...)::FunctionMap

Construct a linear map object, either from an existing `LinearMap` or `AbstractMatrix` `A`,
with the purpose of redefining its properties via the keyword arguments `kwargs`;
a `UniformScaling` object `J` with specified (square) dimension `M`; or
from a function or callable object `f`. In the latter case, one also needs to specify
the size of the equivalent matrix representation `(M, N)`, i.e., for functions `f` acting
on length `N` vectors and producing length `M` vectors (with default value `N=M`). Preferably,
also the `eltype` `T` of the corresponding matrix representation needs to be specified, i.e.
whether the action of `f` on a vector will be similar to, e.g., multiplying by numbers of type `T`.
If not specified, the devault value `T=Float64` will be assumed. Optionally, a corresponding
function `fc` can be specified that implements the adjoint (=transpose in the real case) of `f`.

The keyword arguments and their default values for the function-based constructor are:
*   `issymmetric::Bool = false` : whether `A` or `f` acts as a symmetric matrix
*   `ishermitian::Bool = issymmetric & T<:Real` : whether `A` or `f` acts as a Hermitian matrix
*   `isposdef::Bool = false` : whether `A` or `f` acts as a positive definite matrix.
For existing linear maps or matrices `A`, the default values will be taken by calling
`issymmetric`, `ishermitian` and `isposdef` on the existing object `A`.

For the function-based constructor, there is one more keyword argument:
*   `ismutating::Bool` : flags whether the function acts as a mutating matrix multiplication
    `f(y,x)` where the result vector `y` is the first argument (in case of `true`),
    or as a normal matrix multiplication that is called as `y=f(x)` (in case of `false`).
    The default value is guessed by looking at the number of arguments of the first occurrence
    of `f` in the method table.
"""
LinearMap(A::MapOrMatrix; kwargs...) = WrappedMap(A; kwargs...)
LinearMap(J::UniformScaling, M::Int) = UniformScalingMap(J.λ, M)
LinearMap(f, M::Int; kwargs...) = LinearMap{Float64}(f, M; kwargs...)
LinearMap(f, M::Int, N::Int; kwargs...) = LinearMap{Float64}(f, M, N; kwargs...)
LinearMap(f, fc, M::Int; kwargs...) = LinearMap{Float64}(f, fc, M; kwargs...)
LinearMap(f, fc, M::Int, N::Int; kwargs...) = LinearMap{Float64}(f, fc, M, N; kwargs...)

LinearMap{T}(A::MapOrMatrix; kwargs...) where {T} = WrappedMap{T}(A; kwargs...)
LinearMap{T}(f, args...; kwargs...) where {T} = FunctionMap{T}(f, args...; kwargs...)

end # module
