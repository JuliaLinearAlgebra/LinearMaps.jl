module LinearMaps

export LinearMap, AbstractLinearMap

import Base: +, -, *, \, /, ==

abstract type LinearMap{T} end

const AbstractLinearMap = LinearMap # will be deprecated

Base.eltype{T}(::LinearMap{T}) = T
Base.eltype{T,L<:LinearMap{T}}(::Type{L})=T

Base.isreal(A::LinearMap) = eltype(A) <: Real
Base.issymmetric(::LinearMap) = false # default assumptions
Base.ishermitian{T<:Real}(A::LinearMap{T}) = issymmetric(A)
Base.ishermitian(::LinearMap) = false # default assumptions
Base.isposdef(::LinearMap) = false # default assumptions

Base.ndims(::LinearMap) = 2
Base.size(A::LinearMap, n) = (n==1 || n==2 ? size(A)[n] : error("LinearMap objects have only 2 dimensions"))
Base.length(A::LinearMap) = size(A)[1] * size(A)[2]

# any LinearMap subtype will have to overwrite at least one of the two following methods to avoid running in circles
*(A::LinearMap, x::AbstractVector) = Base.A_mul_B!(similar(x, promote_type(eltype(A),eltype(x)), size(A,1)), A, x)
Base.A_mul_B!(y::AbstractVector, A::LinearMap, x::AbstractVector) = begin
    length(y) == size(A,1) || throw(DimensionMismatch("A_mul_B!"))
    copy!(y, A*x)
end

# the following for multiplying with transpose and ctranspose map are optional:
# subtypes can overwrite nonmutating methods, implement mutating methods or do nothing
function Base.At_mul_B(A::LinearMap, x::AbstractVector)
    if length(methods(Base.At_mul_B!,Tuple{AbstractVector, typeof(A), AbstractVector})) > 1
        Base.At_mul_B!(similar(x, promote_type(eltype(A), eltype(x)), size(A,2)), A, x)
    else
        throw(MethodError(Base.At_mul_B, (A, x)))
    end
end
function Base.At_mul_B!(y::AbstractVector, A::LinearMap, x::AbstractVector)
    length(y) == size(A, 2) || throw(DimensionMismatch("At_mul_B!"))
    if length(methods(Base.At_mul_B,Tuple{typeof(A), AbstractVector})) > 1
        copy!(y, Base.At_mul_B(A, x))
    else
        throw(MethodError(Base.At_mul_B!, (y, A, x)))
    end
end
function Base.Ac_mul_B(A::LinearMap,x::AbstractVector)
    if length(methods(Base.Ac_mul_B!,Tuple{AbstractVector, typeof(A), AbstractVector})) > 1
        Base.Ac_mul_B!(similar(x, promote_type(eltype(A), eltype(x)), size(A,2)), A, x)
    else
        throw(MethodError(Base.Ac_mul_B, (A, x)))
    end
end
function
Base.Ac_mul_B!(y::AbstractVector, A::LinearMap, x::AbstractVector)
    length(y)==size(A,2) || throw(DimensionMismatch("At_mul_B!"))
    if length(methods(Base.Ac_mul_B,Tuple{typeof(A), AbstractVector})) > 1
        copy!(y, Base.Ac_mul_B(A, x))
    else
        throw(MethodError(Base.Ac_mul_B!, (y, A, x)))
    end
end

# full: create matrix representation of LinearMap
function Base.full(A::LinearMap)
    M, N = size(A)
    T = eltype(A)
    mat = zeros(T, (M, N))
    v = zeros(T, N)
    for i = 1:N
        v[i] = one(T)
        A_mul_B!(view(mat,:,i), A, v)
        v[i] = zero(T)
    end
    return mat
end

include("transpose.jl") # transposing linear maps
include("linearcombination.jl") # defining linear combinations of linear maps
include("composition.jl") # composition of linear maps
include("wrappedmap.jl") # wrap a matrix of linear map in a new type, thereby allowing to alter its properties
include("identitymap.jl") # the identity map, to be able to make linear combinations of LinearMap objects and I
include("functionmap.jl") # using a function as linear map

LinearMap(A::Union{AbstractMatrix,LinearMap}; kwargs...) = WrappedMap(A; kwargs...)
LinearMap(f, M::Int; kwargs...) = LinearMap{Float64}(f, M; kwargs...)
LinearMap(f, M::Int, N::Int; kwargs...) = LinearMap{Float64}(f, M, N; kwargs...)
LinearMap(f, fc, M::Int; kwargs...) = LinearMap{Float64}(f, fc, M; kwargs...)
LinearMap(f, fc, M::Int, N::Int; kwargs...) = LinearMap{Float64}(f, fc, M, N; kwargs...)

(::Type{LinearMap{T}}){T}(A::Union{AbstractMatrix,LinearMap}; kwargs...) = WrappedMap{T}(A; kwargs...)
(::Type{LinearMap{T}}){T}(f, args...; kwargs...) = FunctionMap{T}(f, args...; kwargs...)

@deprecate LinearMap(f, T::Type, args...; kwargs...) LinearMap{T}(f, args...; kwargs...)
@deprecate LinearMap(f, fc, T::Type, args...; kwargs...) LinearMap{T}(f, fc, args...; kwargs...)

@deprecate LinearMap(f, M::Int, T::Type; kwargs...) LinearMap{T}(f, M; kwargs...)
@deprecate LinearMap(f, M::Int, N::Int, T::Type; kwargs...) LinearMap{T}(f, M, N; kwargs...)
@deprecate LinearMap(f, fc, M::Int, T::Type; kwargs...) LinearMap{T}(f, fc, M; kwargs...)
@deprecate LinearMap(f, fc, M::Int, N::Int, T::Type; kwargs...) LinearMap{T}(f, fc, M, N; kwargs...)

end # module
