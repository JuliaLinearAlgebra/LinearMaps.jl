module LinearMaps

export LinearMap, AbstractLinearMap

import Base: +, -, *, \, /, ==

abstract type LinearMap{T} end

const AbstractLinearMap = LinearMap # will be deprecated

Base.eltype{T}(::LinearMap{T})=T
Base.eltype{T}(::Type{LinearMap{T}})=T
Base.eltype{L<:LinearMap}(::Type{L})=eltype(super(L))

Base.isreal{T<:Real}(::LinearMap{T}) = true
Base.isreal(::LinearMap) = false # standard assumptions
Base.issymmetric(::LinearMap) = false # standard assumptions
Base.ishermitian{T<:Real}(A::LinearMap{T}) = issymmetric(A)
Base.ishermitian(::LinearMap) = false # standard assumptions
Base.isposdef(::LinearMap) = false # standard assumptions

Base.ndims(::LinearMap) = 2
Base.size(A::LinearMap, n) = (n==1 || n==2 ? size(A)[n] : error("LinearMap objects have only 2 dimensions"))
Base.length(A::LinearMap) = reduce(*, size(A))

# any LinearMap subtype will have to overwrite at least one of the two following methods to avoid running in circles
*(A::LinearMap, x::AbstractVector) = Base.A_mul_B!(similar(x, promote_type(eltype(A),eltype(x)), size(A,1)), A, x)
Base.A_mul_B!(y::AbstractVector, A::LinearMap, x::AbstractVector) = begin
    length(y)==size(A,1) || throw(DimensionMismatch("A_mul_B!"))
    copy!(y,A*x)
end

# the following for multiplying with transpose and ctranspose map are optional:
# subtypes can overwrite nonmutating methods, implement mutating methods or do nothing
Base.At_mul_B(A::LinearMap,x::AbstractVector)=(@which Base.At_mul_B!(x,A,x))!=methods(Base.At_mul_B!,(AbstractVector,LinearMap,AbstractVector))[end] ?
    Base.At_mul_B!(similar(x,promote_type(eltype(A),eltype(x)),size(A,2)),A,x) : throw(MethodError(Base.At_mul_B,(A,x)))
Base.At_mul_B!(y::AbstractVector,A::LinearMap,x::AbstractVector)=begin
    length(y)==size(A,2) || throw(DimensionMismatch("At_mul_B!"))
    (@which Base.At_mul_B(A,x))!=methods(Base.At_mul_B,(LinearMap,AbstractVector))[end] ? copy!(y,Base.At_mul_B(A,x)) : throw(MethodError(Base.At_mul_B!,(y,A,x)))
end
Base.Ac_mul_B(A::LinearMap,x::AbstractVector)=(@which Base.Ac_mul_B!(x,A,x))!=methods(Base.Ac_mul_B!,(AbstractVector,LinearMap,AbstractVector))[end] ?
    Base.Ac_mul_B!(similar(x,promote_type(eltype(A),eltype(x)),size(A,2)),A,x) : throw(MethodError(Base.Ac_mul_B,(A,x)))
Base.Ac_mul_B!(y::AbstractVector,A::LinearMap,x::AbstractVector)=begin
    length(y)==size(A,2) || throw(DimensionMismatch("At_mul_B!"))
    (@which Base.Ac_mul_B(A,x))!=methods(Base.Ac_mul_B,(LinearMap,AbstractVector))[end] ? copy!(y,Base.Ac_mul_B(A,x)) : throw(MethodError(Base.Ac_mul_B!,(y,A,x)))
end

# full: create matrix representation of LinearMap
function Base.full{T}(A::LinearMap{T})
    M, N = size(A)
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

LinearMap(A::Union{AbstractMatrix,LinearMap; kwargs...) = WrappedMap(A; kwargs...)
LinearMap(args...; kwargs...) = FunctionMap(args...; kwargs...)

(::Type{LinearMap{T}})(args...; kwargs...) where {T} = FunctionMap{T}(args...; kwargs...)


end # module
