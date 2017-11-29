__precompile__(true)
module LinearMaps

export LinearMap, AbstractLinearMap

import Base: +, -, *, \, /, ==, transpose

if VERSION >= v"0.7.0-DEV.1415"
    const adjoint = Base.adjoint
else
    const adjoint = Base.ctranspose
end


abstract type LinearMap{T} end

const AbstractLinearMap = LinearMap # will be deprecated

Base.eltype(::LinearMap{T}) where {T} = T
Base.eltype(::Type{L}) where {T,L<:LinearMap{T}} = T

Base.isreal(A::LinearMap) = eltype(A) <: Real
Base.issymmetric(::LinearMap) = false # default assumptions
Base.ishermitian(A::LinearMap{<:Real}) = issymmetric(A)
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

# the following for multiplying with transpose and adjoint map are optional:
# subtypes can overwrite nonmutating methods, implement mutating methods or do nothing
function Base.At_mul_B(A::LinearMap, x::AbstractVector)
    l = methods(Base.At_mul_B!,Tuple{AbstractVector, typeof(A), AbstractVector})
    if length(l) > 0 && first(l.ms).sig.parameters[3] != LinearMap
        Base.At_mul_B!(similar(x, promote_type(eltype(A), eltype(x)), size(A,2)), A, x)
    else
        throw(MethodError(Base.At_mul_B, (A, x)))
    end
end
function Base.At_mul_B!(y::AbstractVector, A::LinearMap, x::AbstractVector)
    length(y) == size(A, 2) || throw(DimensionMismatch("At_mul_B!"))
    l = methods(Base.At_mul_B,Tuple{typeof(A), AbstractVector})
    if length(l) > 0 && first(l.ms).sig.parameters[2] != LinearMap
        copy!(y, Base.At_mul_B(A, x))
    else
        throw(MethodError(Base.At_mul_B!, (y, A, x)))
    end
end
function Base.Ac_mul_B(A::LinearMap,x::AbstractVector)
    l = methods(Base.Ac_mul_B!,Tuple{AbstractVector, typeof(A), AbstractVector})
    if length(l) > 0 && first(l.ms).sig.parameters[3] != LinearMap
        Base.Ac_mul_B!(similar(x, promote_type(eltype(A), eltype(x)), size(A,2)), A, x)
    else
        throw(MethodError(Base.Ac_mul_B, (A, x)))
    end
end
function
Base.Ac_mul_B!(y::AbstractVector, A::LinearMap, x::AbstractVector)
    length(y)==size(A,2) || throw(DimensionMismatch("At_mul_B!"))
    l = methods(Base.Ac_mul_B,Tuple{typeof(A), AbstractVector})
    if length(l) > 0 && first(l.ms).sig.parameters[2] != LinearMap
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

# sparse: create sparse matrix representtion of LinearMap
function Base.sparse(A::LinearMap)
    M, N = size(A)
    T = eltype(A)
    rowind = Int[]
    nzval = T[]
    colptr = Vector{Int}(N+1)
    v = zeros(T, N)

    for i = 1:N
        v[i] = one(T)
        Lv = A*v
        js = find(Lv)
        colptr[i] = length(nzval)+1
        if length(js) > 0
            append!(rowind, js)
            append!(nzval, Lv[js])
        end
        v[i] = zero(T)
    end
    colptr[N+1] = length(nzval)+1
    
    return SparseMatrixCSC(M, N, colptr, rowind, nzval)
end

include("transpose.jl") # transposing linear maps
include("linearcombination.jl") # defining linear combinations of linear maps
include("composition.jl") # composition of linear maps
include("wrappedmap.jl") # wrap a matrix of linear map in a new type, thereby allowing to alter its properties
include("identitymap.jl") # the identity map, to be able to make linear combinations of LinearMap objects and I
include("functionmap.jl") # using a function as linear map

"""
    LinearMap(A; kwargs...)
    LinearMap{T=Float64}(f, [fc,], M::Int, N::Int = M; kwargs...)

Construct a linear map object, either from an existing `LinearMap` or `AbstractMatrix` `A`,
with the purpose of redefining its properties via the keyword arguments `kwargs`, or
from a function or callable object `f`. In the latter case, one also needs to specialize
the size of the equivalent matrix representation `(M,N)`, i.e. for functions `f` acting
on length `N` vectors and producing length `M` vectors (with default value `N=M`). Preferably,
also the `eltype` `T` of the corresponding matrix representation needs to be specified, i.e.
whether the action of `f` on a vector will be similar to e.g. multiplying by numbers of type `T`.
If not specified, the devault value `T=Float64` will be assumed. Optionally, a corresponding
function `fc` can be specified that implements the (conjugate) transpose of `f`.

The keyword arguments and their default values for functions `f` are
*   issymmetric::Bool = false : whether `A` or `f` acts as a symmetric matrix
*   ishermitian::Bool = issymmetric & T<:Real : whether `A` or `f` acts as a Hermitian matrix
*   isposdef::Bool = false : whether `A` or `f` acts as a positive definite matrix.
For existing linear maps or matrices `A`, the default values will be taken by calling
`issymmetric`, `ishermitian` and `isposdef` on the exising object `A`.

For functions `f`, there is one more keyword arguments
*   ismutating::Bool : flags whether the function acts as a mutating matrix multiplication
    `f(y,x)` where the result vector `y` is the first argument (in case of `true`),
    or as a normal matrix multiplication that is called as `y=f(x)` (in case of `false`).
    The default value is guessed by looking at the number of arguments of the first occurence
    of `f` in the method table.
"""
LinearMap(A::Union{AbstractMatrix,LinearMap}; kwargs...) = WrappedMap(A; kwargs...)
LinearMap(f, M::Int; kwargs...) = LinearMap{Float64}(f, M; kwargs...)
LinearMap(f, M::Int, N::Int; kwargs...) = LinearMap{Float64}(f, M, N; kwargs...)
LinearMap(f, fc, M::Int; kwargs...) = LinearMap{Float64}(f, fc, M; kwargs...)
LinearMap(f, fc, M::Int, N::Int; kwargs...) = LinearMap{Float64}(f, fc, M, N; kwargs...)

(::Type{LinearMap{T}})(A::Union{AbstractMatrix,LinearMap}; kwargs...) where {T} = WrappedMap{T}(A; kwargs...)
(::Type{LinearMap{T}})(f, args...; kwargs...) where {T} = FunctionMap{T}(f, args...; kwargs...)

@deprecate LinearMap(f, T::Type, args...; kwargs...) LinearMap{T}(f, args...; kwargs...)
@deprecate LinearMap(f, fc, T::Type, args...; kwargs...) LinearMap{T}(f, fc, args...; kwargs...)

@deprecate LinearMap(f, M::Int, T::Type; kwargs...) LinearMap{T}(f, M; kwargs...)
@deprecate LinearMap(f, M::Int, N::Int, T::Type; kwargs...) LinearMap{T}(f, M, N; kwargs...)
@deprecate LinearMap(f, fc, M::Int, T::Type; kwargs...) LinearMap{T}(f, fc, M; kwargs...)
@deprecate LinearMap(f, fc, M::Int, N::Int, T::Type; kwargs...) LinearMap{T}(f, fc, M, N; kwargs...)

end # module
