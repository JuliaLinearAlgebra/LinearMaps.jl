struct FunctionMap{T, F1, F2, iip} <: LinearMap{T}
    f::F1
    fc::F2
    M::Int
    N::Int
    _issymmetric::Bool
    _ishermitian::Bool
    _isposdef::Bool
end

"""
    FunctionMap{T,iip}(f, [fc,], M, N = M; kwargs...)

Construct a `FunctionMap` object from a function or callable object `f` that represents a
linear map of size `(M, N)`, where `N` can be omitted for square operators of size `(M,M)`.
Furthermore, the `eltype` `T` of the corresponding matrix representation needs to be
specified, i.e., whether the action of `f` on a vector will be similar to, e.g., multiplying
by numbers of type `T`. Optionally, a second function `fc` can be specified that implements
the adjoint (or transpose in the real case) of `f`.

Accepted keyword arguments and their default values are as in the [`LinearMap`](@ref)
constructor.

# Examples
```jldoctest
julia> F = FunctionMap{Int64,false}(cumsum, 2)
2×2 FunctionMap{Int64,false}(cumsum; issymmetric=false, ishermitian=false, isposdef=false)

julia> F * ones(Int64, 2)
2-element Vector{Int64}:
 1
 2

julia> Matrix(F)
2×2 Matrix{Int64}:
 1  0
 1  1
"""
function FunctionMap{T,iip}(f::F1, fc::F2, M::Int, N::Int;
    issymmetric::Bool = false,
    ishermitian::Bool = (T<:Real && issymmetric),
    isposdef::Bool    = false) where {T, F1, F2, iip}
    FunctionMap{T, F1, F2, iip}(f, fc, M, N, issymmetric, ishermitian, isposdef)
end

# additional constructors
FunctionMap{T,iip}(f, M::Int, N::Int=M; kwargs...) where {T, iip} =
    FunctionMap{T,iip}(f, nothing, M, N; kwargs...)
FunctionMap{T,iip}(f, fc, M::Int; kwargs...) where {T, iip} =
    FunctionMap{T,iip}(f, fc, M, M; kwargs...)

FunctionMap{T}(f, fc, M::Int, N::Int; ismutating::Bool = _ismutating(f), kwargs...) where {T} =
    FunctionMap{T, ismutating}(f, fc, M, N; kwargs...)
FunctionMap{T}(f, M::Int; ismutating::Bool = _ismutating(f), kwargs...) where {T} =
    FunctionMap{T, ismutating}(f, nothing, M, M; kwargs...)
FunctionMap{T}(f, M::Int, N::Int; ismutating::Bool = _ismutating(f), kwargs...) where {T} =
    FunctionMap{T, ismutating}(f, nothing, M, N; kwargs...)
FunctionMap{T}(f, fc, M::Int; ismutating::Bool = _ismutating(f), kwargs...) where {T} =
    FunctionMap{T, ismutating}(f, fc, M, M; kwargs...)

const OOPFunctionMap{T,F1,F2} = FunctionMap{T,F1,F2,false}
const IIPFunctionMap{T,F1,F2} = FunctionMap{T,F1,F2,true}

# properties
Base.size(A::FunctionMap) = (A.M, A.N)
LinearAlgebra.issymmetric(A::FunctionMap) = A._issymmetric
LinearAlgebra.ishermitian(A::FunctionMap) = A._ishermitian
LinearAlgebra.isposdef(A::FunctionMap)    = A._isposdef
MulStyle(::OOPFunctionMap) = TwoArg()
MulStyle(::IIPFunctionMap) = ThreeArg()
@deprecate ismutating(A::FunctionMap) (a -> (MulStyle(a) === ThreeArg()))(A) false
_ismutating(f) = first(methods(f)).nargs == 3

# multiplication with vector
const TransposeFunctionMap = TransposeMap{<:Any, <:FunctionMap}
const AdjointFunctionMap = AdjointMap{<:Any, <:FunctionMap}

@inline function _apply_fun(::MulStyle, f!, x, m, T)
    y = similar(x, T, m)
    f!(y, x)
    return y
end
@inline function _apply_fun(::TwoArg, f, x, m, _)
    y = f(x)
    length(y) == m || throw(DimensionMismatch())
    return y
end

function Base.:(*)(A::FunctionMap, x::AbstractVector)
    length(x) == size(A, 2) || throw(DimensionMismatch())
    T = promote_type(eltype(A), eltype(x))
    return _apply_fun(MulStyle(A), A.f, x, size(A, 1), T)
end
function Base.:(*)(A::AdjointFunctionMap, x::AbstractVector)
    Afun = A.lmap
    ishermitian(Afun) && return Afun*x
    length(x) == size(A, 2) || throw(DimensionMismatch())
    T = promote_type(eltype(A), eltype(x))
    if Afun.fc !== nothing
        return _apply_fun(MulStyle(Afun), Afun.fc, x, size(A, 1), T)
    elseif issymmetric(Afun) # but !isreal(A), Afun.f can be used
        y = _apply_fun(MulStyle(Afun), Afun.f, conj(x), size(A, 1), T)
        conj!(y)
        return y
    else
        error("adjoint not implemented for $(A.lmap)")
    end
end
function Base.:(*)(A::TransposeFunctionMap, x::AbstractVector)
    Afun = A.lmap
    (issymmetric(Afun) || (isreal(A) && ishermitian(Afun))) && return Afun*x
    length(x) == size(A, 2) || throw(DimensionMismatch())
    T = promote_type(eltype(A), eltype(x))
    if Afun.fc !== nothing
        z = !isreal(A) ? conj(x) : x
        y = _apply_fun(MulStyle(Afun), Afun.fc, z, size(A, 1), T)
        !isreal(A) && conj!(y)
    elseif ishermitian(Afun) # but !isreal(A), Afun.f can be used
        y = _apply_fun(MulStyle(Afun), Afun.f, conj(x), size(A, 1), T)
        conj!(y)
    else
        error("transpose not implemented for $(A.lmap)")
    end
    return y
end

_unsafe_mul!(y, A::OOPFunctionMap, x::AbstractVector) = copyto!(y, A.f(x))
_unsafe_mul!(y, A::IIPFunctionMap, x::AbstractVector) = (A.f(y, x); return y)

function _unsafe_mul!(y, At::TransposeFunctionMap, x::AbstractVector)
    A = At.lmap
    (issymmetric(A) || (isreal(A) && ishermitian(A))) && return _unsafe_mul!(y, A, x)
    if A.fc !== nothing
        if !isreal(A)
            x = conj(x)
        end
        MulStyle(A) === TwoArg() ? copyto!(y, A.fc(x)) : A.fc(y, x)
        if !isreal(A)
            conj!(y)
        end
        return y
    elseif ishermitian(A) # but !isreal(A)
        _unsafe_mul!(y, A, conj(x))
        conj!(y)
        return y
    else
        error("transpose not implemented for $A")
    end
end

function _unsafe_mul!(y, Ac::AdjointFunctionMap, x::AbstractVector)
    A = Ac.lmap
    ishermitian(A) && return _unsafe_mul!(y, A, x)
    if A.fc !== nothing
        MulStyle(A) === TwoArg() ? copyto!(y, A.fc(x)) : A.fc(y, x)
        return y
    elseif issymmetric(A) # but !isreal(A)
        _unsafe_mul!(y, A, conj(x))
        conj!(y)
        return y
    else
        error("adjoint not implemented for $A")
    end
end
