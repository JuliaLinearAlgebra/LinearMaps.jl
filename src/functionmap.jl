struct FunctionMap{T, F1, F2} <: LinearMap{T}
    f::F1
    fc::F2
    M::Int
    N::Int
    _ismutating::Bool
    _issymmetric::Bool
    _ishermitian::Bool
    _isposdef::Bool
end
function FunctionMap{T}(f::F1, fc::F2, M::Int, N::Int;
    ismutating::Bool  = _ismutating(f),
    issymmetric::Bool = false,
    ishermitian::Bool = (T<:Real && issymmetric),
    isposdef::Bool    = false) where {T, F1, F2}
    FunctionMap{T, F1, F2}(f, fc, M, N, ismutating, issymmetric, ishermitian, isposdef)
end

# additional constructors
FunctionMap{T}(f, M::Int; kwargs...) where {T}         = FunctionMap{T}(f, nothing, M, M; kwargs...)
FunctionMap{T}(f, M::Int, N::Int; kwargs...) where {T} = FunctionMap{T}(f, nothing, M, N; kwargs...)
FunctionMap{T}(f, fc, M::Int; kwargs...) where {T}     = FunctionMap{T}(f, fc, M, M; kwargs...)

# show
function Base.show(io::IO, A::FunctionMap{T, F, Nothing}) where {T, F}
    print(io, "LinearMaps.FunctionMap{$T}($(A.f), $(A.M), $(A.N); ismutating=$(A._ismutating), issymmetric=$(A._issymmetric), ishermitian=$(A._ishermitian), isposdef=$(A._isposdef))")
end
function Base.show(io::IO, A::FunctionMap{T}) where {T}
    print(io, "LinearMaps.FunctionMap{$T}($(A.f), $(A.fc), $(A.M), $(A.N); ismutating=$(A._ismutating), issymmetric=$(A._issymmetric), ishermitian=$(A._ishermitian), isposdef=$(A._isposdef))")
end

# properties
Base.size(A::FunctionMap, n) = n==1 ? A.M : n==2 ? A.N : error("LinearMap objects have only 2 dimensions")
Base.size(A::FunctionMap) = (A.M, A.N)
LinearAlgebra.issymmetric(A::FunctionMap) = A._issymmetric
LinearAlgebra.ishermitian(A::FunctionMap) = A._ishermitian
LinearAlgebra.isposdef(A::FunctionMap)    = A._isposdef
ismutating(A::FunctionMap) = A._ismutating
_ismutating(f) = (mf = methods(f); !isempty(mf) ? first(methods(f)).nargs == 3 : error("transpose/adjoint not implemented"))

# special transposition behavior
function LinearAlgebra.transpose(A::FunctionMap)
    if A.fc!==nothing || A._issymmetric
        return TransposeMap(A)
    else
        error("transpose not implemented for $A")
    end
end
LinearAlgebra.adjoint(A::FunctionMap{<:Real}) = transpose(A)
function LinearAlgebra.adjoint(A::FunctionMap)
    if A.fc!==nothing || A._ishermitian
        return AdjointMap(A)
    else
        error("adjoint not implemented for $A")
    end
end

# multiplication with vector
function Base.:(*)(A::FunctionMap, x::AbstractVector)
    length(x) == A.N || throw(DimensionMismatch())
    if ismutating(A)
        y = similar(x, promote_type(eltype(A), eltype(x)), A.M)
        A.f(y, x)
    else
        y = A.f(x)
    end
    return y
end

function A_mul_B!(y::AbstractVector, A::FunctionMap, x::AbstractVector)
    (length(x) == A.N && length(y) == A.M) || throw(DimensionMismatch("A_mul_B!"))
    ismutating(A) ? A.f(y, x) : copyto!(y, A.f(x))
    return y
end

function At_mul_B!(y::AbstractVector, A::FunctionMap, x::AbstractVector)
    issymmetric(A) && return A_mul_B!(y, A, x)
    (length(x) == A.M && length(y) == A.N) || throw(DimensionMismatch("At_mul_B!"))
    if A.fc !== nothing
        if !isreal(A)
            x = conj(x)
        end
        ismutating(A) ? A.fc(y, x) : copyto!(y, A.fc(x))
        if !isreal(A)
            conj!(y)
        end
        return y
    else
        error("transpose not implemented for $A")
    end
end

function Ac_mul_B!(y::AbstractVector, A::FunctionMap, x::AbstractVector)
    ishermitian(A) && return A_mul_B!(y, A, x)
    (length(x) == A.M && length(y) == A.N) || throw(DimensionMismatch("Ac_mul_B!"))
    if A.fc !== nothing
        ismutating(A) ? A.fc(y, x) : copyto!(y, A.fc(x))
        return y
    else
        error("adjoint not implemented for $A")
    end
end
