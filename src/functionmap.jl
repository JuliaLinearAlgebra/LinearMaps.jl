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
FunctionMap{T}(f, M::Int; kwargs...) where {T} =
    FunctionMap{T}(f, nothing, M, M; kwargs...)
FunctionMap{T}(f, M::Int, N::Int; kwargs...) where {T} =
    FunctionMap{T}(f, nothing, M, N; kwargs...)
FunctionMap{T}(f, fc, M::Int; kwargs...) where {T} =
    FunctionMap{T}(f, fc, M, M; kwargs...)

# properties
Base.size(A::FunctionMap) = (A.M, A.N)
LinearAlgebra.issymmetric(A::FunctionMap) = A._issymmetric
LinearAlgebra.ishermitian(A::FunctionMap) = A._ishermitian
LinearAlgebra.isposdef(A::FunctionMap)    = A._isposdef
ismutating(A::FunctionMap) = A._ismutating
_ismutating(f) = first(methods(f)).nargs == 3

# multiplication with vector
const TransposeFunctionMap = TransposeMap{<:Any, <:FunctionMap}
const AdjointFunctionMap = AdjointMap{<:Any, <:FunctionMap}

function Base.:(*)(A::FunctionMap, x::AbstractVector)
    length(x) == size(A, 2) || throw(DimensionMismatch())
    if ismutating(A)
        y = similar(x, promote_type(eltype(A), eltype(x)), size(A, 1))
        A.f(y, x)
    else
        y = A.f(x)
        length(y) == size(A, 1) || throw(DimensionMismatch())
    end
    return y
end
function Base.:(*)(A::AdjointFunctionMap, x::AbstractVector)
    Afun = A.lmap
    ishermitian(Afun) && return Afun*x
    length(x) == size(A, 2) || throw(DimensionMismatch())
    if Afun.fc !== nothing
        if ismutating(Afun)
            y = similar(x, promote_type(eltype(A), eltype(x)), size(A, 1))
            Afun.fc(y, x)
        else
            y = Afun.fc(x)
            length(y) == size(A, 1) || throw(DimensionMismatch())
        end
        return y
    elseif issymmetric(Afun) # but !isreal(A), Afun.f can be used
        x = conj(x)
        if ismutating(Afun)
            y = similar(x, promote_type(eltype(A), eltype(x)), size(A, 1))
            Afun.f(y, x)
        else
            y = Afun.f(x)
            length(y) == size(A, 1) || throw(DimensionMismatch())
        end
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
    if Afun.fc !== nothing
        if !isreal(A)
            x = conj(x)
        end
        if ismutating(Afun)
            y = similar(x, promote_type(eltype(A), eltype(x)), size(A, 1))
            Afun.fc(y, x)
        else
            y = Afun.fc(x)
            length(y) == size(A, 1) || throw(DimensionMismatch())
        end
        if !isreal(A)
            conj!(y)
        end
        return y
    elseif ishermitian(Afun) # but !isreal(A), Afun.f can be used
        x = conj(x)
        if ismutating(Afun)
            y = similar(x, promote_type(eltype(A), eltype(x)), size(A, 1))
            Afun.f(y, x)
        else
            y = Afun.f(x)
            length(y) == size(A, 1) || throw(DimensionMismatch())
        end
        conj!(y)
        return y
    else
        error("transpose not implemented for $(A.lmap)")
    end
end

function _unsafe_mul!(y::AbstractVecOrMat, A::FunctionMap, x::AbstractVector)
    ismutating(A) ? A.f(y, x) : copyto!(y, A.f(x))
    return y
end

function _unsafe_mul!(y::AbstractVecOrMat, At::TransposeFunctionMap, x::AbstractVector)
    A = At.lmap
    (issymmetric(A) || (isreal(A) && ishermitian(A))) && return _unsafe_mul!(y, A, x)
    if A.fc !== nothing
        if !isreal(A)
            x = conj(x)
        end
        ismutating(A) ? A.fc(y, x) : copyto!(y, A.fc(x))
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

function _unsafe_mul!(y::AbstractVecOrMat, Ac::AdjointFunctionMap, x::AbstractVector)
    A = Ac.lmap
    ishermitian(A) && return _unsafe_mul!(y, A, x)
    if A.fc !== nothing
        ismutating(A) ? A.fc(y, x) : copyto!(y, A.fc(x))
        return y
    elseif issymmetric(A) # but !isreal(A)
        _unsafe_mul!(y, A, conj(x))
        conj!(y)
        return y
    else
        error("adjoint not implemented for $A")
    end
end
