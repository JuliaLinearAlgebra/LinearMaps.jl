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
Base.size(A::FunctionMap) = (A.M, A.N)
LinearAlgebra.issymmetric(A::FunctionMap) = A._issymmetric
LinearAlgebra.ishermitian(A::FunctionMap) = A._ishermitian
LinearAlgebra.isposdef(A::FunctionMap)    = A._isposdef
ismutating(A::FunctionMap) = A._ismutating
_ismutating(f) = first(methods(f)).nargs == 3

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
function Base.:(*)(A::AdjointMap{<:Any,<:FunctionMap}, x::AbstractVector)
    Afun = A.lmap
    ishermitian(Afun) && return Afun*x
    length(x) == size(A, 2) || throw(DimensionMismatch())
    if Afun.fc !== nothing
        if ismutating(Afun)
            y = similar(x, promote_type(eltype(A), eltype(x)), size(A, 1))
            Afun.fc(y, x)
        else
            y = Afun.fc(x)
        end
        return y
    elseif issymmetric(Afun) # but !isreal(A), Afun.f can be used
        x = conj(x)
        if ismutating(Afun)
            y = similar(x, promote_type(eltype(A), eltype(x)), size(A, 1))
            Afun.f(y, x)
        else
            y = Afun.f(x)
        end
        conj!(y)
        return y
    else
        error("adjoint not implemented for $A")
    end
end
function Base.:(*)(A::TransposeMap{<:Any,<:FunctionMap}, x::AbstractVector)
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
        end
        conj!(y)
        return y
    else
        error("transpose not implemented for $A")
    end
end

Base.@propagate_inbounds function LinearAlgebra.mul!(y::AbstractVector, A::FunctionMap, x::AbstractVector)
    @boundscheck check_dim_mul(y, A, x)
    ismutating(A) ? A.f(y, x) : copyto!(y, A.f(x))
    return y
end

Base.@propagate_inbounds function LinearAlgebra.mul!(y::AbstractVector, transA::TransposeMap{<:Any,<:FunctionMap}, x::AbstractVector)
    A = transA.lmap
    issymmetric(A) && return mul!(y, A, x)
    @boundscheck check_dim_mul(y, transA, x)
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
        x = conj(x)
        A_mul_B!(y, A, x)
        conj!(y)
        return y
    else
        error("transpose not implemented for $A")
    end
end

Base.@propagate_inbounds function LinearAlgebra.mul!(y::AbstractVector, adjA::AdjointMap{<:Any,<:FunctionMap}, x::AbstractVector)
    A = adjA.lmap
    ishermitian(A) && return mul!(y, A, x)
    @boundscheck check_dim_mul(y, adjA, x)
    if A.fc !== nothing
        ismutating(A) ? A.fc(y, x) : copyto!(y, A.fc(x))
        return y
    elseif issymmetric(A) # but !isreal(A)
        x = conj(x)
        A_mul_B!(y, A, x)
        conj!(y)
        return y
    else
        error("adjoint not implemented for $A")
    end
end
