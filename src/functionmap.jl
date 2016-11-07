immutable FunctionMap{T,F1,F2}<:AbstractLinearMap{T}
    f::F1
    fc::F2
    M::Int
    N::Int
    _ismutating::Bool
    _issymmetric::Bool
    _ishermitian::Bool
    _isposdef::Bool
end

# additional constructor
FunctionMap{T}(f, M::Int, ::Type{T} = Float64; kwargs...) = FunctionMap(f, nothing, M, M, T; kwargs...)
FunctionMap{T}(f, M::Int, N::Int, ::Type{T} = Float64; kwargs...) = FunctionMap(f, nothing, M, N, T; kwargs...)
FunctionMap{T}(f, fc, M::Int, ::Type{T} = Float64; kwargs...) = FunctionMap(f, nothing, M, M, T; kwargs...)

function FunctionMap{T,F1,F2}(f::F1, fc::F2, M::Int, N::Int, ::Type{T} = Float64;
    ismutating::Bool=false, isreal::Bool=T<:Real, issymmetric::Bool=false, ishermitian::Bool=(isreal && issymmetric), isposdef::Bool=false)
    FunctionMap{T,F1,F2}(f, fc, M, N, ismutating, issymmetric, ishermitian, isposdef)
end
function (::Type{FunctionMap{T}}){T,F1,F2}(f::F1, fc::F2, M::Int, N::Int;
    ismutating::Bool=false, isreal::Bool=T<:Real, issymmetric::Bool=false, ishermitian::Bool=(isreal && issymmetric), isposdef::Bool=false)
    FunctionMap{T,F1,F2}(f, fc, M, N, ismutating, issymmetric, ishermitian, isposdef)
end

# show
function Base.show{T}(io::IO,A::FunctionMap{T})
    print(io,"FunctionMap{$T}($(A.f), $(A.fc), $(A.M), $(A.N); ismutating=$(A._ismutating), issymmetric=$(A._issymmetric), ishermitian=$(A._ishermitian), isposdef=$(A._isposdef))")
end

# properties
Base.size(A::FunctionMap,n) = (n==1 ? A.M : (n==2 ? A.N : error("AbstractLinearMap objects have only 2 dimensions")))
Base.size(A::FunctionMap) = (A.M, A.N)
Base.issymmetric(A::FunctionMap) = A._issymmetric
Base.ishermitian(A::FunctionMap) = A._ishermitian
Base.isposdef(A::FunctionMap) = A._isposdef

# multiplication with vector
function Base.A_mul_B!(y::AbstractVector, A::FunctionMap, x::AbstractVector)
    (length(x) == A.N && length(y) == A.M) || throw(DimensionMismatch())
    A._ismutating ? A.f(y,x) : copy!(y,A.f(x))
    return y
end
function *(A::FunctionMap, x::AbstractVector)
    length(x) == A.N || throw(DimensionMismatch())
    A._ismutating ? A.f(similar(x, promote_type(eltype(A), eltype(x)), A.M), x) : A.f(x)
end

function Base.At_mul_B!(y::AbstractVector, A::FunctionMap, x::AbstractVector)
    A._issymmetric && return Base.A_mul_B!(y, A, x)
    (length(x) == A.M && length(y) == A.N) || throw(DimensionMismatch())
    if A.fc != nothing
        if !A._isreal
            x = conj(x)
        end
        (A._ismutating ? A.fc(y,x) : copy!(y,A.fc(x)))
        if !A._isreal
            conj!(y)
        end
        return y
    else
        error("transpose not implemented for $A")
    end
end
function Base.At_mul_B(A::FunctionMap, x::AbstractVector)
    A._issymmetric && return A*x
    length(x) == A.M || throw(DimensionMismatch())
    if A.fc != nothing
        if !A._isreal
            x = conj(x)
        end
        y = (A._ismutating ? A.fc(similar(x, promote_type(eltype(A), eltype(x)), A.N), x) : A.fc(x))
        if !A._isreal
            conj!(y)
        end
        return y
    else
        error("transpose not implemented for $A")
    end
end

function Base.Ac_mul_B!(y::AbstractVector, A::FunctionMap, x::AbstractVector)
    A._ishermitian && return Base.A_mul_B!(y,A,x)
    (length(x) == A.M && length(y) == A.N) || throw(DimensionMismatch())
    if A.fc != nothing
        return (A._ismutating ? A.fc(y, x) : copy!(y, A.fc(x)))
    else
        error("ctranspose not implemented for $A")
    end
end
function Base.Ac_mul_B(A::FunctionMap, x::AbstractVector)
    A._ishermitian && return A*x
    length(x) == A.M || throw(DimensionMismatch())
    if A.fc != nothing
        return (A._ismutating ? A.fc(similar(x, promote_type(eltype(A), eltype(x)), A.N), x) : A.fc(x))
    else
        error("ctranspose not implemented for $A")
    end
end
