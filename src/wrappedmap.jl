immutable WrappedMap{T,A} <: AbstractLinearMap{T}
    lmap::A
    _isreal::Bool
    _issymmetric::Bool
    _ishermitian::Bool
    _isposdef::Bool
end
(::Type{WrappedMap}){T,A<:Union{AbstractMatrix, AbstractLinearMap}}(lmap::A, ::Type{T} = eltype(lmap);
    isreal::Bool = Base.isreal(lmap), issymmetric::Bool = Base.issymmetric(lmap),
    ishermitian::Bool = Base.ishermitian(lmap), isposdef::Bool = Base.isposdef(lmap)) =
        WrappedMap{T,A}(lmap, isreal, issymmetric, ishermitian, isposdef)

# properties
Base.size(A::WrappedMap) = size(A.lmap)
Base.isreal(A::WrappedMap) = A._isreal
Base.issymmetric(A::WrappedMap) = A._issymmetric
Base.ishermitian(A::WrappedMap) = A._ishermitian
Base.isposdef(A::WrappedMap) = A._isposdef

# comparison
==(A::WrappedMap,B::WrappedMap) = (A.lmap == B.lmap && isreal(A) == isreal(B) &&
    issymmetric(A) == issymmetric(B) && ishermitian(A) == ishermitian(B) && isposdef(A) == isposdef(B))

# multiplication with vector
Base.A_mul_B!(y::AbstractVector, A::WrappedMap,x::AbstractVector) = Base.A_mul_B!(y, A.lmap, x)
*(A::WrappedMap, x::AbstractVector) = *(A.lmap,x)

Base.At_mul_B!(y::AbstractVector, A::WrappedMap,x::AbstractVector) = Base.At_mul_B!(y, A.lmap, x)
Base.At_mul_B(A::WrappedMap,x::AbstractVector) = Base.At_mul_B(A.lmap, x)

Base.Ac_mul_B!(y::AbstractVector,A::WrappedMap,x::AbstractVector) = Base.Ac_mul_B!(y, A.lmap, x)
Base.Ac_mul_B(A::WrappedMap,x::AbstractVector) = Base.Ac_mul_B(A.lmap, x)

# combine AbstractLinearMap and Matrix objects: linear combinations and map composition
+(A1::AbstractLinearMap, A2::AbstractMatrix) = +(A1, WrappedMap(A2))
+(A1::AbstractMatrix, A2::AbstractLinearMap) = +(WrappedMap(A1), A2)
-(A1::AbstractLinearMap, A2::AbstractMatrix) = -(A1, WrappedMap(A2))
-(A1::AbstractMatrix, A2::AbstractLinearMap) = -(WrappedMap(A1), A2)

*(A1::AbstractLinearMap, A2::AbstractMatrix) = *(A1, WrappedMap(A2))
*(A1::AbstractMatrix, A2::AbstractLinearMap) = *(WrappedMap(A1) ,A2)
