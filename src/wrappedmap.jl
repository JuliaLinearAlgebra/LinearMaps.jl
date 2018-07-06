struct WrappedMap{T, A<:Union{AbstractMatrix, LinearMap}} <: LinearMap{T}
    lmap::A
    _issymmetric::Bool
    _ishermitian::Bool
    _isposdef::Bool
end
function (::Type{WrappedMap})(lmap::Union{AbstractMatrix{T}, LinearMap{T}};
    issymmetric::Bool = issymmetric(lmap),
    ishermitian::Bool = ishermitian(lmap),
    isposdef::Bool = isposdef(lmap)) where {T}
    WrappedMap{T,typeof(lmap)}(lmap, issymmetric, ishermitian, isposdef)
end
function (::Type{WrappedMap{T}})(lmap::Union{AbstractMatrix, LinearMap};
    issymmetric::Bool = issymmetric(lmap),
    ishermitian::Bool = ishermitian(lmap),
    isposdef::Bool = isposdef(lmap)) where {T}
    WrappedMap{T,typeof(lmap)}(lmap, issymmetric, ishermitian, isposdef)
end

# properties
Base.size(A::WrappedMap) = size(A.lmap)
LinearAlgebra.issymmetric(A::WrappedMap) = A._issymmetric
LinearAlgebra.ishermitian(A::WrappedMap) = A._ishermitian
LinearAlgebra.isposdef(A::WrappedMap) = A._isposdef

# multiplication with vector
A_mul_B!(y::AbstractVector, A::WrappedMap, x::AbstractVector) = A_mul_B!(y, A.lmap, x)
*(A::WrappedMap, x::AbstractVector) = *(A.lmap, x)

At_mul_B!(y::AbstractVector, A::WrappedMap, x::AbstractVector) =
    (issymmetric(A) || (isreal(A) && ishermitian(A))) ? A_mul_B!(y, A.lmap, x) : At_mul_B!(y, A.lmap, x)
At_mul_B(A::WrappedMap, x::AbstractVector) =
    (issymmetric(A) || (isreal(A) && ishermitian(A))) ? *(A.lmap, x) : At_mul_B(A.lmap, x)

Ac_mul_B!(y::AbstractVector, A::WrappedMap, x::AbstractVector) =
    ishermitian(A) ? A_mul_B!(y, A.lmap, x) : Ac_mul_B!(y, A.lmap, x)
Ac_mul_B(A::WrappedMap, x::AbstractVector) =
    ishermitian(A) ? *(A.lmap, x) : Ac_mul_B(A.lmap, x)

# combine LinearMap and Matrix objects: linear combinations and map composition
Base.:(+)(A1::LinearMap, A2::AbstractMatrix) = +(A1, WrappedMap(A2))
Base.:(+)(A1::AbstractMatrix, A2::LinearMap) = +(WrappedMap(A1), A2)
Base.:(-)(A1::LinearMap, A2::AbstractMatrix) = -(A1, WrappedMap(A2))
Base.:(-)(A1::AbstractMatrix, A2::LinearMap) = -(WrappedMap(A1), A2)

Base.:(*)(A1::LinearMap, A2::AbstractMatrix) = *(A1, WrappedMap(A2))
Base.:(*)(A1::AbstractMatrix, A2::LinearMap) = *(WrappedMap(A1) ,A2)
