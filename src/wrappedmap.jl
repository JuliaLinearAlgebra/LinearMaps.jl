struct WrappedMap{T, A<:Union{AbstractMatrix, LinearMap}} <: LinearMap{T}
    lmap::A
    _issymmetric::Bool
    _ishermitian::Bool
    _isposdef::Bool
end
function (::Type{WrappedMap})(lmap::Union{AbstractMatrix{T}, LinearMap{T}};
    issymmetric::Bool = Base.issymmetric(lmap), ishermitian::Bool = Base.ishermitian(lmap),
    isposdef::Bool = Base.isposdef(lmap)) where {T}
    WrappedMap{T,typeof(lmap)}(lmap, issymmetric, ishermitian, isposdef)
end
function (::Type{WrappedMap{T}})(lmap::Union{AbstractMatrix, LinearMap};
    issymmetric::Bool = Base.issymmetric(lmap), ishermitian::Bool = Base.ishermitian(lmap),
    isposdef::Bool = Base.isposdef(lmap)) where {T}
    WrappedMap{T,typeof(lmap)}(lmap, issymmetric, ishermitian, isposdef)
end

# properties
Base.size(A::WrappedMap) = size(A.lmap)
Base.issymmetric(A::WrappedMap) = A._issymmetric
Base.ishermitian(A::WrappedMap) = A._ishermitian
Base.isposdef(A::WrappedMap) = A._isposdef

# multiplication with vector
Base.A_mul_B!(y::AbstractVector, A::WrappedMap, x::AbstractVector) = Base.A_mul_B!(y, A.lmap, x)
*(A::WrappedMap, x::AbstractVector) = *(A.lmap, x)

Base.At_mul_B!(y::AbstractVector, A::WrappedMap, x::AbstractVector) =
    (issymmetric(A) || (isreal(A) && ishermitian(A))) ? Base.A_mul_B!(y, A.lmap, x) : Base.At_mul_B!(y, A.lmap, x)
Base.At_mul_B(A::WrappedMap, x::AbstractVector) =
    (issymmetric(A) || (isreal(A) && ishermitian(A))) ? *(A.lmap, x) : Base.At_mul_B(A.lmap, x)

Base.Ac_mul_B!(y::AbstractVector, A::WrappedMap, x::AbstractVector) =
    ishermitian(A) ? Base.A_mul_B!(y, A.lmap, x) : Base.Ac_mul_B!(y, A.lmap, x)
Base.Ac_mul_B(A::WrappedMap, x::AbstractVector) =
    ishermitian(A) ? *(A.lmap, x) : Base.Ac_mul_B(A.lmap, x)

# combine LinearMap and Matrix objects: linear combinations and map composition
+(A1::LinearMap, A2::AbstractMatrix) = +(A1, WrappedMap(A2))
+(A1::AbstractMatrix, A2::LinearMap) = +(WrappedMap(A1), A2)
-(A1::LinearMap, A2::AbstractMatrix) = -(A1, WrappedMap(A2))
-(A1::AbstractMatrix, A2::LinearMap) = -(WrappedMap(A1), A2)

*(A1::LinearMap, A2::AbstractMatrix) = *(A1, WrappedMap(A2))
*(A1::AbstractMatrix, A2::LinearMap) = *(WrappedMap(A1) ,A2)
