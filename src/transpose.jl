struct TransposeMap{T, A<:LinearMap{T}} <: LinearMap{T}
    lmap::A
end
struct AdjointMap{T, A<:LinearMap{T}} <: LinearMap{T}
    lmap::A
end
(::Type{TransposeMap})(lmap::LinearMap{T}) where {T} = TransposeMap{T, typeof(lmap)}(lmap)
(::Type{AdjointMap})(lmap::LinearMap{T}) where {T} = AdjointMap{T, typeof(lmap)}(lmap)

# transposition behavior of LinearMap objects
Base.transpose(A::TransposeMap) = A.lmap
adjoint(A::AdjointMap) = A.lmap

Base.transpose(A::LinearMap) = TransposeMap(A)
adjoint(A::LinearMap{<:Real}) = transpose(A)
adjoint(A::LinearMap) = AdjointMap(A)

# properties
Base.size(A::Union{TransposeMap,AdjointMap}) = (size(A.lmap,2), size(A.lmap,1))
Base.issymmetric(A::Union{TransposeMap,AdjointMap}) = issymmetric(A.lmap)
Base.ishermitian(A::Union{TransposeMap,AdjointMap}) = ishermitian(A.lmap)
Base.isposdef(A::Union{TransposeMap,AdjointMap}) = isposdef(A.lmap)

# comparison of TransposeMap objects
==(A::TransposeMap, B::TransposeMap) = A.lmap == B.lmap
==(A::AdjointMap, B::AdjointMap) = A.lmap == B.lmap
==(A::TransposeMap, B::AdjointMap) = isreal(B) && A.lmap == B.lmap
==(A::AdjointMap, B::TransposeMap) = isreal(A) && A.lmap == B.lmap
==(A::TransposeMap, B::LinearMap) = issymmetric(B) && A.lmap == B
==(A::AdjointMap, B::LinearMap) = ishermitian(B) && A.lmap == B
==(A::LinearMap, B::TransposeMap) = issymmetric(A) && B.lmap == A
==(A::LinearMap, B::AdjointMap) = ishermitian(A) && B.lmap == A

# multiplication with vector
Base.A_mul_B!(y::AbstractVector, A::TransposeMap, x::AbstractVector) =
    (issymmetric(A.lmap) ? Base.A_mul_B!(y, A.lmap, x) : Base.At_mul_B!(y, A.lmap, x))
*(A::TransposeMap, x::AbstractVector) =
    (issymmetric(A.lmap) ? *(A.lmap, x) : Base.At_mul_B(A.lmap, x))

Base.At_mul_B!(y::AbstractVector, A::TransposeMap, x::AbstractVector) = Base.A_mul_B!(y, A.lmap, x)
Base.At_mul_B(A::TransposeMap, x::AbstractVector) = *(A.lmap, x)

Base.Ac_mul_B!(y::AbstractVector, A::TransposeMap, x::AbstractVector) =
    isreal(A.lmap) ? Base.A_mul_B!(y, A.lmap, x) : (Base.A_mul_B!(y, A.lmap, conj(x)); conj!(y))
Base.Ac_mul_B(A::TransposeMap, x::AbstractVector) = isreal(A.lmap) ? *(A.lmap, x) : conj!(*(A.lmap, conj(x)))

Base.A_mul_B!(y::AbstractVector, A::AdjointMap, x::AbstractVector) =
    (ishermitian(A.lmap) ? Base.A_mul_B!(y, A.lmap, x) : Base.Ac_mul_B!(y, A.lmap, x))
*(A::AdjointMap,x::AbstractVector) = (ishermitian(A.lmap) ? *(A.lmap, x) : Base.Ac_mul_B(A.lmap, x))

Base.At_mul_B!(y::AbstractVector, A::AdjointMap, x::AbstractVector) =
    isreal(A.lmap) ? Base.A_mul_B!(y, A.lmap, x) : (Base.A_mul_B!(y, A.lmap, conj(x)); conj!(y))
Base.At_mul_B(A::AdjointMap, x::AbstractVector) =
    isreal(A.lmap) ? *(A.lmap, x) : conj!(*(A.lmap, conj(x)))

Base.Ac_mul_B!(y::AbstractVector, A::AdjointMap, x::AbstractVector) = Base.A_mul_B!(y, A.lmap, x)
Base.Ac_mul_B(A::AdjointMap, x::AbstractVector) = *(A.lmap, x)
