struct TransposeMap{T, A<:LinearMap{T}} <: LinearMap{T}
    lmap::A
end
struct AdjointMap{T, A<:LinearMap{T}} <: LinearMap{T}
    lmap::A
end

TransposeMap(lmap::LinearMap{T}) where {T} = TransposeMap{T, typeof(lmap)}(lmap)
AdjointMap(lmap::LinearMap{T}) where {T}   = AdjointMap{T, typeof(lmap)}(lmap)

# transposition behavior of LinearMap objects
transpose(A::TransposeMap) = A.lmap
adjoint(A::AdjointMap) = A.lmap

transpose(A::LinearMap) = TransposeMap(A)
adjoint(A::LinearMap{<:Real}) = transpose(A)
adjoint(A::LinearMap) = AdjointMap(A)

# properties
Base.size(A::Union{TransposeMap,AdjointMap}) = (size(A.lmap,2), size(A.lmap,1))
issymmetric(A::Union{TransposeMap,AdjointMap}) = issymmetric(A.lmap)
ishermitian(A::Union{TransposeMap,AdjointMap}) = ishermitian(A.lmap)
isposdef(A::Union{TransposeMap,AdjointMap}) = isposdef(A.lmap)

# comparison of TransposeMap objects
==(A::TransposeMap, B::TransposeMap)    = A.lmap == B.lmap
==(A::AdjointMap, B::AdjointMap)        = A.lmap == B.lmap
==(A::TransposeMap, B::AdjointMap)      = isreal(B) && A.lmap == B.lmap
==(A::AdjointMap, B::TransposeMap)      = isreal(A) && A.lmap == B.lmap
==(A::TransposeMap, B::LinearMap)       = issymmetric(B) && A.lmap == B
==(A::AdjointMap, B::LinearMap)         = ishermitian(B) && A.lmap == B
==(A::LinearMap, B::TransposeMap)       = issymmetric(A) && B.lmap == A
==(A::LinearMap, B::AdjointMap)         = ishermitian(A) && B.lmap == A

# multiplication with vector
A_mul_B!(y::AbstractVector, A::TransposeMap, x::AbstractVector) =
    (issymmetric(A.lmap) ? A_mul_B!(y, A.lmap, x) : At_mul_B!(y, A.lmap, x))

At_mul_B!(y::AbstractVector, A::TransposeMap, x::AbstractVector) = A_mul_B!(y, A.lmap, x)
At_mul_B(A::TransposeMap, x::AbstractVector) = *(A.lmap, x)

Ac_mul_B!(y::AbstractVector, A::TransposeMap, x::AbstractVector) =
    isreal(A.lmap) ? A_mul_B!(y, A.lmap, x) : (A_mul_B!(y, A.lmap, conj(x)); conj!(y))
Ac_mul_B(A::TransposeMap, x::AbstractVector) = isreal(A.lmap) ? *(A.lmap, x) : conj!(*(A.lmap, conj(x)))

A_mul_B!(y::AbstractVector, A::AdjointMap, x::AbstractVector) =
    (ishermitian(A.lmap) ? A_mul_B!(y, A.lmap, x) : Ac_mul_B!(y, A.lmap, x))

At_mul_B!(y::AbstractVector, A::AdjointMap, x::AbstractVector) =
    isreal(A.lmap) ? A_mul_B!(y, A.lmap, x) : (A_mul_B!(y, A.lmap, conj(x)); conj!(y))
At_mul_B(A::AdjointMap, x::AbstractVector) = isreal(A.lmap) ? *(A.lmap, x) : conj!(*(A.lmap, conj(x)))

Ac_mul_B!(y::AbstractVector, A::AdjointMap, x::AbstractVector) = A_mul_B!(y, A.lmap, x)
Ac_mul_B(A::AdjointMap, x::AbstractVector) = *(A.lmap, x)
