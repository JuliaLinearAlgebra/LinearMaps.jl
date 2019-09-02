struct TransposeMap{T, A<:LinearMap{T}} <: LinearMap{T}
    lmap::A
end
struct AdjointMap{T, A<:LinearMap{T}} <: LinearMap{T}
    lmap::A
end

# transposition behavior of LinearMap objects
LinearAlgebra.transpose(A::TransposeMap) = A.lmap
LinearAlgebra.adjoint(A::AdjointMap) = A.lmap

LinearAlgebra.transpose(A::LinearMap) = TransposeMap(A)
LinearAlgebra.adjoint(A::LinearMap{<:Real}) = transpose(A)
LinearAlgebra.adjoint(A::LinearMap) = AdjointMap(A)

# properties
Base.size(A::Union{TransposeMap, AdjointMap}, n) = n==1 ? size(A.lmap, 2) : n==2 ? size(A.lmap, 1) : error("LinearMap objects have only 2 dimensions")
Base.size(A::Union{TransposeMap, AdjointMap}) = reverse(size(A.lmap))
LinearAlgebra.issymmetric(A::Union{TransposeMap, AdjointMap}) = issymmetric(A.lmap)
LinearAlgebra.ishermitian(A::Union{TransposeMap, AdjointMap}) = ishermitian(A.lmap)
LinearAlgebra.isposdef(A::Union{TransposeMap, AdjointMap}) = isposdef(A.lmap)

# comparison of TransposeMap objects
Base.:(==)(A::TransposeMap, B::TransposeMap) = A.lmap == B.lmap
Base.:(==)(A::AdjointMap, B::AdjointMap)     = A.lmap == B.lmap
Base.:(==)(A::TransposeMap, B::AdjointMap)   = false # isreal(B) && A.lmap == B.lmap # isreal(::AdjointMap) == false
Base.:(==)(A::AdjointMap, B::TransposeMap)   = false # isreal(A) && A.lmap == B.lmap # isreal(::AdjointMap) == false
Base.:(==)(A::TransposeMap, B::LinearMap)    = issymmetric(B) && A.lmap == B
Base.:(==)(A::AdjointMap, B::LinearMap)      = ishermitian(B) && A.lmap == B
Base.:(==)(A::LinearMap, B::TransposeMap)    = issymmetric(A) && B.lmap == A
Base.:(==)(A::LinearMap, B::AdjointMap)      = ishermitian(A) && B.lmap == A

# multiplication with vector
A_mul_B!(y::AbstractVector, A::TransposeMap, x::AbstractVector) =
    issymmetric(A.lmap) ? A_mul_B!(y, A.lmap, x) : At_mul_B!(y, A.lmap, x)

At_mul_B!(y::AbstractVector, A::TransposeMap, x::AbstractVector) = A_mul_B!(y, A.lmap, x)

Ac_mul_B!(y::AbstractVector, A::TransposeMap, x::AbstractVector) =
    isreal(A.lmap) ? A_mul_B!(y, A.lmap, x) : (A_mul_B!(y, A.lmap, conj(x)); conj!(y))

A_mul_B!(y::AbstractVector, A::AdjointMap, x::AbstractVector) =
    ishermitian(A.lmap) ? A_mul_B!(y, A.lmap, x) : Ac_mul_B!(y, A.lmap, x)

At_mul_B!(y::AbstractVector, A::AdjointMap, x::AbstractVector) =
    isreal(A.lmap) ? A_mul_B!(y, A.lmap, x) : (A_mul_B!(y, A.lmap, conj(x)); conj!(y))

Ac_mul_B!(y::AbstractVector, A::AdjointMap, x::AbstractVector) = A_mul_B!(y, A.lmap, x)
