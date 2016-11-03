immutable TransposeMap{T, A<:AbstractLinearMap} <: AbstractLinearMap{T}
    lmap::A
end
immutable CTransposeMap{T, A<:AbstractLinearMap} <: AbstractLinearMap{T}
    lmap::A
end
(::Type{TransposeMap}){T}(lmap::AbstractLinearMap{T}) = TransposeMap{T,typeof(lmap)}(lmap)
(::Type{CTransposeMap}){T}(lmap::AbstractLinearMap{T}) = CTransposeMap{T,typeof(lmap)}(lmap)

# transposition behavior of AbstractLinearMap objects
Base.transpose(A::TransposeMap)=A.lmap
Base.ctranspose(A::CTransposeMap)=A.lmap

Base.transpose(A::AbstractLinearMap) = TransposeMap(A)
Base.ctranspose{T<:Real}(A::AbstractLinearMap{T}) = transpose(A)
Base.ctranspose(A::AbstractLinearMap) = CTransposeMap(A)

# properties
Base.size(A::Union{TransposeMap,CTransposeMap},n) = (n==1 ? size(A.lmap,2) :
    (n==2 ? size(A.lmap,1) : error("AbstractLinearMap objects have only 2 dimensions")))
Base.size(A::Union{TransposeMap,CTransposeMap}) = (size(A.lmap,2), size(A.lmap,1))
Base.isreal(A::Union{TransposeMap,CTransposeMap}) = isreal(A.lmap)
Base.issymmetric(A::Union{TransposeMap,CTransposeMap}) = issymmetric(A.lmap)
Base.ishermitian(A::Union{TransposeMap,CTransposeMap}) = ishermitian(A.lmap)
Base.isposdef(A::Union{TransposeMap,CTransposeMap}) = isposdef(A.lmap)

# comparison of TransposeMap objects
==(A::TransposeMap, B::TransposeMap) = A.lmap == B.lmap
==(A::CTransposeMap, B::CTransposeMap) = A.lmap == B.lmap
==(A::TransposeMap, B::AbstractLinearMap) = issymmetric(B) && A.lmap == B
==(A::CTransposeMap, B::AbstractLinearMap) = ishermitian(B) && A.lmap == B
==(A::AbstractLinearMap, B::TransposeMap) = issymmetric(A) && B.lmap == A
==(A::AbstractLinearMap, B::CTransposeMap) = ishermitian(A) && B.lmap == A

# multiplication with vector
Base.A_mul_B!(y::AbstractVector, A::TransposeMap, x::AbstractVector) =
    (issymmetric(A.lmap) ? Base.A_mul_B!(y,A.lmap,x) : Base.At_mul_B!(y,A.lmap,x))
*(A::TransposeMap, x::AbstractVector) =
    (issymmetric(A.lmap) ? *(A.lmap,x) : Base.At_mul_B(A.lmap,x))

Base.At_mul_B!(y::AbstractVector, A::TransposeMap, x::AbstractVector) =
    Base.A_mul_B!(y,A.lmap,x)
Base.At_mul_B(A::TransposeMap, x::AbstractVector) =
    *(A.lmap,x)

Base.Ac_mul_B!(y::AbstractVector, A::TransposeMap, x::AbstractVector) =
    isreal(A.lmap) ? Base.A_mul_B!(y,A.lmap,x) : (Base.A_mul_B!(y,A.lmap,conj(x));conj!(y))
Base.Ac_mul_B(A::TransposeMap, x::AbstractVector) = isreal(A.lmap) ? *(A.lmap,x) : conj!(*(A.lmap,conj(x)))

Base.A_mul_B!(y::AbstractVector, A::CTransposeMap, x::AbstractVector) =
    (ishermitian(A.lmap) ? Base.A_mul_B!(y,A.lmap,x) : Base.Ac_mul_B!(y,A.lmap,x))
*(A::CTransposeMap,x::AbstractVector) = (ishermitian(A.lmap) ? *(A.lmap,x) : Base.Ac_mul_B(A.lmap,x))

Base.At_mul_B!(y::AbstractVector, A::CTransposeMap, x::AbstractVector) =
    isreal(A.lmap) ? Base.A_mul_B!(y,A.lmap,x) : (Base.A_mul_B!(y,A.lmap,conj(x));conj!(y))
Base.At_mul_B(A::CTransposeMap, x::AbstractVector) =
    isreal(A.lmap) ? *(A.lmap,x) : conj!(*(A.lmap,conj(x)))

Base.Ac_mul_B!(y::AbstractVector, A::CTransposeMap, x::AbstractVector) = Base.A_mul_B!(y,A.lmap,x)
Base.Ac_mul_B(A::CTransposeMap, x::AbstractVector) = *(A.lmap,x)
