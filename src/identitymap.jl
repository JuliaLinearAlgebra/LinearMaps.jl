struct IdentityMap{T} <: LinearMap{T} # T will be determined from maps to which I is added
    M::Int
end
IdentityMap(T::Type, M::Int) = IdentityMap{T}(M)
IdentityMap(T::Type, M::Int, N::Int) = (M==N ? IdentityMap{T}(M) : error("IdentityMap needs to be square"))
IdentityMap(T::Type, sz::Tuple{Int, Int}) = (sz[1]==sz[2] ? IdentityMap{T}(sz[1]) : error("IdentityMap needs to be square"))
IdentityMap(M::Int) = IdentityMap(Int8, M)
IdentityMap(M::Int, N::Int) = IdentityMap(Int8, M, N)
IdentityMap(sz::Tuple{Int, Int}) = IdentityMap(Int8, sz)

# properties
Base.size(A::IdentityMap) = (A.M, A.M)
Base.isreal(::IdentityMap) = true
LinearAlgebra.issymmetric(::IdentityMap) = true
LinearAlgebra.ishermitian(::IdentityMap) = true
LinearAlgebra.isposdef(::IdentityMap) = true

# multiplication with vector
A_mul_B!(y::AbstractVector, A::IdentityMap, x::AbstractVector) =
    (length(x)==length(y)==A.M ? copyto!(y, x) : throw(DimensionMismatch("A_mul_B!")))
Base.:(*)(A::IdentityMap, x::AbstractVector) = x

At_mul_B!(y::AbstractVector, A::IdentityMap, x::AbstractVector) =
    (length(x)==length(y)==A.M ? copyto!(y, x) : throw(DimensionMismatch("At_mul_B!")))

Ac_mul_B!(y::AbstractVector, A::IdentityMap, x::AbstractVector) =
    (length(x)==length(y)==A.M ? copyto!(y, x) : throw(tMismatch("Ac_mul_B!")))

# combine LinearMap and UniformScaling objects in linear combinations
Base.:(+)(A1::LinearMap, A2::UniformScaling{T}) where {T} = A1 + A2[1,1] * IdentityMap{T}(size(A1, 1))
Base.:(+)(A1::UniformScaling{T}, A2::LinearMap) where {T} = A1[1,1] * IdentityMap{T}(size(A2, 1)) + A2
Base.:(-)(A1::LinearMap, A2::UniformScaling{T}) where {T} = A1 - A2[1,1] * IdentityMap{T}(size(A1, 1))
Base.:(-)(A1::UniformScaling{T}, A2::LinearMap) where {T} = A1[1,1] * IdentityMap{T}(size(A2, 1)) - A2
