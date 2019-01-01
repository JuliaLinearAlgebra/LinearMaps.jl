struct UniformScalingMap{T} <: LinearMap{T} # T will be determined from maps to which this is added
    λ::T
    M::Int
end
UniformScalingMap(λ::T, M::Int, N::Int) where {T} =
    (M == N ? UniformScalingMap(λ, M) : error("UniformScalingMap needs to be square"))
UniformScalingMap(λ::T, sz::Tuple{Int, Int}) where {T} =
    (sz[1] == sz[2] ? UniformScalingMap(λ, sz[1]) : error("UniformScalingMap needs to be square"))

# properties
Base.size(A::UniformScalingMap) = (A.M, A.M)
Base.isreal(A::UniformScalingMap) = isreal(A.λ)
LinearAlgebra.issymmetric(::UniformScalingMap) = true
LinearAlgebra.ishermitian(A::UniformScalingMap) = isreal(A)
LinearAlgebra.isposdef(A::UniformScalingMap) = isposdef(A.λ)

# special transposition behavior
LinearAlgebra.transpose(A::UniformScalingMap) = A
LinearAlgebra.adjoint(A::UniformScalingMap)   = UniformScalingMap(conj(A.λ), size(A))

# multiplication with vector
A_mul_B!(y::AbstractVector, A::UniformScalingMap, x::AbstractVector) =
    (length(x) == length(y) == A.M ? mul!(y, A.λ, x) : throw(DimensionMismatch("A_mul_B!")))
Base.:(*)(A::UniformScalingMap, x::AbstractVector) = A.λ * x

At_mul_B!(y::AbstractVector, A::UniformScalingMap, x::AbstractVector) =
    (length(x) == length(y) == A.M ? mul!(y, A.λ, x) : throw(DimensionMismatch("At_mul_B!")))

Ac_mul_B!(y::AbstractVector, A::UniformScalingMap, x::AbstractVector) =
    (length(x) == length(y) == A.M ? mul!(y, conj(A.λ), x) : throw(DimensionMismatch("Ac_mul_B!")))

# combine LinearMap and UniformScaling objects in linear combinations
Base.:(+)(A1::LinearMap, A2::UniformScaling{T}) where {T} = A1 + UniformScalingMap{T}(A2[1,1], size(A1, 1))
Base.:(+)(A1::UniformScaling{T}, A2::LinearMap) where {T} = UniformScalingMap{T}(A1[1,1], size(A2, 1)) + A2
Base.:(-)(A1::LinearMap, A2::UniformScaling{T}) where {T} = A1 - UniformScalingMap{T}(A2[1,1], size(A1, 1))
Base.:(-)(A1::UniformScaling{T}, A2::LinearMap) where {T} = UniformScalingMap{T}(A1[1,1], size(A2, 1)) - A2
