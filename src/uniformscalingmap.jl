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
Base.:(*)(A::UniformScalingMap, x::AbstractVector) = A.λ * x
# call of LinearAlgebra.generic_mul! since order of arguments in mul! in stdlib/LinearAlgebra/src/generic.jl
# TODO: either leave it as is or use mul! (and lower bound on version) once fixed in LinearAlgebra
function A_mul_B!(y::AbstractVector, A::UniformScalingMap, x::AbstractVector)
    (length(x) == length(y) == A.M || throw(DimensionMismatch("A_mul_B!")))
    if iszero(A.λ)
        return fill!(y, 0)
    elseif isone(A.λ)
        return copyto!(y, x)
    else
        return LinearAlgebra.generic_mul!(y, A.λ, x)
    end
end

function At_mul_B!(y::AbstractVector, A::UniformScalingMap, x::AbstractVector)
    (length(x) == length(y) == A.M || throw(DimensionMismatch("At_mul_B!")))
    if iszero(A.λ)
        return fill!(y, 0)
    elseif isone(A.λ)
        return copyto!(y, x)
    else
        return LinearAlgebra.generic_mul!(y, A.λ, x)
    end
end

function Ac_mul_B!(y::AbstractVector, A::UniformScalingMap, x::AbstractVector)
    (length(x) == length(y) == A.M || throw(DimensionMismatch("At_mul_B!")))
    if iszero(A.λ)
        return fill!(y, 0)
    elseif isone(A.λ)
        return copyto!(y, x)
    else
        return LinearAlgebra.generic_mul!(y, conj(A.λ), x)
    end
end

# combine LinearMap and UniformScaling objects in linear combinations
Base.:(+)(A1::LinearMap{T}, A2::UniformScaling) where {T} =
    A1 + UniformScalingMap(convert(T, A2[1,1]), size(A1, 1))
Base.:(+)(A1::UniformScaling, A2::LinearMap{T}) where {T} =
    UniformScalingMap(convert(T, A1[1,1]), size(A2, 1)) + A2
Base.:(-)(A1::LinearMap{T}, A2::UniformScaling) where {T} =
    A1 - UniformScalingMap(convert(T, A2[1,1]), size(A1, 1))
Base.:(-)(A1::UniformScaling, A2::LinearMap{T}) where {T} =
    UniformScalingMap(convert(T, A1[1,1]), size(A2, 1)) - A2
