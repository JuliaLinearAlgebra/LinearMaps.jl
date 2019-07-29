struct UniformScalingMap{T} <: LinearMap{T}
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
Base.:(*)(A::UniformScalingMap, x::AbstractVector) =
    length(x) == A.M ? A.λ * x : throw(DimensionMismatch("A_mul_B!"))

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
At_mul_B!(y::AbstractVector, A::UniformScalingMap, x::AbstractVector) = A_mul_B!(y, transpose(A), x)
Ac_mul_B!(y::AbstractVector, A::UniformScalingMap, x::AbstractVector) = A_mul_B!(y, adjoint(A), x)

function LinearAlgebra.mul!(y::AbstractVector, J::UniformScalingMap{T}, x::AbstractVector, α::Number=one(T), β::Number=zero(T)) where {T}
    @boundscheck (length(x) == length(y) == J.M || throw(DimensionMismatch("mul!")))
    @inbounds if isone(α)
        if iszero(β)
            A_mul_B!(y, J, x)
            return y
        elseif isone(β)
            iszero(J.λ) && return y
            isone(J.λ) && return y .+= x
            y .+= J.λ .* x
            return y
        else # β != 0, 1
            iszero(J.λ) && (rmul!(y, β); return y)
            isone(J.λ) && (y .= y .* β .+ x; return y)
            y .= y .* β .+ J.λ .* x
            return y
        end
    elseif iszero(α)
        iszero(β) && (fill!(y, zero(eltype(y))); return y)
        isone(β) && return y
        # β != 0, 1
        rmul!(y, β)
        return y
    else # α != 0, 1
        iszero(β) && (y .= J.λ .* x .* α; return y)
        isone(β) && (y .+= J.λ .* x .* α; return y)
        # β != 0, 1
        y .= y .* β .+ J.λ .* x .* α
        return y
    end
end

# combine LinearMap and UniformScaling objects in linear combinations
Base.:(+)(A1::LinearMap, A2::UniformScaling) = A1 + UniformScalingMap(A2.λ, size(A1, 1))
Base.:(+)(A1::UniformScaling, A2::LinearMap) = UniformScalingMap(A1.λ, size(A2, 1)) + A2
Base.:(-)(A1::LinearMap, A2::UniformScaling) = A1 - UniformScalingMap(A2.λ, size(A1, 1))
Base.:(-)(A1::UniformScaling, A2::LinearMap) = UniformScalingMap(A1.λ, size(A2, 1)) - A2
