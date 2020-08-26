# left.jl
# "special cases" related left vector multiplication like x = y'*A
# The subtlety is that y' is a AdjointAbsVec or a TransposeAbsVec
# which is a subtype of AbstractMatrix but it is really "vector like"
# so we want handle it like a vector (Issue#99)
# So this is an exception to the left multiplication rule by a AbstractMatrix
# that usually makes a WrappedMap.

# x = y'*A ⇐⇒ x' = (A'*y)
Base.:(*)(y::LinearAlgebra.AdjointAbsVec, A::LinearMap) = adjoint(A' * y')
Base.:(*)(y::LinearAlgebra.TransposeAbsVec, A::LinearMap) = transpose(transpose(A) * transpose(y))

# multiplication with vector/matrix
const AdjointAbsVecOrMat{T} = Adjoint{T,<:AbstractVecOrMat}
const TransposeAbsVecOrMat{T} = Transpose{T,<:AbstractVecOrMat}

@propagate_inbounds function mul!(x::AbstractMatrix, y::AdjointAbsVecOrMat, A::LinearMap)
    @boundscheck check_dim_mul(x, y, A)
    @inbounds mul!(x', A', y')
    return x
end

@propagate_inbounds function mul!(x::AbstractMatrix, y::AdjointAbsVecOrMat, A::LinearMap, α::Number, β::Number)
    @boundscheck check_dim_mul(x, y, A)
    @inbounds mul!(x', conj(α)*A', y', true, conj(β))
    return x
end

@propagate_inbounds function mul!(x::AbstractMatrix, y::TransposeAbsVecOrMat, A::LinearMap)
    @boundscheck check_dim_mul(x, y, A)
    @inbounds mul!(transpose(x), transpose(A), transpose(y))
    return x
end

@propagate_inbounds function mul!(x::AbstractMatrix, y::TransposeAbsVecOrMat, A::LinearMap, α::Number, β::Number)
    @boundscheck check_dim_mul(x, y, A)
    @inbounds mul!(transpose(x), α*transpose(A), transpose(y), true, β)
    return x
end
