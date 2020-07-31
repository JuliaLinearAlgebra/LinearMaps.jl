# left.jl
# "special cases" related left vector multiplication like x = y'*A
# The subtlety is that y' is a AdjointAbsVec or a TransposeAbsVec
# which is a subtype of AbstractMatrix but it is really "vector like"
# so we want handle it like a vector (Issue#99)
# So this is an exception to the left multiplication rule by a AbstractMatrix
# that usually makes a WrappedMap.
# The "transpose" versions may be of dubious use, but the adjoint versions
# are useful for ensuring that (y'A)*x ≈ y'*(A*x) are both scalars.

import LinearAlgebra: AdjointAbsVec, TransposeAbsVec

# x = y'*A ⇐⇒ x' = (A'*y)
Base.:(*)(y::AdjointAbsVec, A::LinearMap) = adjoint(*(A', y'))
Base.:(*)(y::TransposeAbsVec, A::LinearMap) = transpose(transpose(A) * transpose(y))

# mul!(x, y', A)
Base.@propagate_inbounds function LinearAlgebra.mul!(x::AbstractMatrix, y::AdjointAbsVec, A::LinearMap)
    @boundscheck check_dim_mul(x, y, A)
    @inbounds mul!(adjoint(x), A', y')
    return adjoint(x)
end

Base.@propagate_inbounds function LinearAlgebra.mul!(x::AbstractMatrix, y::AdjointAbsVec,
        A::LinearMap, α::Number, β::Number)
    @boundscheck check_dim_mul(x, y, A)
    @inbounds mul!(adjoint(x), A', y', α, β)
    return adjoint(x)
end

Base.@propagate_inbounds function LinearAlgebra.mul!(x::AbstractMatrix, y::TransposeAbsVec, A::LinearMap)
    @boundscheck check_dim_mul(x, y, A)
    @inbounds mul!(transpose(x), transpose(A), transpose(y))
    return transpose(x)
end

Base.@propagate_inbounds function LinearAlgebra.mul!(x::AbstractMatrix, y::TransposeAbsVec,
         A::LinearMap, α::Number, β::Number)
    @boundscheck check_dim_mul(x, y, A)
    @inbounds mul!(transpose(x), transpose(A), transpose(y), α, β)
    return transpose(x)
end
