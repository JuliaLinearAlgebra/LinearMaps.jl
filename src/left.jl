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
"""
    *(::LinearAlgebra.AdjointAbsVec, A::LinearMap)
    *(::LinearAlgebra.TransposeAbsVec, A::LinearMap)

Compute the right-action of the linear map `A` on the adjoint and transpose vector `x`
and return an adjoint and transpose vector, respectively.

## Examples
```jldoctest; setup=(using LinearAlgebra, LinearMaps)
julia> A=LinearMap([1.0 2.0; 3.0 4.0]); x=[1.0, 1.0]; x'A
1×2 Adjoint{Float64,Array{Float64,1}}:
 4.0  6.0
```
"""
Base.:(*)(y::AdjointAbsVec, A::LinearMap) = adjoint(A' * y')
Base.:(*)(y::TransposeAbsVec, A::LinearMap) = transpose(transpose(A) * transpose(y))

# multiplication with vector/matrix
for Atype in (AdjointAbsVec, Adjoint{<:Any,<:AbstractMatrix})
    @eval begin
        Base.@propagate_inbounds function LinearAlgebra.mul!(x::AbstractMatrix, y::$Atype, A::LinearMap)
            @boundscheck check_dim_mul(x, y, A)
            @inbounds mul!(x', A', y')
            return x
        end

        Base.@propagate_inbounds function LinearAlgebra.mul!(x::AbstractMatrix, y::$Atype, A::LinearMap,
                α::Number, β::Number)
            @boundscheck check_dim_mul(x, y, A)
            @inbounds mul!(x', A', y', conj(α), conj(β))
            return x
        end
    end
end
for Atype in (TransposeAbsVec, Transpose{<:Any,<:AbstractMatrix})
    @eval begin
        Base.@propagate_inbounds function LinearAlgebra.mul!(x::AbstractMatrix, y::$Atype, A::LinearMap)
            @boundscheck check_dim_mul(x, y, A)
            @inbounds mul!(transpose(x), transpose(A), transpose(y))
            return x
        end

        Base.@propagate_inbounds function LinearAlgebra.mul!(x::AbstractMatrix, y::$Atype, A::LinearMap,
                α::Number, β::Number)
            @boundscheck check_dim_mul(x, y, A)
            @inbounds mul!(transpose(x), transpose(A), transpose(y), α, β)
            return x
        end
    end
end
