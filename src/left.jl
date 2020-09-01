# left.jl
# "special cases" related left vector multiplication like x = y'*A
# The subtlety is that y' is a AdjointAbsVec or a TransposeAbsVec
# which is a subtype of AbstractMatrix but it is really "vector like"
# so we want handle it like a vector (Issue#99)
# So this is an exception to the left multiplication rule by a AbstractMatrix
# that usually makes a WrappedMap.

# x = y'*A ⇐⇒ x' = (A'*y)
"""
    *(x::LinearAlgebra.AdjointAbsVec, A::LinearMap)::AdjointAbsVec

Compute the right-action of the linear map `A` on the adjoint vector `x`
and return an adjoint vector.

## Examples
```jldoctest; setup=(using LinearAlgebra, LinearMaps)
julia> A=LinearMap([1.0 2.0; 3.0 4.0]); x=[1.0, 1.0]; x'A
1×2 Adjoint{Float64,Array{Float64,1}}:
 4.0  6.0
```
"""
Base.:(*)(y::LinearAlgebra.AdjointAbsVec, A::LinearMap) = adjoint(A' * y')

"""
    *(x::LinearAlgebra.TransposeAbsVec, A::LinearMap)::TransposeAbsVec

Compute the right-action of the linear map `A` on the transpose vector `x`
and return a transpose vector.

## Examples
```jldoctest; setup=(using LinearAlgebra, LinearMaps)
julia> A=LinearMap([1.0 2.0; 3.0 4.0]); x=[1.0, 1.0]; transpose(x)*A
1×2 Transpose{Float64,Array{Float64,1}}:
 4.0  6.0
```
"""
Base.:(*)(y::LinearAlgebra.TransposeAbsVec, A::LinearMap) = transpose(transpose(A) * transpose(y))

# multiplication with vector/matrix
const AdjointAbsVecOrMat{T} = Adjoint{T,<:AbstractVecOrMat}
const TransposeAbsVecOrMat{T} = Transpose{T,<:AbstractVecOrMat}

function mul!(x::AbstractMatrix, y::AdjointAbsVecOrMat, A::LinearMap)
    check_dim_mul(x, y, A)
    mul!(x', A', y')
    return x
end

function mul!(x::AbstractMatrix, y::AdjointAbsVecOrMat, A::LinearMap, α::Number, β::Number)
    check_dim_mul(x, y, A)
    mul!(x', conj(α)*A', y', true, conj(β))
    return x
end

function mul!(x::AbstractMatrix, y::TransposeAbsVecOrMat, A::LinearMap)
    check_dim_mul(x, y, A)
    mul!(transpose(x), transpose(A), transpose(y))
    return x
end

function mul!(x::AbstractMatrix, y::TransposeAbsVecOrMat, A::LinearMap, α::Number, β::Number)
    check_dim_mul(x, y, A)
    mul!(transpose(x), α*transpose(A), transpose(y), true, β)
    return x
end
