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
const TransposeAbsVecOrMat{T} = Transpose{T,<:AbstractVecOrMat}

# handles both y::AbstractMatrix and y::AdjointAbsVecOrMat
"""
    mul!(C::AbstractMatrix, A::AbstractMatrix, B::LinearMap) -> C

Calculates the matrix representation of `A*B` and stores the result in `C`,
overwriting the existing value of `C`. Note that `C` must not be aliased with
either `A` or `B`. The computation `C = A*B` is performed via `C' = B'A'`.

## Examples
```jldoctest; setup=(using LinearAlgebra, LinearMaps)
julia> A=[1.0 1.0; 1.0 1.0]; B=LinearMap([1.0 2.0; 3.0 4.0]); C = similar(A); mul!(C, A, B);

julia> C
2×2 Array{Float64,2}:
 4.0  6.0
 4.0  6.0
```
"""
function mul!(X::AbstractMatrix, Y::AbstractMatrix, A::LinearMap)
    check_dim_mul(X, Y, A)
    _unsafe_mul!(X', A', Y')
    return X
end

function mul!(X::AbstractMatrix, Y::TransposeAbsVecOrMat, A::LinearMap)
    check_dim_mul(X, Y, A)
    _unsafe_mul!(transpose(X), transpose(A), transpose(Y))
    return X
end

# commutative case, handles both the abstract and adjoint case
function mul!(X::AbstractMatrix{<:RealOrComplex}, Y::AbstractMatrix{<:RealOrComplex}, A::LinearMap{<:RealOrComplex},
                α::RealOrComplex, β::RealOrComplex)
    check_dim_mul(X, Y, A)
    _unsafe_mul!(X', A', Y', conj(α), conj(β))
    return X
end

function mul!(X::AbstractMatrix{<:RealOrComplex}, Y::TransposeAbsVecOrMat{<:RealOrComplex}, A::LinearMap{<:RealOrComplex},
                α::RealOrComplex, β::RealOrComplex)
    check_dim_mul(X, Y, A)
    _unsafe_mul!(transpose(X), transpose(A), transpose(Y), α, β)
    return X
end

# non-commutative case
function mul!(X::AbstractMatrix, Y::AbstractMatrix, A::LinearMap, α::Number, β::Number)
    check_dim_mul(X, Y, A)
    if iszero(β)
        _unsafe_mul!(X', α*A', Y')
    else
        !isone(β) && rmul!(X, β)
        _unsafe_mul!(X', conj(α)*A', Y', true, true)
    end
    return X
end
