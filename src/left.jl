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
1×2 Adjoint{Float64,Vector{Float64}}:
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
1×2 Transpose{Float64,Vector{Float64}}:
 4.0  6.0
```
"""
Base.:(*)(y::LinearAlgebra.TransposeAbsVec, A::LinearMap) = transpose(transpose(A) * transpose(y))

# multiplication with vector/matrix

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
2×2 Matrix{Float64}:
 4.0  6.0
 4.0  6.0
```
"""
function mul!(X::AbstractMatrix, Y::AbstractMatrix, A::LinearMap)
    check_dim_mul(X, Y, A)
    _unsafe_mul!(X, Y, A)
    return X
end

function _unsafe_mul!(X, Y::AbstractMatrix, A::LinearMap)
    _unsafe_mul!(X', A', Y')
    return X
end
function _unsafe_mul!(X, Y::TransposeAbsVecOrMat, A::LinearMap)
    _unsafe_mul!(transpose(X), transpose(A), transpose(Y))
    return X
end
# unwrap WrappedMaps
_unsafe_mul!(X, Y::AbstractMatrix, A::WrappedMap) = _unsafe_mul!(X, Y, A.lmap)
# disambiguation
_unsafe_mul!(X, Y::TransposeAbsVecOrMat, A::WrappedMap) = _unsafe_mul!(X, Y, A.lmap)

"""
    mul!(C::AbstractMatrix, A::AbstractMatrix, B::LinearMap, α, β) -> C

Combined inplace multiply-add ``A B α + C β``. The result is stored in `C` by overwriting
it. Note that `C` must not be aliased with either `A` or `B`.

## Examples
```jldoctest; setup=(using LinearAlgebra, LinearMaps)
julia> A=[1.0 1.0; 1.0 1.0]; B=LinearMap([1.0 2.0; 3.0 4.0]); C = copy(A);

julia> mul!(C, A, B, 1, 1)
2×2 Matrix{Float64}:
 5.0  7.0
 5.0  7.0
```
"""
function mul!(X::AbstractMatrix, Y::AbstractMatrix, A::LinearMap, α, β)
    check_dim_mul(X, Y, A)
    _unsafe_mul!(X, Y, A, α, β)
end

function _unsafe_mul!(X, Y::AbstractMatrix, A::LinearMap, α, β)
    if iszero(β)
        _unsafe_mul!(X', conj(α)*A', Y')
    else
        !isone(β) && rmul!(X, β)
        _unsafe_mul!(X', conj(α)*A', Y', true, true)
    end
    return X
end
# unwrap WrappedMaps
_unsafe_mul!(X, Y::AbstractMatrix, A::WrappedMap, α, β) = _unsafe_mul!(X, Y, A.lmap, α, β)
# disambiguation
_unsafe_mul!(X, Y::TransposeAbsVecOrMat, A::WrappedMap, α, β) =
    _unsafe_mul!(X, Y, A.lmap, α, β)
