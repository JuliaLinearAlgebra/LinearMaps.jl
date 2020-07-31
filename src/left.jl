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
LinearAlgebra.mul!(x::AdjointAbsVec, y::AdjointAbsVec, A::LinearMap) =
    mul!(x, y, A, true, false)

# todo: not sure if we need bounds checks and propagate inbounds stuff here

# mul!(x, y', A, α, β)
# the key here is that "adjoint" and "transpose" are lazy "views"
function LinearAlgebra.mul!(x::AdjointAbsVec, y::AdjointAbsVec,
        A::LinearMap, α::Number, β::Number)
    check_dim_mul(x, y, A)
    mul!(adjoint(x), A', y', α, β)
    return adjoint(x)
end

# mul!(x, transpose(y), A, α, β)
function LinearAlgebra.mul!(x::TransposeAbsVec, y::TransposeAbsVec,
         A::LinearMap, α::Number, β::Number)
    check_dim_mul(x, y, A)
    mul!(transpose(x), transpose(A), transpose(y), α, β)
    return transpose(x)
end
