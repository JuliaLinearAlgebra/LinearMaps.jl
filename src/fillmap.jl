"""
    FillMap(λ, (m, n))::FillMap
    FillMap(λ, m, n)::FillMap

Construct a (lazy) representation of an operator whose matrix representation
would be an m×n-matrix filled constantly with the value `λ`.
"""
struct FillMap{T} <: LinearMap{T}
    λ::T
    size::Dims{2}
    function FillMap(λ::T, dims::Dims{2}) where {T}
        (dims[1]>=0 && dims[2]>=0) || throw(ArgumentError("dims of FillMap must be non-negative"))
        return new{T}(λ, dims)
    end
end

FillMap(λ, m::Int, n::Int) = FillMap(λ, (m, n))

# properties
Base.size(A::FillMap) = A.size
MulStyle(A::FillMap) = FiveArg()
LinearAlgebra.issymmetric(A::FillMap) = A.size[1] == A.size[2]
LinearAlgebra.ishermitian(A::FillMap) = isreal(A.λ) && A.size[1] == A.size[2]
LinearAlgebra.isposdef(A::FillMap) = (size(A, 1) == size(A, 2) == 1 && isposdef(A.λ))
Base.:(==)(A::FillMap, B::FillMap) = A.λ == B.λ && A.size == B.size

LinearAlgebra.adjoint(A::FillMap) = FillMap(adjoint(A.λ), reverse(A.size))
LinearAlgebra.transpose(A::FillMap) = FillMap(transpose(A.λ), reverse(A.size))

function Base.:(*)(A::FillMap, x::AbstractVector)
    T = typeof(oneunit(eltype(A)) * (zero(eltype(x)) + zero(eltype(x))))
    return fill(iszero(A.λ) ? zero(T) : A.λ*sum(x), A.size[1])
end

function _unsafe_mul!(y::AbstractVecOrMat, A::FillMap, x::AbstractVector)
    return fill!(y, iszero(A.λ) ? zero(eltype(y)) : A.λ*sum(x))
end

_unsafe_mul!(Y::AbstractMatrix, A::FillMap, x::Number) = fill!(Y, A.λ*x)
function _unsafe_mul!(Y::AbstractMatrix, A::FillMap, x::Number, α::Number, β::Number)
    LinearAlgebra._rmul_or_fill!(Y, β)
    Y .+= A.λ*x*α
    return Y
end

function _unsafe_mul!(y::AbstractVecOrMat, A::FillMap, x::AbstractVector, α::Number, β::Number)
    if iszero(α)
        !isone(β) && rmul!(y, β)
    else
        temp = A.λ * sum(x) * α
        if iszero(β)
            y .= temp
        elseif isone(β)
            y .+= temp
        else
            y .= y .* β .+ temp
        end
    end
    return y
end

Base.:(+)(A::FillMap, B::FillMap) = A.size == B.size ? FillMap(A.λ + B.λ, A.size) : throw(DimensionMismatch())
Base.:(-)(A::FillMap) = FillMap(-A.λ, A.size)
Base.:(*)(λ::Number, A::FillMap) = FillMap(λ * A.λ, size(A))
Base.:(*)(A::FillMap, λ::Number) = FillMap(A.λ * λ, size(A))
Base.:(*)(λ::RealOrComplex, A::FillMap) = FillMap(λ * A.λ, size(A))
Base.:(*)(A::FillMap, λ::RealOrComplex) = FillMap(A.λ * λ, size(A))

function Base.:(*)(A::FillMap, B::FillMap)
    check_dim_mul(A, B)
    return FillMap(A.λ*B.λ*size(A, 2), (size(A, 1), size(B, 2)))
end
