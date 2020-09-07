struct FillMap{T} <: LinearMaps.LinearMap{T}
    λ::T
    size::Dims{2}
end

# properties
Base.size(A::FillMap) = A.size
MulStyle(A::FillMap) = FiveArg()
LinearAlgebra.issymmetric(A::FillMap) = A.size[1] == A.size[2]
LinearAlgebra.ishermitian(A::FillMap) = isreal(A) && A.size[1] == A.size[2]
LinearAlgebra.isposdef(A::FillMap) = false
Base.:(==)(A::FillMap, B::FillMap) = A.λ == B.λ && A.size == B.size

LinearAlgebra.adjoint(A::FillMap) = FillMap(adjoint(A.λ), revert(A.size))
LinearAlgebra.transpose(A::FillMap) = FillMap(transpose(A.λ), revert(A.size))

function _unsafe_mul!(y::AbstractVecOrMat, A::FillMap, x::AbstractVector)
    return fill!(y, iszero(A.λ) ? zero(eltype(y)) : A.λ*sum(x))
end

function _unsafe_mul!(y::AbstractVecOrMat, A::FillMap, x::AbstractVector, α::Number, β::Number)
    if iszero(α)
        !isone(β) && rmul!(y, β)
        return y
    else
        temp = A.λ * sum(x) * α
        if iszero(β)
            y .= temp
        elseif isone(β)
            y .+= temp
        else
            y .= y .* β .+ temp
        end
    return y
end

Base.:(+)(A::FillMap, B::FillMap) = A.size == B.size ? FillMap(A.λ + B.λ, A.size) : throw(DimensionMismatch())
Base.:(-)(A::FillMap) = FillMap(-A.λ, A.size)
Base.:(*)(λ::Number, A::FillMap) = FillMap(λ * A.λ, size(A))
Base.:(*)(A::FillMap, λ::Number) = FillMap(A.λ * λ, size(A))

function Base.:(*)(A::FillMap, B::FillMap)
    mA, nA = size(A)
    mB, nB = size(B)
    nA != mB && throw(DimensionMismatch())
    return FillMap(A.λ*B.λ*nA, (mA, nB))
end
