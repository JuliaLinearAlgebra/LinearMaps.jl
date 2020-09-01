struct FillMap{T} <: LinearMaps.LinearMap{T}
    value::T
    size::Dims{2}
end

# properties
Base.size(A::FillMap) = A.size
MulStyle(A::FillMap) = FiveArg()
LinearAlgebra.issymmetric(A::FillMap) = A.size[1] == A.size[2]
LinearAlgebra.ishermitian(A::FillMap) = isreal(A) && A.size[1] == A.size[2]
LinearAlgebra.isposdef(A::FillMap) = false
Base.:(==)(A::FillMap, B::FillMap) = A.value == B.value && A.size == B.size

LinearAlgebra.adjoint(A::FillMap) = FillMap(adjoint(value), revert(A.size))
LinearAlgebra.transpose(A::FillMap) = FillMap(transpose(value), revert(A.size))

function LinearMaps.A_mul_B!(y::AbstractVecOrMat, A::FillMap, x::AbstractVector)
    checkbounds(y, A, x)
    if iszero(A.value)
        fill!(y, zero(eltype(y)))
    else
        temp = sum(x)
        fill!(y, A.value*temp)
    end
    return y
end

Base.@propagate_inbounds function mul!(y::AbstractVecOrMat, A::FillMap, x::AbstractVector, α::Number, β::Number)
    @boundscheck checkbounds(y, A, x)
    if iszero(α)
        !isone(β) && rmul!(y, β)
        return y
    else
        temp = sum(x)*α
        if iszero(β)
            y .+= temp
        elseif isone(β)
            y .= y .+ temp
        else
            y .= y.*β .+ temp
        end
    return y
end

Base.:(+)(A::FillMap, B::FillMap) = A.size == B.size ? FillMap(A.value + B.value, A.size) : throw(DimensionMismatch())
Base.:(-)(A::FillMap, B::FillMap) = A.size == B.size ? FillMap(A.value - B.value, A.size) : throw(DimensionMismatch())
Base.:(*)(λ::Number, A::FillMap) = FillMap(λ*A.value, size(A))
Base.:(*)(A::FillMap, λ::Number) = FillMap(A.value*λ, size(A))
Base.:(*)(J::UniformScaling, A::FillMap) = FillMap(J.λ*A.value, size(A))
Base.:(*)(A::FillMap, J::UniformScaling) = FillMap(A.value*J.λ, size(A))
Base.:(*)(J::LinearMaps.UniformScalingMap, A::FillMap) = FillMap(J.λ*A.value, size(A))
Base.:(*)(A::FillMap, J::LinearMaps.UniformScalingMap) = FillMap(A.value*J.λ, size(A))

function Base.:(*)(A::FillMap, B::FillMap)
    mA, nA = size(A)
    mB, nB = size(B)
    nA != mB && throw(DimensionMismatch())
    return FillMap(A.value*B.value*nA, (mA, nB))
end
