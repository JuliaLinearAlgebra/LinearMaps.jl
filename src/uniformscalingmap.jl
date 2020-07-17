struct UniformScalingMap{T} <: LinearMap{T}
    λ::T
    M::Int
    function UniformScalingMap(λ::Number, M::Int)
        M < 0 && throw(ArgumentError("size of UniformScalingMap must be non-negative, got $M"))
        return new{typeof(λ)}(λ, M)
    end
end
UniformScalingMap(λ::Number, M::Int, N::Int) =
    (M == N ? UniformScalingMap(λ, M) : error("UniformScalingMap needs to be square"))
UniformScalingMap(λ::T, sz::Dims{2}) where {T} =
    (sz[1] == sz[2] ? UniformScalingMap(λ, sz[1]) : error("UniformScalingMap needs to be square"))

MulStyle(::UniformScalingMap) = FiveArg()

# properties
Base.size(A::UniformScalingMap) = (A.M, A.M)
Base.isreal(A::UniformScalingMap) = isreal(A.λ)
LinearAlgebra.issymmetric(::UniformScalingMap) = true
LinearAlgebra.ishermitian(A::UniformScalingMap) = isreal(A)
LinearAlgebra.isposdef(A::UniformScalingMap) = isposdef(A.λ)

# comparison of UniformScalingMap objects
Base.:(==)(A::UniformScalingMap, B::UniformScalingMap) = (A.λ == B.λ && A.M == B.M)

# special transposition behavior
LinearAlgebra.transpose(A::UniformScalingMap) = A
LinearAlgebra.adjoint(A::UniformScalingMap)   = UniformScalingMap(conj(A.λ), size(A))

# multiplication with scalar
Base.:(*)(A::UniformScaling, B::LinearMap) = A.λ * B
Base.:(*)(A::LinearMap, B::UniformScaling) = A * B.λ
Base.:(*)(α::Number, J::UniformScalingMap) = UniformScalingMap(α * J.λ, size(J))
Base.:(*)(J::UniformScalingMap, α::Number) = UniformScalingMap(J.λ * α, size(J))
# needed for disambiguation
Base.:(*)(α::RealOrComplex, J::UniformScalingMap) = UniformScalingMap(α * J.λ, size(J))
Base.:(*)(J::UniformScalingMap, α::RealOrComplex) = UniformScalingMap(J.λ * α, size(J))

# multiplication with vector
Base.:(*)(A::UniformScalingMap, x::AbstractVector) =
    length(x) == A.M ? A.λ * x : throw(DimensionMismatch("*"))

# multiplication with vector/matrix
for Atype in (AbstractVector, AbstractMatrix)
    @eval Base.@propagate_inbounds function LinearAlgebra.mul!(y::$Atype, J::UniformScalingMap, x::$Atype,
                α::Number, β::Number)
        @boundscheck check_dim_mul(y, J, x)
        _scaling!(y, J.λ, x, α, β)
        return y
    end
end

function _scaling!(y, λ::Number, x, α::Number, β::Number)
    if (iszero(α) || iszero(λ))
        iszero(β) && (fill!(y, zero(eltype(y))); return y)
        isone(β) && return y
        # β != 0, 1
        rmul!(y, β)
        return y
    elseif isone(α)
        if iszero(β)
            isone(λ) && return copyto!(y, x)
            y .= λ .* x
            return y
        elseif isone(β)
            isone(λ) && return y .+= x
            y .+= λ .* x
            return y
        else # β != 0, 1
            isone(λ) && (axpby!(one(eltype(x)), x, β, y); return y)
            y .= y .* β .+ λ .* x
            return y
        end
    else # α != 0, 1
        if iszero(β)
            isone(λ) && (y .= x .* α; return y)
            y .= λ .* x .* α
            return y
        elseif isone(β)
            isone(λ) && (axpby!(α, x, β, y); return y)
            y .+= λ .* x .* α
            return y
        else # β != 0, 1
            isone(λ) && (y .= y .* β .+ x .* α; return y)
            y .= y .* β .+ λ .* x .* α
            return y
        end # β-cases
    end # α-cases
end # function _scaling!

Base.@propagate_inbounds function LinearAlgebra.mul!(y::AbstractVector, A::TransposeMap{<:Any,<:UniformScalingMap}, x::AbstractVector)
    mul!(y, transpose(A.lmap), x)
end
Base.@propagate_inbounds function LinearAlgebra.mul!(y::AbstractVector, A::TransposeMap{<:Any,<:UniformScalingMap}, x::AbstractVector, α::Number, β::Number)
    mul!(y, transpose(A.lmap), x, α, β)
end

Base.@propagate_inbounds function LinearAlgebra.mul!(y::AbstractVector, A::AdjointMap{<:Any,<:UniformScalingMap}, x::AbstractVector)
    mul!(y, adjoint(A.lmap), x)
end
Base.@propagate_inbounds function LinearAlgebra.mul!(y::AbstractVector, A::AdjointMap{<:Any,<:UniformScalingMap}, x::AbstractVector, α::Number, β::Number)
    mul!(y, adjoint(A.lmap), x, α, β)
end

# combine LinearMap and UniformScaling objects in linear combinations
Base.:(+)(A₁::LinearMap, A₂::UniformScaling) = A₁ + UniformScalingMap(A₂.λ, size(A₁, 1))
Base.:(+)(A₁::UniformScaling, A₂::LinearMap) = UniformScalingMap(A₁.λ, size(A₂, 1)) + A₂
Base.:(-)(A₁::LinearMap, A₂::UniformScaling) = A₁ - UniformScalingMap(A₂.λ, size(A₁, 1))
Base.:(-)(A₁::UniformScaling, A₂::LinearMap) = UniformScalingMap(A₁.λ, size(A₂, 1)) - A₂
