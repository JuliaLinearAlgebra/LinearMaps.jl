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

# properties
Base.size(A::UniformScalingMap, n) = (n==1 || n==2 ? A.M : error("LinearMap objects have only 2 dimensions"))
Base.size(A::UniformScalingMap) = (A.M, A.M)
Base.isreal(A::UniformScalingMap) = isreal(A.λ)
LinearAlgebra.issymmetric(::UniformScalingMap) = true
LinearAlgebra.ishermitian(A::UniformScalingMap) = isreal(A)
LinearAlgebra.isposdef(A::UniformScalingMap) = isposdef(A.λ)

# comparison of UniformScalingMap objects, sufficient but not necessary
Base.:(==)(A::UniformScalingMap, B::UniformScalingMap) = (eltype(A) == eltype(B) && A.λ == B.λ && A.M == B.M)

# special transposition behavior
LinearAlgebra.transpose(A::UniformScalingMap) = A
LinearAlgebra.adjoint(A::UniformScalingMap)   = UniformScalingMap(conj(A.λ), size(A))

# multiplication with scalar
Base.:(*)(α::Number, J::UniformScalingMap) = UniformScalingMap(α * J.λ, size(J))
Base.:(*)(J::UniformScalingMap, α::Number) = UniformScalingMap(J.λ * α, size(J))

# multiplication with vector
Base.:(*)(A::UniformScalingMap, x::AbstractVector) =
    length(x) == A.M ? A.λ * x : throw(DimensionMismatch("A_mul_B!"))

if VERSION < v"1.3.0-alpha.115"
function A_mul_B!(y::AbstractVector, A::UniformScalingMap, x::AbstractVector)
    (length(x) == length(y) == A.M || throw(DimensionMismatch("A_mul_B!")))
    if iszero(A.λ)
        return fill!(y, zero(eltype(y)))
    elseif isone(A.λ)
        return copyto!(y, x)
    else
        # call of LinearAlgebra.generic_mul! since order of arguments in mul! in
        # stdlib/LinearAlgebra/src/generic.jl reversed
        return LinearAlgebra.generic_mul!(y, A.λ, x)
    end
end
else # 5-arg mul! exists and order of arguments is corrected
function A_mul_B!(y::AbstractVector, A::UniformScalingMap, x::AbstractVector)
    (length(x) == length(y) == A.M || throw(DimensionMismatch("A_mul_B!")))
    λ = A.λ
    if iszero(λ)
        return fill!(y, zero(eltype(y)))
    elseif isone(λ)
        return copyto!(y, x)
    else
        return y .= λ .* x
    end
end
end # VERSION

@inline function LinearAlgebra.mul!(y::AbstractVector, J::UniformScalingMap, x::AbstractVector, α::Number=true, β::Number=false)
    @boundscheck (length(x) == length(y) == J.M || throw(DimensionMismatch("mul!")))
    _scaling!(y, J, x, α, β)
    return y
end

@inline function LinearAlgebra.mul!(Y::AbstractMatrix, J::UniformScalingMap, X::AbstractMatrix, α::Number=true, β::Number=false)
    @boundscheck size(X) == size(Y) || throw(DimensionMismatch("mul!"))
    @boundscheck size(X,1) == J.M || throw(DimensionMismatch("mul!"))
    _scaling!(Y, J, X, α, β)
    return Y
end

function _scaling!(y, J::UniformScalingMap, x, α::Number=true, β::Number=false)
    λ = J.λ
    @inbounds if isone(α)
        if iszero(β)
            iszero(λ) && return fill!(y, zero(eltype(y)))
            isone(λ) && return copyto!(y, x)
            y .= λ .* x
            return y
        elseif isone(β)
            iszero(λ) && return y
            isone(λ) && return y .+= x
            y .+= λ .* x
            return y
        else # β != 0, 1
            iszero(λ) && (rmul!(y, β); return y)
            isone(λ) && (y .= y .* β .+ x; return y)
            y .= y .* β .+ λ .* x
            return y
        end
    elseif iszero(α)
        iszero(β) && (fill!(y, zero(eltype(y))); return y)
        isone(β) && return y
        # β != 0, 1
        rmul!(y, β)
        return y
    else # α != 0, 1
        iszero(β) && (y .= λ .* x .* α; return y)
        isone(β) && (y .+= λ .* x .* α; return y)
        # β != 0, 1
        y .= y .* β .+ λ .* x .* α
        return y
    end
end

At_mul_B!(y::AbstractVector, A::UniformScalingMap, x::AbstractVector) = A_mul_B!(y, transpose(A), x)
Ac_mul_B!(y::AbstractVector, A::UniformScalingMap, x::AbstractVector) = A_mul_B!(y, adjoint(A), x)


# combine LinearMap and UniformScaling objects in linear combinations
Base.:(+)(A₁::LinearMap, A₂::UniformScaling) = A₁ + UniformScalingMap(A₂.λ, size(A₁, 1))
Base.:(+)(A₁::UniformScaling, A₂::LinearMap) = UniformScalingMap(A₁.λ, size(A₂, 1)) + A₂
Base.:(-)(A₁::LinearMap, A₂::UniformScaling) = A₁ - UniformScalingMap(A₂.λ, size(A₁, 1))
Base.:(-)(A₁::UniformScaling, A₂::LinearMap) = UniformScalingMap(A₁.λ, size(A₂, 1)) - A₂
