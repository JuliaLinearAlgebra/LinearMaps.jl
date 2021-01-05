"""
    struct ScaledMap{T, S<:RealOrComplex, A<:LinearMap} <: LinearMap{T}

Lazy representation of real or complex scaled maps ``\\alpha A``.
"""
struct ScaledMap{T, S<:RealOrComplex, A<:LinearMap} <: LinearMap{T}
    λ::S
    lmap::A
    function ScaledMap{T}(λ::S, lmap::A) where {T, S <: RealOrComplex, A <: LinearMap{<:RealOrComplex}}
        @assert Base.promote_op(*, S, eltype(lmap)) == T "target type $T cannot hold products of $S and $(eltype(lmap)) objects"
        new{T,S,A}(λ, lmap)
    end
end

# constructor
ScaledMap(λ::RealOrComplex, lmap::LinearMap{<:RealOrComplex}) =
    ScaledMap{Base.promote_op(*, typeof(λ), eltype(lmap))}(λ, lmap)

# basic methods
Base.size(A::ScaledMap) = size(A.lmap)
Base.isreal(A::ScaledMap) = isreal(A.λ) && isreal(A.lmap)
LinearAlgebra.issymmetric(A::ScaledMap) = issymmetric(A.lmap)
LinearAlgebra.ishermitian(A::ScaledMap) = isreal(A.λ) && ishermitian(A.lmap)
LinearAlgebra.isposdef(A::ScaledMap) = isposdef(A.λ) && isposdef(A.lmap)

Base.transpose(A::ScaledMap) = A.λ * transpose(A.lmap)
Base.adjoint(A::ScaledMap) = conj(A.λ) * adjoint(A.lmap)

# comparison (sufficient, not necessary)
Base.:(==)(A::ScaledMap, B::ScaledMap) =
    eltype(A) == eltype(B) && A.lmap == B.lmap && A.λ == B.λ

# scalar multiplication and division
Base.:(*)(α::RealOrComplex, A::LinearMap{<:RealOrComplex}) = ScaledMap(α, A)
Base.:(*)(A::LinearMap{<:RealOrComplex}, α::RealOrComplex) = ScaledMap(α, A)

Base.:(*)(α::Number, A::ScaledMap) = (α * A.λ) * A.lmap
Base.:(*)(A::ScaledMap, α::Number) = A.lmap * (A.λ * α)
# needed for disambiguation
Base.:(*)(α::RealOrComplex, A::ScaledMap) = (α * A.λ) * A.lmap
Base.:(*)(A::ScaledMap, α::RealOrComplex) = (A.λ * α) * A.lmap
Base.:(-)(A::LinearMap) = -1 * A

# composition (not essential, but might save multiple scaling operations)
Base.:(*)(A::ScaledMap, B::ScaledMap) = (A.λ * B.λ) * (A.lmap * B.lmap)
Base.:(*)(A::ScaledMap, B::LinearMap) = A.λ * (A.lmap * B)
Base.:(*)(A::LinearMap, B::ScaledMap) = (A * B.lmap) * B.λ

# multiplication with vectors/matrices
for (In, Out) in (
        (AbstractVector{<:RealOrComplex}, AbstractVecOrMat{<:RealOrComplex}),
        (AbstractMatrix{<:RealOrComplex}, AbstractMatrix{<:RealOrComplex}),
    )
    @eval begin
        function _unsafe_mul!(y::$Out, A::ScaledMap, x::$In)
            return _unsafe_mul!(y, A.lmap, x, A.λ, false)
        end
        function _unsafe_mul!(y::$Out, A::ScaledMap, x::$In, α::Number, β::Number)
            return _unsafe_mul!(y, A.lmap, x, A.λ * α, β)
        end
    end
end
