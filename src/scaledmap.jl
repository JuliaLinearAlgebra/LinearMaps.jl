struct ScaledMap{T, S<:RealOrComplex, A<:LinearMap} <: LinearMap{T}
    λ::S
    lmap::A
    function ScaledMap{T,S,A}(λ::S, lmap::A) where {T, S <: RealOrComplex, A <: LinearMap}
        Base.promote_op(*, S, eltype(lmap)) == T || throw(InexactError())
        new{T,S,A}(λ, lmap)
    end
end

# constructor
ScaledMap{T}(λ::S, lmap::A) where {T,S<:RealOrComplex,A<:LinearMap} =
    ScaledMap{Base.promote_op(*, S, eltype(lmap)),S,A}(λ, lmap)

# show
function Base.show(io::IO, A::ScaledMap{T}) where {T}
    println(io, "LinearMaps.ScaledMap{$T}, scale = $(A.λ)")
    show(io, A.lmap)
end

# basic methods
Base.size(A::ScaledMap) = size(A.lmap)
Base.isreal(A::ScaledMap) = isreal(A.λ) && isreal(A.lmap)
LinearAlgebra.issymmetric(A::ScaledMap) = issymmetric(A.lmap)
LinearAlgebra.ishermitian(A::ScaledMap) = ishermitian(A.lmap)

LinearAlgebra.isposdef(A::ScaledMap) = isposdef(A.λ) && isposdef(A.lmap)

Base.transpose(A::ScaledMap) = A.λ * transpose(A.lmap)
Base.adjoint(A::ScaledMap) = conj(A.λ) * adjoint(A.lmap)

# comparison (sufficient, not necessary)
Base.:(==)(A::ScaledMap, B::ScaledMap) =
    (eltype(A) == eltype(B) && A.lmap == B.lmap) && A.λ == B.λ

# approximate comparison (because == for real scalars is dubious)
# Base.:(≈)(A::ScaledMap, B::ScaledMap) =
#     (eltype(A) == eltype(B) && A.lmap == B.lmap) && A.λ ≈ B.λ

# x * conj(x) is real in math, but not perfectly so in computation,
# but we want it to be real here (when possible) for isposdef()
# and so that α*conj(α)*A is has a real scale when A is Real
# @inline _scalar_product(a, b) = isreal(a*b) ? real(a*b) : a*b

# scalar multiplication and division
function Base.:(*)(α::RealOrComplex, A::LinearMap)
    T = Base.promote_op(*, typeof(α), eltype(A))
    return ScaledMap{T}(α, A)
end
function Base.:(*)(A::LinearMap, α::RealOrComplex)
    T = Base.promote_op(*, typeof(α), eltype(A))
    return ScaledMap{T}(α, A)
end

Base.:(*)(α::Number, A::ScaledMap) = (α * A.λ) * A.lmap
Base.:(*)(A::ScaledMap, α::Number) = (A.λ * α) * A.lmap
Base.:(*)(A::UniformScaling, B::LinearMap) = A.λ * B
Base.:(*)(A::LinearMap, B::UniformScaling) = A * B.λ
# needed for disambiguation
Base.:(*)(α::RealOrComplex, A::ScaledMap) = (α * A.λ) * A.lmap
Base.:(*)(A::ScaledMap, α::RealOrComplex) = (A.λ * α) * A.lmap
Base.:(-)(A::LinearMap) = -1 * A

# composition (not essential, but might save multiple scaling operations)
Base.:(*)(A::ScaledMap, B::ScaledMap) = (A.λ * B.λ) * (A.lmap * B.lmap)
Base.:(*)(A::ScaledMap, B::LinearMap) = A.λ * (A.lmap * B)
Base.:(*)(A::LinearMap, B::ScaledMap) = (A * B.lmap) * B.λ

# multiplication with vectors
function A_mul_B!(y::AbstractVector, A::ScaledMap, x::AbstractVector)
    # no size checking, will be done by map
    mul!(y, A.lmap, x, A.λ, false)
end

function LinearAlgebra.mul!(y::AbstractVector, A::ScaledMap, x::AbstractVector, α::Number, β::Number)
    # no size checking, will be done by map
    mul!(y, A.lmap, x, A.λ * α, β)
end

At_mul_B!(y::AbstractVector, A::ScaledMap, x::AbstractVector) = A_mul_B!(y, transpose(A), x)
Ac_mul_B!(y::AbstractVector, A::ScaledMap, x::AbstractVector) = A_mul_B!(y, adjoint(A), x)
