struct WrappedMap{T, A<:MapOrMatrix} <: LinearMap{T}
    lmap::A
    _issymmetric::Bool
    _ishermitian::Bool
    _isposdef::Bool
end
function WrappedMap(lmap::MapOrMatrix{T};
    issymmetric::Bool = issymmetric(lmap),
    ishermitian::Bool = ishermitian(lmap),
    isposdef::Bool = isposdef(lmap)) where {T}
    WrappedMap{T, typeof(lmap)}(lmap, issymmetric, ishermitian, isposdef)
end
function WrappedMap{T}(lmap::MapOrMatrix;
    issymmetric::Bool = issymmetric(lmap),
    ishermitian::Bool = ishermitian(lmap),
    isposdef::Bool = isposdef(lmap)) where {T}
    WrappedMap{T, typeof(lmap)}(lmap, issymmetric, ishermitian, isposdef)
end

const MatrixMap{T} = WrappedMap{T,<:AbstractMatrix}

MulStyle(A::WrappedMap) = MulStyle(A.lmap)

LinearAlgebra.transpose(A::MatrixMap{T}) where {T} =
    WrappedMap{T}(transpose(A.lmap); issymmetric=A._issymmetric, ishermitian=A._ishermitian, isposdef=A._isposdef)
LinearAlgebra.adjoint(A::MatrixMap{T}) where {T} =
    WrappedMap{T}(adjoint(A.lmap); issymmetric=A._issymmetric, ishermitian=A._ishermitian, isposdef=A._isposdef)

Base.:(==)(A::MatrixMap, B::MatrixMap) =
    (eltype(A)==eltype(B) && A.lmap==B.lmap && A._issymmetric==B._issymmetric &&
     A._ishermitian==B._ishermitian && A._isposdef==B._isposdef)

# properties
Base.size(A::WrappedMap) = size(A.lmap)
LinearAlgebra.issymmetric(A::WrappedMap) = A._issymmetric
LinearAlgebra.ishermitian(A::WrappedMap) = A._ishermitian
LinearAlgebra.isposdef(A::WrappedMap) = A._isposdef

# multiplication with vectors & matrices
A_mul_B!(y::AbstractVector, A::WrappedMap, x::AbstractVector) = A_mul_B!(y, A.lmap, x)
Base.:(*)(A::WrappedMap, x::AbstractVector) = *(A.lmap, x)

At_mul_B!(y::AbstractVector, A::WrappedMap, x::AbstractVector) =
    (issymmetric(A) || (isreal(A) && ishermitian(A))) ? A_mul_B!(y, A.lmap, x) : At_mul_B!(y, A.lmap, x)

Ac_mul_B!(y::AbstractVector, A::WrappedMap, x::AbstractVector) =
    ishermitian(A) ? A_mul_B!(y, A.lmap, x) : Ac_mul_B!(y, A.lmap, x)

if VERSION ≥ v"1.3.0-alpha.115"
    for Atype in (AbstractVector, AbstractMatrix)
        @eval Base.@propagate_inbounds LinearAlgebra.mul!(y::$Atype, A::WrappedMap, x::$Atype,
                        α::Number=true, β::Number=false) =
            mul!(y, A.lmap, x, α, β)
    end
else
# This is somewhat suboptimal, because the absence of 5-arg mul! for MatrixMaps
# doesn't allow to define a 5-arg mul! for WrappedMaps which do have a 5-arg mul!
# I'd assume, however, that 5-arg mul! becomes standard in Julia v≥1.3 anyway
# the idea is to let the fallback handle 5-arg calls
    for Atype in (AbstractVector, AbstractMatrix)
        @eval Base.@propagate_inbounds LinearAlgebra.mul!(Y::$Atype, A::WrappedMap, X::$Atype) =
            mul!(Y, A.lmap, X)
    end
end # VERSION

# combine LinearMap and Matrix objects: linear combinations and map composition
Base.:(+)(A₁::LinearMap, A₂::AbstractMatrix) = +(A₁, WrappedMap(A₂))
Base.:(+)(A₁::AbstractMatrix, A₂::LinearMap) = +(WrappedMap(A₁), A₂)
Base.:(-)(A₁::LinearMap, A₂::AbstractMatrix) = -(A₁, WrappedMap(A₂))
Base.:(-)(A₁::AbstractMatrix, A₂::LinearMap) = -(WrappedMap(A₁), A₂)

Base.:(*)(A₁::LinearMap, A₂::AbstractMatrix) = *(A₁, WrappedMap(A₂))
Base.:(*)(A₁::AbstractMatrix, A₂::LinearMap) = *(WrappedMap(A₁), A₂)
