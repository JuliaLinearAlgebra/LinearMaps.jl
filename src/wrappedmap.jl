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

# combine LinearMap and Matrix objects: linear combinations and map composition
Base.:(+)(A₁::LinearMap, A₂::AbstractMatrix) = +(A₁, WrappedMap(A₂))
Base.:(+)(A₁::AbstractMatrix, A₂::LinearMap) = +(WrappedMap(A₁), A₂)
Base.:(-)(A₁::LinearMap, A₂::AbstractMatrix) = -(A₁, WrappedMap(A₂))
Base.:(-)(A₁::AbstractMatrix, A₂::LinearMap) = -(WrappedMap(A₁), A₂)

Base.:(*)(A₁::LinearMap, A₂::AbstractMatrix) = *(A₁, WrappedMap(A₂))
Base.:(*)(A₁::AbstractMatrix, A₂::LinearMap) = *(WrappedMap(A₁), A₂)

# multiplication with vector/matrix
for Atype in (AbstractVector, AbstractMatrix)
    @eval Base.@propagate_inbounds LinearAlgebra.mul!(y::$Atype, A::WrappedMap, x::$Atype,
                            α::Number, β::Number=false) =
        _muladd!(MulStyle(A), y, A.lmap, x, α, β)
    @eval Base.@propagate_inbounds function LinearAlgebra.mul!(y::$Atype, A::TransposeMap{<:Any,<:WrappedMap}, x::$Atype,
                            α::Number, β::Number=false)
        if (issymmetric(A) || (isreal(A) && ishermitian(A)))
            _muladd!(MulStyle(A), y, A.lmap, x, α, β)
        else
            _muladd!(MulStyle(A), y, transpose(A.lmap.lmap), x, α, β)
        end
        return y
    end
    @eval Base.@propagate_inbounds function LinearAlgebra.mul!(y::$Atype, adjA::AdjointMap{<:Any,<:WrappedMap}, x::$Atype,
                            α::Number, β::Number=false)
        if ishermitian(adjA)
            _muladd!(MulStyle(adjA), y, adjA.lmap, x, α, β)
        else
            _muladd!(MulStyle(adjA), y, adjoint(adjA.lmap.lmap), x, α, β)
        end
        return y
    end
end

Base.@propagate_inbounds muladd!(y, A::WrappedMap, x, α, β, z) =
    _muladd!(MulStyle(A), y, A.lmap, x, α, β, z)
Base.@propagate_inbounds function muladd!(y, A::TransposeMap{<:Any,<:WrappedMap}, x, α, β, z)
    if (issymmetric(A) || (isreal(A) && ishermitian(A)))
        _muladd!(MulStyle(A), y, A.lmap, x, α, β, z)
    else
        _muladd!(MulStyle(A), y, transpose(A.lmap.lmap), x, α, β, z)
    end
    return y
end
Base.@propagate_inbounds function muladd!(y, adjA::AdjointMap{<:Any,<:WrappedMap}, x, α, β, z)
    if ishermitian(adjA)
        _muladd!(MulStyle(adjA), y, adjA.lmap, x, α, β, z)
    else
        _muladd!(MulStyle(adjA), y, adjoint(adjA.lmap.lmap), x, α, β, z)
    end
    return y
end
