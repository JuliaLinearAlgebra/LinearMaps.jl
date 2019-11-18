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

if VERSION ≥ v"1.3.0-alpha.115"
# multiplication with vector
Base.@propagate_inbounds LinearAlgebra.mul!(y::AbstractVector, A::WrappedMap, x::AbstractVector,
                    α::Number=true, β::Number=false) =
    mul!(y, A.lmap, x, α, β)

Base.@propagate_inbounds function LinearAlgebra.mul!(y::AbstractVector, transA::TransposeMap{<:Any,<:WrappedMap}, x::AbstractVector,
                                            α::Number=true, β::Number=false)
    if (issymmetric(transA) || (isreal(transA) && ishermitian(transA)))
        mul!(y, transA.lmap, x, α, β)
    else
        mul!(y, transpose(transA.lmap.lmap), x, α, β)
    end
    return y
end

Base.@propagate_inbounds function LinearAlgebra.mul!(y::AbstractVector, adjA::AdjointMap{<:Any,<:WrappedMap}, x::AbstractVector,
                                            α::Number=true, β::Number=false)
    if ishermitian(adjA)
        mul!(y, adjA.lmap, x, α, β)
    else
        mul!(y, adjoint(adjA.lmap.lmap), x, α, β)
    end
    return y
end

Base.@propagate_inbounds LinearAlgebra.mul!(Y::AbstractMatrix, A::MatrixMap, X::AbstractMatrix,
                    α::Number=true, β::Number=false) =
    mul!(Y, A.lmap, X, α, β)

else

LinearAlgebra.mul!(y::AbstractVector, A::WrappedMap, x::AbstractVector) =
    mul!(y, A.lmap, x)

Base.@propagate_inbounds function LinearAlgebra.mul!(y::AbstractVector, A::TransposeMap{<:Any,<:WrappedMap}, x::AbstractVector)
    if (issymmetric(A) || (isreal(A) && ishermitian(A)))
        mul!(y, A.lmap, x)
    else
        mul!(y, transpose(A.lmap.lmap), x)
    end
    return y
end

Base.@propagate_inbounds function LinearAlgebra.mul!(y::AbstractVector, A::AdjointMap{<:Any,<:WrappedMap}, x::AbstractVector)
    if ishermitian(A)
        mul!(y, A.lmap, x)
    else
        mul!(y, adjoint(A.lmap.lmap), x)
    end
    return y
end

Base.@propagate_inbounds LinearAlgebra.mul!(Y::AbstractMatrix, A::MatrixMap, X::AbstractMatrix) =
    mul!(Y, A.lmap, X)

end # VERSION

# properties
Base.size(A::WrappedMap) = size(A.lmap)
LinearAlgebra.issymmetric(A::WrappedMap) = A._issymmetric
LinearAlgebra.ishermitian(A::WrappedMap) = A._ishermitian
LinearAlgebra.isposdef(A::WrappedMap) = A._isposdef

# # multiplication with vector
# Base.@propagate_inbounds LinearAlgebra.mul!(y::AbstractVector, A::WrappedMap, x::AbstractVector) =
#     mul!(y, A.lmap, x)
#
# Base.@propagate_inbounds LinearAlgebra.mul!(y::AbstractVector, A::TransposeMap{<:Any,<:WrappedMap}, x::AbstractVector) =
#     (issymmetric(A) || (isreal(A) && ishermitian(A))) ? mul!(y, A.lmap, x) : mul!(y, transpose(A.lmap.lmap), x)
#
# Base.@propagate_inbounds function LinearAlgebra.mul!(y::AbstractVector, A::AdjointMap{<:Any,<:WrappedMap}, x::AbstractVector) =
#     ishermitian(A) ? mul!(y, A.lmap, x) : mul!(y, adjoint(A.lmap.lmap), x)

# Base.@propagate_inbounds LinearAlgebra.mul!(y::AbstractVector, A::WrappedMap, x::AbstractVector) =
#     mul!(y, A.lmap, x)
Base.:(*)(A::WrappedMap, x::AbstractVector) = *(A.lmap, x)

# combine LinearMap and Matrix objects: linear combinations and map composition
Base.:(+)(A₁::LinearMap, A₂::AbstractMatrix) = +(A₁, WrappedMap(A₂))
Base.:(+)(A₁::AbstractMatrix, A₂::LinearMap) = +(WrappedMap(A₁), A₂)
Base.:(-)(A₁::LinearMap, A₂::AbstractMatrix) = -(A₁, WrappedMap(A₂))
Base.:(-)(A₁::AbstractMatrix, A₂::LinearMap) = -(WrappedMap(A₁), A₂)

Base.:(*)(A₁::LinearMap, A₂::AbstractMatrix) = *(A₁, WrappedMap(A₂))
Base.:(*)(A₁::AbstractMatrix, A₂::LinearMap) = *(WrappedMap(A₁), A₂)
