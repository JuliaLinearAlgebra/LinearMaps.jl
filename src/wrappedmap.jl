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
    # multiplication with vector/matrix
    for Atype in (AbstractVector, AbstractMatrix)
        @eval Base.@propagate_inbounds LinearAlgebra.mul!(y::$Atype, A::WrappedMap, x::$Atype,
                            α::Number=true, β::Number=false) =
            mul!(y, A.lmap, x, α, β)

        @eval Base.@propagate_inbounds function LinearAlgebra.mul!(y::$Atype, transA::TransposeMap{<:Any,<:WrappedMap}, x::$Atype,
                                                    α::Number=true, β::Number=false)
            if (issymmetric(transA) || (isreal(transA) && ishermitian(transA)))
                mul!(y, transA.lmap, x, α, β)
            else
                mul!(y, transpose(transA.lmap.lmap), x, α, β)
            end
            return y
        end

        @eval Base.@propagate_inbounds function LinearAlgebra.mul!(y::$Atype, adjA::AdjointMap{<:Any,<:WrappedMap}, x::$Atype,
                                                    α::Number=true, β::Number=false)
            if ishermitian(adjA)
                mul!(y, adjA.lmap, x, α, β)
            else
                mul!(y, adjoint(adjA.lmap.lmap), x, α, β)
            end
            return y
        end
    end
else # generic 5-arg mul! for matrices is not available => can't provide 5-arg mul!'s here
    # multiplication with vector/matrix
    for Atype in (AbstractVector, AbstractMatrix)
        @eval Base.@propagate_inbounds LinearAlgebra.mul!(y::$Atype, A::WrappedMap, x::$Atype) =
            mul!(y, A.lmap, x)

        @eval Base.@propagate_inbounds function LinearAlgebra.mul!(y::$Atype, A::TransposeMap{<:Any,<:WrappedMap}, x::$Atype)
            if (issymmetric(A) || (isreal(A) && ishermitian(A)))
                mul!(y, A.lmap, x)
            else
                mul!(y, transpose(A.lmap.lmap), x)
            end
            return y
        end

        @eval Base.@propagate_inbounds function LinearAlgebra.mul!(y::$Atype, A::AdjointMap{<:Any,<:WrappedMap}, x::$Atype)
            if ishermitian(A)
                mul!(y, A.lmap, x)
            else
                mul!(y, adjoint(A.lmap.lmap), x)
            end
            return y
        end
    end
end # VERSION

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
