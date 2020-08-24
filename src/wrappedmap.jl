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
Base.:(*)(A::WrappedMap, x::AbstractVector) = *(A.lmap, x)

for (intype, outtype) in Any[Any[AbstractVector, AbstractVecOrMat], Any[AbstractMatrix, AbstractMatrix]]
    @eval begin
        Base.@propagate_inbounds function mul!(y::$outtype, A::WrappedMap, x::$intype)
            @boundscheck check_dim_mul(y, A, x)
            return @inbounds mul!(y, A.lmap, x)
        end
        Base.@propagate_inbounds function mul!(y::$outtype, At::TransposeMap{<:Any,<:WrappedMap}, x::$intype)
            @boundscheck check_dim_mul(y, At, x)
            A = At.lmap
            return @inbounds (issymmetric(A) || (isreal(A) && ishermitian(A))) ? mul!(y, A.lmap, x) : mul!(y, transpose(A.lmap), x)
        end
        Base.@propagate_inbounds function mul!(y::$outtype, Ac::AdjointMap{<:Any,<:WrappedMap}, x::$intype)
            @boundscheck check_dim_mul(y, Ac, x)
            A = Ac.lmap
            return @inbounds ishermitian(A) ? mul!(y, A.lmap, x) : mul!(y, adjoint(A.lmap), x)
        end
    end
end

if VERSION ≥ v"1.3.0-alpha.115"
    for (intype, outtype) in Any[Any[AbstractVector, AbstractVecOrMat], Any[AbstractMatrix, AbstractMatrix]]
        @eval begin
            Base.@propagate_inbounds function mul!(y::$outtype, A::WrappedMap, x::$intype, α::Number, β::Number)
                @boundscheck check_dim_mul(y, A, x)
                return @inbounds mul!(y, A.lmap, x, α, β)
            end
            Base.@propagate_inbounds function mul!(y::$outtype, At::TransposeMap{<:Any,<:WrappedMap}, x::$intype, α::Number, β::Number)
                @boundscheck check_dim_mul(y, At, x)
                A = At.lmap
                return @inbounds (issymmetric(A) || (isreal(A) && ishermitian(A))) ? mul!(y, A.lmap, x, α, β) : mul!(y, transpose(A.lmap), x, α, β)
            end
            Base.@propagate_inbounds function mul!(y::$outtype, Ac::AdjointMap{<:Any,<:WrappedMap}, x::$intype, α::Number, β::Number)
                @boundscheck check_dim_mul(y, Ac, x)
                A = Ac.lmap
                return @inbounds ishermitian(A) ? mul!(y, A.lmap, x, α, β) : mul!(y, adjoint(A.lmap), x, α, β)
            end
        end
    end
end # VERSION

# combine LinearMap and Matrix objects: linear combinations and map composition
Base.:(+)(A₁::LinearMap, A₂::AbstractMatrix) = +(A₁, WrappedMap(A₂))
Base.:(+)(A₁::AbstractMatrix, A₂::LinearMap) = +(WrappedMap(A₁), A₂)
Base.:(-)(A₁::LinearMap, A₂::AbstractMatrix) = -(A₁, WrappedMap(A₂))
Base.:(-)(A₁::AbstractMatrix, A₂::LinearMap) = -(WrappedMap(A₁), A₂)

Base.:(*)(A₁::LinearMap, A₂::AbstractMatrix) = *(A₁, WrappedMap(A₂))
Base.:(*)(A₁::AbstractMatrix, A₂::LinearMap) = *(WrappedMap(A₁), A₂)
