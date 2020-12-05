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
    WrappedMap{T}(transpose(A.lmap);
                    issymmetric = A._issymmetric,
                    ishermitian = A._ishermitian,
                    isposdef = A._isposdef)
LinearAlgebra.adjoint(A::MatrixMap{T}) where {T} =
    WrappedMap{T}(adjoint(A.lmap);
                    issymmetric = A._issymmetric,
                    ishermitian = A._ishermitian,
                    isposdef = A._isposdef)

Base.:(==)(A::MatrixMap, B::MatrixMap) =
    (eltype(A) == eltype(B) && A.lmap == B.lmap && A._issymmetric == B._issymmetric &&
     A._ishermitian == B._ishermitian && A._isposdef == B._isposdef)

# properties
Base.size(A::WrappedMap) = size(A.lmap)
Base.parent(A::WrappedMap) = A.lmap
LinearAlgebra.issymmetric(A::WrappedMap) = A._issymmetric
LinearAlgebra.ishermitian(A::WrappedMap) = A._ishermitian
LinearAlgebra.isposdef(A::WrappedMap) = A._isposdef

# multiplication with vectors & matrices
for (In, Out) in ((AbstractVector, AbstractVecOrMat), (AbstractMatrix, AbstractMatrix))
    @eval begin
        _unsafe_mul!(y::$Out, A::WrappedMap, x::$In) = _unsafe_mul!(y, A.lmap, x)
        function _unsafe_mul!(y::$Out, At::TransposeMap{<:Any,<:WrappedMap}, x::$In)
            A = At.lmap
            return (issymmetric(A) || (isreal(A) && ishermitian(A))) ?
                _unsafe_mul!(y, A.lmap, x) :
                _unsafe_mul!(y, transpose(A.lmap), x)
        end
        function _unsafe_mul!(y::$Out, Ac::AdjointMap{<:Any,<:WrappedMap}, x::$In)
            A = Ac.lmap
            return ishermitian(A) ?
                _unsafe_mul!(y, A.lmap, x) :
                _unsafe_mul!(y, adjoint(A.lmap), x)
        end
    end
end

if VERSION ≥ v"1.3.0-alpha.115"
    for (In, Out) in ((AbstractVector, AbstractVecOrMat), (AbstractMatrix, AbstractMatrix))
        @eval begin
            function _unsafe_mul!(y::$Out, A::WrappedMap, x::$In, α::Number, β::Number)
                return _unsafe_mul!(y, A.lmap, x, α, β)
            end
            function _unsafe_mul!(y::$Out, At::TransposeMap{<:Any,<:WrappedMap}, x::$In,
                                    α::Number, β::Number)
                A = At.lmap
                return (issymmetric(A) || (isreal(A) && ishermitian(A))) ?
                    _unsafe_mul!(y, A.lmap, x, α, β) :
                    _unsafe_mul!(y, transpose(A.lmap), x, α, β)
            end
            function _unsafe_mul!(y::$Out, Ac::AdjointMap{<:Any,<:WrappedMap}, x::$In, α::Number, β::Number)
                A = Ac.lmap
                return ishermitian(A) ?
                    _unsafe_mul!(y, A.lmap, x, α, β) :
                    _unsafe_mul!(y, adjoint(A.lmap), x, α, β)
            end
        end
    end
end # VERSION

# combine LinearMap and Matrix objects: linear combinations and map composition
Base.:(+)(A₁::LinearMap, A₂::AbstractMatrix) = +(A₁, WrappedMap(A₂))
Base.:(+)(A₁::AbstractMatrix, A₂::LinearMap) = +(WrappedMap(A₁), A₂)
Base.:(-)(A₁::LinearMap, A₂::AbstractMatrix) = -(A₁, WrappedMap(A₂))
Base.:(-)(A₁::AbstractMatrix, A₂::LinearMap) = -(WrappedMap(A₁), A₂)

"""
    *(A::LinearMap, X::AbstractMatrix)::CompositeMap

Return the `CompositeMap` `A*LinearMap(X)`, interpreting the matrix `X` as a linear
operator, rather than a collection of column vectors. To compute the action of `A` on each
column of `X`, call `Matrix(A*X)` or use the in-place multiplication `mul!(Y, A, X[, α, β])`
with an appropriately sized, preallocated matrix `Y`.

## Examples
```jldoctest; setup=(using LinearMaps)
julia> A=LinearMap([1.0 2.0; 3.0 4.0]); X=[1.0 1.0; 1.0 1.0];

julia> A*X isa LinearMaps.CompositeMap
true
```
"""
Base.:(*)(A₁::LinearMap, A₂::AbstractMatrix) = *(A₁, WrappedMap(A₂))

"""
    *(X::AbstractMatrix, A::LinearMap)::CompositeMap

Return the `CompositeMap` `LinearMap(X)*A`, interpreting the matrix `X` as a linear
operator. To compute the right-action of `A` on each row of `X`, call `Matrix(X*A)`.

## Examples
```jldoctest; setup=(using LinearMaps)
julia> X=[1.0 1.0; 1.0 1.0]; A=LinearMap([1.0 2.0; 3.0 4.0]);

julia> X*A isa LinearMaps.CompositeMap
true
```
"""
Base.:(*)(A₁::AbstractMatrix, A₂::LinearMap) = *(WrappedMap(A₁), A₂)
