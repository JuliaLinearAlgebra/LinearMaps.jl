struct WrappedMap{T, A<:MapOrVecOrMat} <: LinearMap{T}
    lmap::A
    _issymmetric::Bool
    _ishermitian::Bool
    _isposdef::Bool
end
function WrappedMap{T}(lmap::MapOrMatrix;
                        issymmetric::Bool = _issymmetric(lmap),
                        ishermitian::Bool = _ishermitian(lmap),
                        isposdef::Bool = _isposdef(lmap)) where {T}
    WrappedMap{T, typeof(lmap)}(lmap, issymmetric, ishermitian, isposdef)
end
function WrappedMap{T}(lmap::AbstractVector;
                        issym::Bool = false,
                        isherm::Bool = false,
                        ispd::Bool = false) where {T}
    WrappedMap{T, typeof(lmap)}(lmap,
                                length(lmap) == 1 && issymmetric(first(lmap)),
                                length(lmap) == 1 && ishermitian(first(lmap)),
                                length(lmap) == 1 && isposdef(first(lmap)))
end
WrappedMap(lmap::MapOrVecOrMat{T}; kwargs...) where {T} = WrappedMap{T}(lmap; kwargs...)

# cheap property checks (usually by type)
_issymmetric(A::AbstractMatrix) = false
_issymmetric(A::AbstractQ) = false
_issymmetric(A::LinearMap) = issymmetric(A)
_issymmetric(A::LinearAlgebra.RealHermSymComplexSym) = issymmetric(A)
_issymmetric(A::Union{Bidiagonal,Diagonal,SymTridiagonal,Tridiagonal}) = issymmetric(A)

_ishermitian(A::AbstractMatrix) = false
_ishermitian(A::AbstractQ) = false
_ishermitian(A::LinearMap) = ishermitian(A)
_ishermitian(A::LinearAlgebra.RealHermSymComplexHerm) = ishermitian(A)
_ishermitian(A::Union{Bidiagonal,Diagonal,SymTridiagonal,Tridiagonal}) = ishermitian(A)

_isposdef(A::AbstractMatrix) = false
_isposdef(A::AbstractQ) = false
_isposdef(A::LinearMap) = isposdef(A)

const VecOrMatMap{T} = WrappedMap{T,<:AbstractVecOrMatOrQ}

MulStyle(A::VecOrMatMap) = MulStyle(A.lmap)

LinearAlgebra.transpose(A::VecOrMatMap{T}) where {T} =
    WrappedMap{T}(transpose(A.lmap);
                    issymmetric = A._issymmetric,
                    ishermitian = A._ishermitian,
                    isposdef = A._isposdef)
LinearAlgebra.adjoint(A::VecOrMatMap{T}) where {T} =
    WrappedMap{T}(adjoint(A.lmap);
                    issymmetric = A._issymmetric,
                    ishermitian = A._ishermitian,
                    isposdef = A._isposdef)

Base.:(==)(A::VecOrMatMap, B::VecOrMatMap) =
    (eltype(A) == eltype(B) && A.lmap == B.lmap && A._issymmetric == B._issymmetric &&
     A._ishermitian == B._ishermitian && A._isposdef == B._isposdef)

# properties
Base.size(A::WrappedMap) = size(A.lmap)
Base.size(A::WrappedMap{<:Any,<:AbstractVector}) = (Int(length(A.lmap))::Int, 1)
Base.axes(A::WrappedMap) = axes(A.lmap)
LinearAlgebra.issymmetric(A::WrappedMap) = A._issymmetric
LinearAlgebra.ishermitian(A::WrappedMap) = A._ishermitian
LinearAlgebra.isposdef(A::WrappedMap) = A._isposdef

# multiplication with vectors & matrices
for In in (AbstractVector, AbstractMatrix)
    @eval begin
        _unsafe_mul!(y, A::WrappedMap, x::$In) = _unsafe_mul!(y, A.lmap, x)
        function _unsafe_mul!(y, At::TransposeMap{<:Any,<:WrappedMap}, x::$In)
            A = At.lmap
            return (issymmetric(A) || (isreal(A) && ishermitian(A))) ?
                _unsafe_mul!(y, A.lmap, x) :
                _unsafe_mul!(y, transpose(A.lmap), x)
        end
        function _unsafe_mul!(y, Ac::AdjointMap{<:Any,<:WrappedMap}, x::$In)
            A = Ac.lmap
            return ishermitian(A) ?
                _unsafe_mul!(y, A.lmap, x) :
                _unsafe_mul!(y, adjoint(A.lmap), x)
        end
    end
end

for In in (AbstractVector, AbstractMatrix)
    @eval begin
        function _unsafe_mul!(y, A::WrappedMap, x::$In, α, β)
            return _unsafe_mul!(y, A.lmap, x, α, β)
        end
        function _unsafe_mul!(y, At::TransposeMap{<:Any,<:WrappedMap}, x::$In, α, β)
            A = At.lmap
            return (issymmetric(A) || (isreal(A) && ishermitian(A))) ?
                _unsafe_mul!(y, A.lmap, x, α, β) :
                _unsafe_mul!(y, transpose(A.lmap), x, α, β)
        end
        function _unsafe_mul!(y, Ac::AdjointMap{<:Any,<:WrappedMap}, x::$In, α, β)
            A = Ac.lmap
            return ishermitian(A) ?
                _unsafe_mul!(y, A.lmap, x, α, β) :
                _unsafe_mul!(y, adjoint(A.lmap), x, α, β)
        end
    end
end

_unsafe_mul!(Y, A::VecOrMatMap, s::Number) = _unsafe_mul!(Y, A.lmap, s)
_unsafe_mul!(Y, A::VecOrMatMap, s::Number, α, β) = _unsafe_mul!(Y, A.lmap, s, α, β)

# combine LinearMap and Matrix objects: linear combinations and map composition
Base.:(+)(A₁::LinearMap, A₂::Union{AbstractMatrix,AbstractQ}) = +(A₁, WrappedMap(A₂))
Base.:(+)(A₁::Union{AbstractMatrix,AbstractQ}, A₂::LinearMap) = +(WrappedMap(A₁), A₂)
Base.:(-)(A₁::LinearMap, A₂::Union{AbstractMatrix,AbstractQ}) = -(A₁, WrappedMap(A₂))
Base.:(-)(A₁::Union{AbstractMatrix,AbstractQ}, A₂::LinearMap) = -(WrappedMap(A₁), A₂)

"""
    *(A::LinearMap, X::Union{AbstractMatrix,AbstractQ})::CompositeMap

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
Base.:(*)(A₁::LinearMap, A₂::Union{AbstractMatrix,AbstractQ}) = *(A₁, WrappedMap(A₂))

"""
    *(X::Union{AbstractMatrix,AbstractQ}, A::LinearMap)::CompositeMap

Return the `CompositeMap` `LinearMap(X)*A`, interpreting the matrix `X` as a linear
operator. To compute the right-action of `A` on each row of `X`, call `Matrix(X*A)`
or `mul!(Y, X, A)` for the in-place version.

## Examples
```jldoctest; setup=(using LinearMaps)
julia> X=[1.0 1.0; 1.0 1.0]; A=LinearMap([1.0 2.0; 3.0 4.0]);

julia> X*A isa LinearMaps.CompositeMap
true
```
"""
Base.:(*)(A₁::Union{AbstractMatrix,AbstractQ}, A₂::LinearMap) = *(WrappedMap(A₁), A₂)
