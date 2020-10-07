struct TransposeMap{T, A<:LinearMap{T}} <: LinearMap{T}
    lmap::A
end
struct AdjointMap{T, A<:LinearMap{T}} <: LinearMap{T}
    lmap::A
end

MulStyle(A::Union{TransposeMap,AdjointMap}) = MulStyle(A.lmap)

# transposition behavior of LinearMap objects
LinearAlgebra.transpose(A::TransposeMap) = A.lmap
LinearAlgebra.adjoint(A::AdjointMap) = A.lmap

Base.parent(A::Union{AdjointMap,TransposeMap}) = A.lmap

"""
    transpose(A::LinearMap)

Construct a lazy representation of the transpose of `A`. This can be either a
`TransposeMap` wrapper of `A`, or a suitably redefined instance of the same type
as `A`. For instance, for a linear combination of linear maps ``A + B``, the transpose
is given by ``A^⊤ + B^⊤``, i.e., another linear combination of linear maps.
"""
LinearAlgebra.transpose(A::LinearMap) = TransposeMap(A)

"""
    adjoint(A::LinearMap)

Construct a lazy representation of the adjoint of `A`. This can be either a
`AdjointMap` wrapper of `A`, or a suitably redefined instance of the same type
as `A`. For instance, for a linear combination of linear maps ``A + B``, the adjoint
is given by ``A^* + B^*``, i.e., another linear combination of linear maps.
"""
LinearAlgebra.adjoint(A::LinearMap) = AdjointMap(A)
LinearAlgebra.adjoint(A::LinearMap{<:Real}) = transpose(A)

# properties
Base.size(A::Union{TransposeMap, AdjointMap}) = reverse(size(A.lmap))
LinearAlgebra.issymmetric(A::Union{TransposeMap, AdjointMap}) = issymmetric(A.lmap)
LinearAlgebra.ishermitian(A::Union{TransposeMap, AdjointMap}) = ishermitian(A.lmap)
LinearAlgebra.isposdef(A::Union{TransposeMap, AdjointMap}) = isposdef(A.lmap)

# comparison of TransposeMap objects
Base.:(==)(A::TransposeMap, B::TransposeMap) = A.lmap == B.lmap
Base.:(==)(A::AdjointMap, B::AdjointMap)     = A.lmap == B.lmap
Base.:(==)(A::TransposeMap, B::AdjointMap)   = false # isreal(B) && A.lmap == B.lmap # isreal(::AdjointMap) == false
Base.:(==)(A::AdjointMap, B::TransposeMap)   = false # isreal(A) && A.lmap == B.lmap # isreal(::AdjointMap) == false
Base.:(==)(A::TransposeMap, B::LinearMap)    = issymmetric(B) && A.lmap == B
Base.:(==)(A::AdjointMap, B::LinearMap)      = ishermitian(B) && A.lmap == B
Base.:(==)(A::LinearMap, B::TransposeMap)    = issymmetric(A) && B.lmap == A
Base.:(==)(A::LinearMap, B::AdjointMap)      = ishermitian(A) && B.lmap == A

# multiplication with vector/matrices
# # TransposeMap
function _unsafe_mul!(y::AbstractVecOrMat, A::TransposeMap, x::AbstractVector)
    issymmetric(A.lmap) ? _unsafe_mul!(y, A.lmap, x) : error("transpose not implemented for $(A.lmap)")
end
function _unsafe_mul!(y::AbstractMatrix, A::TransposeMap, x::AbstractMatrix)
    issymmetric(A.lmap) ? _unsafe_mul!(y, A.lmap, x) : _generic_mapmat_mul!(y, A, x)
end
function _unsafe_mul!(y::AbstractVecOrMat, A::TransposeMap, x::AbstractVector, α::Number, β::Number)
    issymmetric(A.lmap) ? _unsafe_mul!(y, A.lmap, x, α, β) : _generic_mapvec_mul!(y, A, x, α, β)
end
function _unsafe_mul!(y::AbstractMatrix, A::TransposeMap, x::AbstractMatrix, α::Number, β::Number)
    issymmetric(A.lmap) ? _unsafe_mul!(y, A.lmap, x, α, β) : _generic_mapmat_mul!(y, A, x, α, β)
end
# # AdjointMap
function _unsafe_mul!(y::AbstractVecOrMat, A::AdjointMap, x::AbstractVector)
    ishermitian(A.lmap) ? _unsafe_mul!(y, A.lmap, x) : error("adjoint not implemented for $(A.lmap)")
end
function _unsafe_mul!(y::AbstractMatrix, A::AdjointMap, x::AbstractMatrix)
    ishermitian(A.lmap) ? _unsafe_mul!(y, A.lmap, x) : _generic_mapmat_mul!(y, A, x)
end
function _unsafe_mul!(y::AbstractVecOrMat, A::AdjointMap, x::AbstractVector, α::Number, β::Number)
    ishermitian(A.lmap) ? _unsafe_mul!(y, A.lmap, x, α, β) : _generic_mapvec_mul!(y, A, x, α, β)
end
function _unsafe_mul!(y::AbstractMatrix, A::AdjointMap, x::AbstractMatrix, α::Number, β::Number)
    ishermitian(A.lmap) ? _unsafe_mul!(y, A.lmap, x, α, β) : _generic_mapmat_mul!(y, A, x, α, β)
end
# # ConjugateMap
for (intype, outtype) in ((AbstractVector, AbstractVecOrMat), (AbstractMatrix, AbstractMatrix))
    @eval begin
        function _unsafe_mul!(y::$outtype, Ac::AdjointMap{<:Any,<:TransposeMap}, x::$intype)
            return _conjmul!(y, Ac.lmap.lmap, x)
        end
        function _unsafe_mul!(y::$outtype, Ac::AdjointMap{<:Any,<:TransposeMap}, x::$intype, α::Number, β::Number)
            return _conjmul!(y, Ac.lmap.lmap, x, α, β)
        end
        function _unsafe_mul!(y::$outtype, At::TransposeMap{<:Any,<:AdjointMap}, x::$intype)
            return _conjmul!(y, At.lmap.lmap, x)
        end
        function _unsafe_mul!(y::$outtype, At::TransposeMap{<:Any,<:AdjointMap}, x::$intype, α::Number, β::Number)
            return _conjmul!(y, At.lmap.lmap, x, α, β)
        end
    end
end

# multiplication helper function
_conjmul!(y, A, x) = conj!(mul!(y, A, conj(x)))
function _conjmul!(y, A, x::AbstractVector, α, β)
    xca = conj!(x * α)
    z = A*xca
    y .= y .* β + conj.(z)
    return y
end
function _conjmul!(y, A, x::AbstractMatrix, α, β)
    xca = conj!(x * α)
    z = similar(y, size(A, 1), size(x, 2))
    _unsafe_mul!(z, A, xca)
    y .= y .* β + conj.(z)
    return y
end
