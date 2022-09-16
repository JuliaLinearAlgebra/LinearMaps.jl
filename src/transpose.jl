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
Base.axes(A::Union{TransposeMap, AdjointMap}) = reverse(axes(A.lmap))
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
for (Typ, prop, text) in ((AdjointMap, ishermitian, "adjoint"), (TransposeMap, issymmetric, "transpose"))
    @eval _unsafe_mul!(y, A::$Typ, x::AbstractVector) =
        $prop(A.lmap) ?
            _unsafe_mul!(y, A.lmap, x) : error($text * " not implemented for $(A.lmap)")
    @eval _unsafe_mul!(y, A::$Typ, x::AbstractVector, α, β) =
        $prop(A.lmap) ?
            _unsafe_mul!(y, A.lmap, x, α, β) : _generic_map_mul!(y, A, x, α, β)

    for In in (Number, AbstractMatrix)
        @eval _unsafe_mul!(y, A::$Typ, x::$In) =
            $prop(A.lmap) ?
                _unsafe_mul!(y, A.lmap, x) : _generic_map_mul!(y, A, x)

        @eval _unsafe_mul!(y, A::$Typ, x::$In, α, β) =
            ishermitian(A.lmap) ?
                _unsafe_mul!(y, A.lmap, x, α, β) : _generic_map_mul!(y, A, x, α, β)
    end
end

# # ConjugateMap
const ConjugateMap = AdjointMap{<:Any, <:TransposeMap}
# canonical order of adjoint followed by transpose
LinearAlgebra.transpose(A::AdjointMap) = adjoint(transpose(A.lmap))
LinearAlgebra.transpose(A::ConjugateMap) = adjoint(A.lmap.lmap)
for In in (AbstractVector, AbstractMatrix)
    @eval begin
        _unsafe_mul!(y, Ac::ConjugateMap, x::$In) = conj!(_unsafe_mul!(y, Ac.lmap.lmap, conj(x)))
        function _unsafe_mul!(y, Ac::ConjugateMap, x::$In, α, β)
            xca = conj!(x * α)
            if !iszero(β)
                z = similar(y)
                _unsafe_mul!(z, Ac.lmap.lmap, xca)
                y .= y .* β + conj.(z)
            else
                conj!(_unsafe_mul!(y, Ac.lmap.lmap, xca))
            end
            return y
        end
    end
end
_unsafe_mul!(y, Ac::ConjugateMap, x::Number) = conj!(_unsafe_mul!(y, Ac.lmap.lmap, conj(x)))
