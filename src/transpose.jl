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

LinearAlgebra.transpose(A::LinearMap) = TransposeMap(A)
LinearAlgebra.adjoint(A::LinearMap{<:Real}) = transpose(A)
LinearAlgebra.adjoint(A::LinearMap) = AdjointMap(A)

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

# multiplication with vector
Base.@propagate_inbounds LinearAlgebra.mul!(y::AbstractVector, A::TransposeMap, x::AbstractVector,
                    α::Number=true, β::Number=false) =
    issymmetric(A.lmap) ? mul!(y, A.lmap, x, α, β) : error("transpose not defined for $(typeof(A.lmap))")

Base.@propagate_inbounds LinearAlgebra.mul!(y::AbstractVector, A::AdjointMap{<:Any,<:TransposeMap}, x::AbstractVector,
                    α::Number=true, β::Number=false) =
    isreal(A.lmap) ? mul!(y, A.lmap.lmap, x, α, β) : (mul!(y, A.lmap.lmap, conj(x), conj(α), conj(β)); conj!(y))

Base.@propagate_inbounds LinearAlgebra.mul!(y::AbstractVector, A::AdjointMap, x::AbstractVector,
                    α::Number=true, β::Number=false) =
    ishermitian(A.lmap) ? mul!(y, A.lmap, x) : error("adjoint not defined for $(typeof(A.lmap))")

Base.@propagate_inbounds LinearAlgebra.mul!(y::AbstractVector, A::TransposeMap{<:Any,<:AdjointMap}, x::AbstractVector,
                    α::Number=true, β::Number=false) =
    isreal(A.lmap) ? mul!(y, A.lmap.lmap, x, α, β) : (mul!(y, A.lmap.lmap, conj(x), conj(α), conj(β)); conj!(y))
