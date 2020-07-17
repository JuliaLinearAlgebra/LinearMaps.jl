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
Base.@propagate_inbounds function LinearAlgebra.mul!(y::AbstractVector, A::TransposeMap, x::AbstractVector)
    issymmetric(A.lmap) ? mul!(y, A.lmap, x) : error("transpose not implemented for $A")
end
Base.@propagate_inbounds function LinearAlgebra.mul!(y::AbstractVector, A::TransposeMap, x::AbstractVector, α::Number, β::Number)
    issymmetric(A.lmap) ? mul!(y, A.lmap, x, α, β) : error("transpose not implemented for $A")
end

Base.@propagate_inbounds function LinearAlgebra.mul!(y::AbstractVector, A::TransposeMap{<:Any,<:TransposeMap}, x::AbstractVector)
    mul!(y, A.lmap.lmap, x)
end
Base.@propagate_inbounds function LinearAlgebra.mul!(y::AbstractVector, A::TransposeMap{<:Any,<:TransposeMap}, x::AbstractVector, α::Number, β::Number)
    mul!(y, A.lmap.lmap, x, α, β)
end

Base.@propagate_inbounds function LinearAlgebra.mul!(y::AbstractVector, Ac::AdjointMap{<:Any,<:TransposeMap}, x::AbstractVector)
    A = Ac.lmap
    return _conjmul!(y, A.lmap, x)
end
Base.@propagate_inbounds function LinearAlgebra.mul!(y::AbstractVector, Ac::AdjointMap{<:Any,<:TransposeMap}, x::AbstractVector, α::Number, β::Number)
    A = Ac.lmap
    return _conjmul!(y, A.lmap, x, α, β)
end

Base.@propagate_inbounds function LinearAlgebra.mul!(y::AbstractVector, A::AdjointMap, x::AbstractVector)
    ishermitian(A.lmap) ? mul!(y, A.lmap, x) : error("adjoint not implemented for $A")
end
Base.@propagate_inbounds function LinearAlgebra.mul!(y::AbstractVector, A::AdjointMap, x::AbstractVector, α::Number, β::Number)
    ishermitian(A.lmap) ? mul!(y, A.lmap, x, α, β) : error("adjoint not implemented for $A")
end

Base.@propagate_inbounds function LinearAlgebra.mul!(y::AbstractVector, At::TransposeMap{<:Any,<:AdjointMap}, x::AbstractVector)
    A = At.lmap
    isreal(A.lmap) ? mul!(y, A.lmap, x) : (mul!(y, A.lmap, conj(x)); conj!(y))
end
Base.@propagate_inbounds function LinearAlgebra.mul!(y::AbstractVector, At::TransposeMap{<:Any,<:AdjointMap}, x::AbstractVector, α::Number, β::Number)
    A = At.lmap
    return _conjmul!(y, A.lmap, x, α, β)
end

Base.@propagate_inbounds function LinearAlgebra.mul!(y::AbstractVector, A::AdjointMap{<:Any,<:AdjointMap}, x::AbstractVector)
    mul!(y, A.lmap.lmap, x)
end
Base.@propagate_inbounds function LinearAlgebra.mul!(y::AbstractVector, A::AdjointMap{<:Any,<:AdjointMap}, x::AbstractVector, α::Number, β::Number)
    mul!(y, A.lmap.lmap, x, α, β)
end

# multiplication helper function
function _conjmul!(y, A, x)
    if isreal(A)
        return mul!(y, A, x)
    else
        mul!(y, A, conj(x))
        return conj!(y)
    end
end
function _conjmul!(y, A, x, α, β)
    if isreal(A)
        mul!(y, A, x, α, β)
    else
        xca = rmul!(conj(x), α)
        z = A*xca
        rmul!(y, β)
        y .+= conj!(z)
        return y
    end
end
