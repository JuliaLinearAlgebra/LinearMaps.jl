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

# multiplication with vector/matrices
# # TransposeMap
Base.@propagate_inbounds function mul!(y::AbstractVecOrMat, A::TransposeMap, x::AbstractVector)
    @boundscheck check_dim_mul(y, A, x)
    @inbounds issymmetric(A.lmap) ? mul!(y, A.lmap, x) : error("transpose not implemented for $A")
end
Base.@propagate_inbounds function mul!(y::AbstractMatrix, A::TransposeMap, x::AbstractMatrix)
    @boundscheck check_dim_mul(y, A, x)
    @inbounds issymmetric(A.lmap) ? mul!(y, A.lmap, x) : _generic_mapmat_mul!(y, A, x)
end
Base.@propagate_inbounds function mul!(y::AbstractVecOrMat, A::TransposeMap, x::AbstractVector, α::Number, β::Number)
    @boundscheck check_dim_mul(y, A, x)
    @inbounds issymmetric(A.lmap) ? mul!(y, A.lmap, x, α, β) : _generic_mapvec_mul!(y, A, x, α, β)
end
Base.@propagate_inbounds function mul!(y::AbstractMatrix, A::TransposeMap, x::AbstractMatrix, α::Number, β::Number)
    @boundscheck check_dim_mul(y, A, x)
    @inbounds issymmetric(A.lmap) ? mul!(y, A.lmap, x, α, β) : _generic_mapmat_mul!(y, A, x, α, β)
end
# # AdjointMap
Base.@propagate_inbounds function mul!(y::AbstractVecOrMat, A::AdjointMap, x::AbstractVector)
    @boundscheck check_dim_mul(y, A, x)
    @inbounds issymmetric(A.lmap) ? mul!(y, A.lmap, x) : error("adjoint not implemented for $A")
end
Base.@propagate_inbounds function mul!(y::AbstractMatrix, A::AdjointMap, x::AbstractMatrix)
    @boundscheck check_dim_mul(y, A, x)
    @inbounds issymmetric(A.lmap) ? mul!(y, A.lmap, x) : _generic_mapmat_mul!(y, A, x)
end
Base.@propagate_inbounds function mul!(y::AbstractVecOrMat, A::AdjointMap, x::AbstractVector, α::Number, β::Number)
    @boundscheck check_dim_mul(y, A, x)
    @inbounds issymmetric(A.lmap) ? mul!(y, A.lmap, x, α, β) : _generic_mapvec_mul!(y, A, x, α, β)
end
Base.@propagate_inbounds function mul!(y::AbstractMatrix, A::AdjointMap, x::AbstractMatrix, α::Number, β::Number)
    @boundscheck check_dim_mul(y, A, x)
    @inbounds issymmetric(A.lmap) ? mul!(y, A.lmap, x, α, β) : _generic_mapmat_mul!(y, A, x, α, β)
end
# # ConjugateMap
for (intype, outtype) in Any[Any[AbstractVector, AbstractVecOrMat], Any[AbstractMatrix, AbstractMatrix]]
    @eval begin
        Base.@propagate_inbounds function mul!(y::$outtype, Ac::AdjointMap{<:Any,<:TransposeMap}, x::$intype)
            @boundscheck check_dim_mul(y, Ac, x)
            A = Ac.lmap
            return @inbounds _conjmul!(y, A.lmap, x)
        end
        Base.@propagate_inbounds function mul!(y::$outtype, Ac::AdjointMap{<:Any,<:TransposeMap}, x::$intype, α::Number, β::Number)
            @boundscheck check_dim_mul(y, Ac, x)
            A = Ac.lmap
            return @inbounds _conjmul!(y, A.lmap, x, α, β)
        end
        Base.@propagate_inbounds function mul!(y::$outtype, At::TransposeMap{<:Any,<:AdjointMap}, x::$intype)
            @boundscheck check_dim_mul(y, At, x)
            A = At.lmap
            @inbounds isreal(A.lmap) ? mul!(y, A.lmap, x) : _conjmul!(y, A.lmap, x)
        end
        Base.@propagate_inbounds function mul!(y::$outtype, At::TransposeMap{<:Any,<:AdjointMap}, x::$intype, α::Number, β::Number)
            @boundscheck check_dim_mul(y, At, x)
            A = At.lmap
            return @inbounds _conjmul!(y, A.lmap, x, α, β)
        end
    end
end

# multiplication helper function
Base.@propagate_inbounds _conjmul!(y, A, x) = (mul!(y, A, conj(x)); conj!(y))
Base.@propagate_inbounds function _conjmul!(y, A, x::AbstractVector, α, β)
    xca = rmul!(conj(x), conj(α))
    z = A*xca
    rmul!(y, β)
    y .+= conj.(z)
    return y
end
Base.@propagate_inbounds function _conjmul!(y, A, x::AbstractMatrix, α, β)
    xca = rmul!(conj(x), conj(α))
    z = similar(y, size(A, 1), size(x, 2))
    mul!(z, A, xca)
    rmul!(y, β)
    y .+= conj.(z)
    return y
end
