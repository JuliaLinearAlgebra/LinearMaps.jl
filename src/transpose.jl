immutable TransposeMap{T}<:LinearMap{T}
    lmap::LinearMap{T}
end
immutable CTransposeMap{T}<:LinearMap{T}
    lmap::LinearMap{T}
end

# transposition behavior of LinearMap objects
transpose(A::TransposeMap)=A.lmap
ctranspose(A::CTransposeMap)=A.lmap

transpose{T}(A::LinearMap{T})=issym(A) ? A : TransposeMap{T}(A)
ctranspose{T<:Real}(A::LinearMap{T})=transpose(A)
ctranspose{T}(A::LinearMap{T})=ishermitian(A) ? A : CTransposeMap{T}(A)

# properties
Base.size(A::Union(TransposeMap,CTransposeMap),n)=(n==1 ? size(A.lmap,2) : (n==2 ? size(A.lmap,1) : error("LinearMap objects have only 2 dimensions")))
Base.size(A::Union(TransposeMap,CTransposeMap))=(size(A.lmap,2),size(A.lmap,1))
Base.isreal(A::Union(TransposeMap,CTransposeMap))=isreal(A.lmap)
Base.issym(A::Union(TransposeMap,CTransposeMap))=issym(A.lmap)
Base.ishermitian(A::Union(TransposeMap,CTransposeMap))=ishermitian(A.lmap)
Base.isposdef(A::Union(TransposeMap,CTransposeMap))=isposdef(A.lmap)

# comparison of TransposeMap objects
==(A::TransposeMap,B::TransposeMap)=A.lmap==B.lmap
==(A::CTransposeMap,B::CTransposeMap)=A.lmap==B.lmap

# multiplication with vector: should be ok for all lmap values, also those for which Transpose/CTranpose should not have been created
Base.A_mul_B!(y::AbstractVector,A::TransposeMap,x::AbstractVector)=(issym(A.lmap) ? Base.A_mul_B!(y,A.lmap,x) : Base.At_mul_B!(y,A.lmap,x))
*(A::TransposeMap,x::AbstractVector)=(issym(A.lmap) ? *(A.lmap,x) : Base.At_mul_B(A.lmap,x))

Base.At_mul_B!(y::AbstractVector,A::TransposeMap,x::AbstractVector)=Base.A_mul_B!(y,A.lmap,x)
Base.At_mul_B(A::TransposeMap,x::AbstractVector)=*(A.lmap,x)

Base.Ac_mul_B!(y::AbstractVector,A::TransposeMap,x::AbstractVector)=isreal(A.lmap) ? Base.A_mul_B!(y,A.lmap,x) : (Base.A_mul_B!(y,A.lmap,conj(x));conj!(y))
Base.Ac_mul_B(A::TransposeMap,x::AbstractVector)=isreal(A.lmap) ? *(A.lmap,x) : conj!(*(A.lmap,conj(x)))

Base.A_mul_B!(y::AbstractVector,A::CTransposeMap,x::AbstractVector)=(ishermitian(A.lmap) ? Base.A_mul_B!(y,A.lmap,x) : Base.Ac_mul_B!(y,A.lmap,x))
*(A::CTransposeMap,x::AbstractVector)=(ishermitian(A.lmap) ? *(A.lmap,x) : Base.Ac_mul_B(A.lmap,x))

Base.At_mul_B!(y::AbstractVector,A::CTransposeMap,x::AbstractVector)=isreal(A.lmap) ? Base.A_mul_B!(y,A.lmap,x) : (Base.A_mul_B!(y,A.lmap,conj(x));conj!(y))
Base.At_mul_B(A::CTransposeMap,x::AbstractVector)=isreal(A.lmap) ? *(A.lmap,x) : conj!(*(A.lmap,conj(x)))

Base.Ac_mul_B!(y::AbstractVector,A::CTransposeMap,x::AbstractVector)=Base.A_mul_B!(y,A.lmap,x)
Base.Ac_mul_B(A::CTransposeMap,x::AbstractVector)=*(A.lmap,x)

