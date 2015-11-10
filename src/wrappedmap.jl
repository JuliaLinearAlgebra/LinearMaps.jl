immutable WrappedMap{T}<:AbstractLinearMap{T}
    lmap::Union(AbstractMatrix{T},AbstractLinearMap{T})
    _isreal::Bool
    _issym::Bool
    _ishermitian::Bool
    _isposdef::Bool
    WrappedMap(A::Union(AbstractMatrix{T},AbstractLinearMap{T});isreal::Bool=Base.isreal(A),issym::Bool=Base.issym(A),ishermitian::Bool=Base.ishermitian(A),isposdef::Bool=Base.isposdef(A))=new(A,isreal,issym,ishermitian,isposdef)
end
WrappedMap{T}(A::Union(AbstractMatrix{T},AbstractLinearMap{T});isreal::Bool=Base.isreal(A),issym::Bool=Base.issym(A),ishermitian::Bool=Base.ishermitian(A),isposdef::Bool=Base.isposdef(A))=WrappedMap{T}(A;isreal=isreal,issym=issym,ishermitian=ishermitian,isposdef=isposdef)

# properties
Base.size(A::WrappedMap,n)=(n==1 || n==2 ? size(A.lmap,n) : error("AbstractLinearMap objects have only 2 dimensions"))
Base.size(A::WrappedMap)=size(A.lmap)
Base.isreal(A::WrappedMap)=A._isreal
Base.issym(A::WrappedMap)=A._issym
Base.ishermitian(A::WrappedMap)=A._ishermitian
Base.isposdef(A::WrappedMap)=A._isposdef

# comparison
==(A::WrappedMap,B::WrappedMap)=(A.lmap==B.lmap && isreal(A)==isreal(B) && issym(A)==issym(B) && ishermitian(A)==ishermitian(B) && isposdef(A)==isposdef(B))

# multiplication with vector
Base.A_mul_B!(y::AbstractVector,A::WrappedMap,x::AbstractVector)=Base.A_mul_B!(y,A.lmap,x)
*(A::WrappedMap,x::AbstractVector)=*(A.lmap,x)

Base.At_mul_B!(y::AbstractVector,A::WrappedMap,x::AbstractVector)=Base.At_mul_B!(y,A.lmap,x)
Base.At_mul_B(A::WrappedMap,x::AbstractVector)=Base.At_mul_B(A.lmap,x)

Base.Ac_mul_B!(y::AbstractVector,A::WrappedMap,x::AbstractVector)=Base.Ac_mul_B!(y,A.lmap,x)
Base.Ac_mul_B(A::WrappedMap,x::AbstractVector)=Base.Ac_mul_B(A.lmap,x)

# combine AbstractLinearMap and Matrix objects: linear combinations and map composition
+(A1::AbstractLinearMap,A2::AbstractMatrix)=+(A1,WrappedMap(A2))
+(A1::AbstractMatrix,A2::AbstractLinearMap)=+(WrappedMap(A1),A2)

*(A1::AbstractLinearMap,A2::AbstractMatrix)=*(A1,WrappedMap(A2))
*(A1::AbstractMatrix,A2::AbstractLinearMap)=*(WrappedMap(A1),A2)
