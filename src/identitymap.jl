immutable IdentityMap{T}<:AbstractLinearMap{T} # T will be determined from maps to which I is added
    M::Int
end
IdentityMap(T::Type,M::Int)=IdentityMap{T}(M)
IdentityMap(T::Type,M::Int,N::Int)=(M==N ? IdentityMap{T}(M) : error("IdenityMap needs to be square"))
IdentityMap(T::Type,sz::Tuple{Int,Int})=(sz[1]==sz[2] ? IdentityMap{T}(sz[1]) : error("IdenityMap needs to be square"))
IdentityMap(M::Int)=IdentityMap(Bool,M)
IdentityMap(M::Int,N::Int)=IdentityMap(Bool,M,N)
IdentityMap(sz::Tuple{Int,Int})=IdentityMap(Bool,sz)

# properties
Base.size(A::IdentityMap,n)=(n==1 || n==2 ? A.M : error("AbstractLinearMap objects have only 2 dimensions"))
Base.size(A::IdentityMap)=(A.M,A.M)
Base.isreal(::IdentityMap)=true
Base.issym(::IdentityMap)=true
Base.ishermitian(::IdentityMap)=true
Base.isposdef(::IdentityMap)=true

# multiplication with vector
Base.A_mul_B!(y::AbstractVector,A::IdentityMap,x::AbstractVector)=(length(x)==length(y)==A.M ? copy!(y,x) : throw(DimensionMismatch("A_mul_B!")))
*(A::IdentityMap,x::AbstractVector)=x

Base.At_mul_B!(y::AbstractVector,A::IdentityMap,x::AbstractVector)=(length(x)==length(y)==A.M ? copy!(y,x) : throw(DimensionMismatch("At_mul_B!")))
Base.At_mul_B(A::IdentityMap,x::AbstractVector)=(length(x)==A.M ? x : throw(DimensionMismatch("At_mul_B")))

Base.Ac_mul_B!(y::AbstractVector,A::IdentityMap,x::AbstractVector)=(length(x)==length(y)==A.M ? copy!(y,x) : throw(tMismatch("Ac_mul_B!")))
Base.Ac_mul_B(A::IdentityMap,x::AbstractVector)=(length(x)==A.M ? x : throw(DimensionMismatch("Ac_mul_B")))

# combine AbstractLinearMap and UniformScaling objects in linear combinations
+{T}(A1::AbstractLinearMap,A2::UniformScaling{T})=A1+A2[1,1]*IdentityMap{T}(size(A1,1))
+{T}(A1::UniformScaling{T},A2::AbstractLinearMap)=A1[1,1]*IdentityMap{T}(size(A2,1))+A2
