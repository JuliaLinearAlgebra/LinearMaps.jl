module LinearMaps

export AbstractLinearMap, LinearMap

abstract AbstractLinearMap{T}
Base.eltype{T}(::AbstractLinearMap{T})=T
Base.eltype{T}(::Type{AbstractLinearMap{T}})=T
Base.eltype{L<:AbstractLinearMap}(::Type{L})=eltype(super(L))

Base.isreal{T<:Real}(::AbstractLinearMap{T})=true
Base.isreal(::AbstractLinearMap)=false # standard assumptions
Base.issym(::AbstractLinearMap)=false # standard assumptions
Base.ishermitian{T<:Real}(A::AbstractLinearMap{T})=issym(A)
Base.ishermitian(::AbstractLinearMap)=false # standard assumptions
Base.isposdef(::AbstractLinearMap)=false # standard assumptions

Base.size(A::AbstractLinearMap,n)=(n==1 || n==2 ? size(A)[n] : error("AbstractLinearMap objects have only 2 dimensions"))

# any AbstractLinearMap subtype will have to overwrite at least one of the two following methods to avoid running in circles
*(A::AbstractLinearMap,x::AbstractVector)=Base.A_mul_B!(similar(x,promote_type(eltype(A),eltype(x)),size(A,1)),A,x)
Base.A_mul_B!(y::AbstractVector,A::AbstractLinearMap,x::AbstractVector)=begin
    length(y)==size(A,1) || throw(DimensionMismatch("A_mul_B!"))
    copy!(y,A*x)
end

# the following for multiplying with transpose and ctranspose map are optional:
# subtypes can overwrite nonmutating methods, implement mutating methods or do nothing
Base.At_mul_B(A::AbstractLinearMap,x::AbstractVector)=(@which Base.At_mul_B!(x,A,x))!=methods(Base.At_mul_B!,(AbstractVector,AbstractLinearMap,AbstractVector))[end] ? 
    Base.At_mul_B!(similar(x,promote_type(eltype(A),eltype(x)),size(A,2)),A,x) : throw(MethodError(Base.At_mul_B,(A,x)))
Base.At_mul_B!(y::AbstractVector,A::AbstractLinearMap,x::AbstractVector)=begin
    length(y)==size(A,2) || throw(DimensionMismatch("At_mul_B!"))
    (@which Base.At_mul_B(A,x))!=methods(Base.At_mul_B,(AbstractLinearMap,AbstractVector))[end] ? copy!(y,Base.At_mul_B(A,x)) : throw(MethodError(Base.At_mul_B!,(y,A,x)))
end
Base.Ac_mul_B(A::AbstractLinearMap,x::AbstractVector)=(@which Base.Ac_mul_B!(x,A,x))!=methods(Base.Ac_mul_B!,(AbstractVector,AbstractLinearMap,AbstractVector))[end] ? 
    Base.Ac_mul_B!(similar(x,promote_type(eltype(A),eltype(x)),size(A,2)),A,x) : throw(MethodError(Base.Ac_mul_B,(A,x)))
Base.Ac_mul_B!(y::AbstractVector,A::AbstractLinearMap,x::AbstractVector)=begin
    length(y)==size(A,2) || throw(DimensionMismatch("At_mul_B!"))
    (@which Base.Ac_mul_B(A,x))!=methods(Base.Ac_mul_B,(AbstractLinearMap,AbstractVector))[end] ? copy!(y,Base.Ac_mul_B(A,x)) : throw(MethodError(Base.Ac_mul_B!,(y,A,x)))
end

# full: create matrix representation of AbstractLinearMap
function Base.full{T}(A::AbstractLinearMap{T})
    M,N=size(A)
    mat=zeros(T,(M,N))
    for i=1:N
        v=zeros(T,N)
        v[i]=one(T)
        w=pointer_to_array(pointer(mat,(i-1)*M+1),M)
        A_mul_B!(w,A,v)
    end
    return mat
end

include("transpose.jl") # transposing linear maps
include("linearcombination.jl") # defining linear combinations of linear maps
include("composition.jl") # composition of linear maps
include("wrappedmap.jl") # wrap a matrix of linear map in a new type, thereby allowing to alter its properties
if VERSION.minor >= 3
    include("identitymap.jl") # the identity map, to be able to make linear combinations of AbstractLinearMap objects and I
end
include("functionmap.jl") # using a function as linear map

LinearMap{T}(A::Union(AbstractMatrix{T},AbstractLinearMap{T});isreal::Bool=Base.isreal(A),issym::Bool=Base.issym(A),ishermitian::Bool=Base.ishermitian(A),isposdef::Bool=Base.isposdef(A)) = 
    WrappedMap(A;isreal=isreal,issym=issym,ishermitian=ishermitian,isposdef=isposdef)
LinearMap(f::Function,M::Int,N::Int=M;ismutating::Bool=false,isreal::Bool=true,issym::Bool=false,ishermitian::Bool=(isreal && issym),isposdef::Bool=false,ftranspose::OptionalFunction=nothing,fctranspose::OptionalFunction=nothing) = 
    FunctionMap(f,M,N;ismutating=ismutating,isreal=isreal,issym=issym,ishermitian=ishermitian,isposdef=isposdef,ftranspose=ftranspose,fctranspose=fctranspose)
LinearMap(f::Function,eltype::Type,M::Int,N::Int=M;ismutating::Bool=false,issym::Bool=false,ishermitian::Bool=(T<:Real && issym),isposdef::Bool=false,ftranspose::OptionalFunction=nothing,fctranspose::OptionalFunction=nothing) = 
    FunctionMap{eltype}(f,M,N;ismutating=ismutating,issym=issym,ishermitian=ishermitian,isposdef=isposdef,ftranspose=ftranspose,fctranspose=fctranspose)


end # module
