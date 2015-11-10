typealias OptionalFunction Union{Base.Callable,Void}

immutable FunctionMap{T}<:AbstractLinearMap{T}
    f::Function
    M::Int
    N::Int
    _ismutating::Bool
    _issym::Bool
    _ishermitian::Bool
    _isposdef::Bool
    _fT::OptionalFunction
    _fC::OptionalFunction
    function FunctionMap(f::Function,M::Int,N::Int=M;ismutating::Bool=false,issym::Bool=false,ishermitian::Bool=(T<:Real && issym),isposdef::Bool=false,ftranspose::OptionalFunction=nothing,fctranspose::OptionalFunction=nothing)
        # sanity checks
        (issym || ishermitian || isposdef) && (M!=N) && error("a symmetric or hermitian map should be square")
        T<:Real && (issym!=ishermitian) && error("a real symmetric map is also hermitian")
        isposdef && !ishermitian && error("a positive definite map should be hermitian")
        # construct
        new(f,M,N,ismutating,issym,ishermitian,isposdef,ftranspose,fctranspose)
    end
end
# additional constructor
function FunctionMap(f::Function,M::Int,N::Int=M;ismutating::Bool=false,isreal::Bool=true,issym::Bool=false,ishermitian::Bool=(isreal && issym),isposdef::Bool=false,ftranspose::OptionalFunction=nothing,fctranspose::OptionalFunction=nothing)
    T=(isreal ? Float64 : Complex128) # default assumption
    FunctionMap{T}(f,M,N;ismutating=ismutating,issym=issym,ishermitian=ishermitian,isposdef=isposdef,ftranspose=ftranspose,fctranspose=fctranspose)
end

# show
function Base.show{T}(io::IO,A::FunctionMap{T})
    print(io,"FunctionMap{$T}($(A.f),$(A.M),$(A.N);ismutating=$(A._ismutating),issym=$(A._issym),ishermitian=$(A._ishermitian),isposdef=$(A._isposdef),transpose=$(A._fT),ctranspose=$(A._fC))")
end


# properties
Base.size(A::FunctionMap,n)=(n==1 ? A.M : (n==2 ? A.N : error("AbstractLinearMap objects have only 2 dimensions")))
Base.size(A::FunctionMap)=(A.M,A.N)
Base.issym(A::FunctionMap)=A._issym
Base.ishermitian(A::FunctionMap)=A._ishermitian
Base.isposdef(A::FunctionMap)=A._isposdef

# multiplication with vector
Base.A_mul_B!(y::AbstractVector,A::FunctionMap,x::AbstractVector)=begin
    (length(x)==A.N && length(y)==A.M) || throw(DimensionMismatch())
    A._ismutating ? A.f(y,x) : copy!(y,A.f(x))
    return y
end
*(A::FunctionMap,x::AbstractVector)=begin
    length(x)==A.N || throw(DimensionMismatch())
    A._ismutating ? A.f(similar(x,promote_type(eltype(A),eltype(x)),A.M),x) : A.f(x)
end

Base.At_mul_B!(y::AbstractVector,A::FunctionMap,x::AbstractVector)=begin
    A._issym && return Base.A_mul_B!(y,A,x)
    (length(x)==A.M && length(y)==A.N) || throw(DimensionMismatch())
    if A._fT!=nothing
        return (A._ismutating ? A._fT(y,x) : copy!(y,A._fT(x)))
    elseif A._fC!=nothing
        return (A._ismutating ? conj!(A._fC(y,conj(x))) : copy!(y,conj(A._fC(conj(x)))))
    else
        error("transpose not implemented for $A")
    end
end
Base.At_mul_B(A::FunctionMap,x::AbstractVector)=begin
    A._issym && return A*x
    length(x)==A.M || throw(DimensionMismatch())
    if A._fT!=nothing
        return (A._ismutating ? A._fT(similar(x,promote_type(eltype(A),eltype(x)),A.N),x) : A._fT(x))
    elseif A._fC!=nothing
        return (A._ismutating ? conj!(A._fC(similar(x,promote_type(eltype(A),eltype(x)),A.N),conj(x))) : conj(A._fC(conj(x))))
    else
        error("transpose not implemented for $A")
    end
end
Base.Ac_mul_B!(y::AbstractVector,A::FunctionMap,x::AbstractVector)=begin
    A._ishermitian && return Base.A_mul_B!(y,A,x)
    (length(x)==A.M && length(y)==A.N) || throw(DimensionMismatch())
    if A._fC!=nothing
        return (A._ismutating ? A._fC(y,x) : copy!(y,A._fC(x)))
    elseif A._fT!=nothing
        return (A._ismutating ? conj!(A._fT(y,conj(x))) : copy!(y,conj(A._fT(conj(x)))))
    else
        error("ctranspose not implemented for $A")
    end
end
Base.Ac_mul_B(A::FunctionMap,x::AbstractVector)=begin
    A._ishermitian && return A*x
    length(x)==A.M || throw(DimensionMismatch())
    if A._fC!=nothing
        return (A._ismutating ? A._fC(similar(x,promote_type(eltype(A),eltype(x)),A.N),x) : A._fC(x))
    elseif A._fT!=nothing
        return (A._ismutating ? conj!(A._fT(similar(x,promote_type(eltype(A),eltype(x)),A.N),conj(x))) : conj(A._fT(conj(x))))
    else
        error("ctranspose not implemented for $A")
    end
end
