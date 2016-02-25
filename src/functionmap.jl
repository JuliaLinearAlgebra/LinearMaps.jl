typealias OptionalFunction Union{Base.Callable,Void}

abstract AbstractFunctionMap{T} <: AbstractLinearMap{T}

function sanitycheck(isreal::Bool, issym::Bool, ishermitian::Bool, isposdef::Bool)
    (issym || ishermitian || isposdef) && (M!=N) && error("a symmetric or hermitian map should be square")
    isreal && (issym!=ishermitian) && error("a real symmetric map is also hermitian")
    isposdef && !ishermitian && error("a positive definite map should be hermitian")
end

immutable FunctionMap{T, F, Ft, Fc} <: AbstractFunctionMap{T}
    f::F
    M::Int
    N::Int
    _issym::Bool
    _ishermitian::Bool
    _isposdef::Bool
    _fT::Ft
    _fC::Fc
    function FunctionMap(f,M::Int,N::Int=M, ftranspose=nothing,fctranspose=nothing;issym::Bool=false,ishermitian::Bool=(T<:Real && issym),isposdef::Bool=false)
        sanitycheck(T<:Real, issym, ishermitian,isposdef)
        new(f, M, N, issym, ishermitian, isposdef, ftranspose, fctranspose)
    end
end

# additional constructor
function FunctionMap(f,M::Int,N::Int=M;isreal::Bool=true,issym::Bool=false,ishermitian::Bool=(isreal && issym),isposdef::Bool=false,ftranspose=nothing,fctranspose=nothing)
    T=(isreal ? Float64 : Complex128) # default assumption
    F = typeof(f)
    Ft = typeof(ftranspose)
    Fc = typeof(fctranspose)
    FunctionMap{T, F, Ft, Fc}(f,M,N, ftranspose, fctranspose;issym=issym,ishermitian=ishermitian,isposdef=isposdef)
end

immutable MutatingFunctionMap{T, F, Ft, Fc} <: AbstractFunctionMap{T}
    f::F
    M::Int
    N::Int
    _issym::Bool
    _ishermitian::Bool
    _isposdef::Bool
    _fT::Ft
    _fC::Fc
    function MutatingFunctionMap(f,M::Int,N::Int=M, ftranspose=nothing,fctranspose=nothing;issym::Bool=false,ishermitian::Bool=(T<:Real && issym),isposdef::Bool=false)
        checkproperties(T<:Real, issym, ishermitian,isposdef)
        new(f, M, N, issym, ishermitian, isposdef, ftranspose, fctranspose)
    end
end

function MutatingFunctionMap(f,M::Int,N::Int=M;isreal::Bool=true,issym::Bool=false,ishermitian::Bool=(isreal && issym),isposdef::Bool=false,ftranspose=nothing,fctranspose=nothing)
    T=(isreal ? Float64 : Complex128) # default assumption
    F = typeof(f)
    Ft = typeof(ftranspose)
    Fc = typeof(fctranspose)
    MutatingFunctionMap{T, F, Ft, Fc}(f,M,N, ftranspose, fctranspose;issym=issym,ishermitian=ishermitian,isposdef=isposdef)
end


# show
function Base.show{T}(io::IO,A::FunctionMap{T})
    print(io,"FunctionMap{$T}($(A.f),$(A.M),$(A.N);issym=$(A._issym),ishermitian=$(A._ishermitian),isposdef=$(A._isposdef),transpose=$(A._fT),ctranspose=$(A._fC))")
end

function Base.show{T}(io::IO,A::MutatingFunctionMap{T})
    print(io,"MutatingFunctionMap{$T}($(A.f),$(A.M),$(A.N);issym=$(A._issym),ishermitian=$(A._ishermitian),isposdef=$(A._isposdef),transpose=$(A._fT),ctranspose=$(A._fC))")
end


# properties
Base.size(A::AbstractFunctionMap,n)=(n==1 ? A.M : (n==2 ? A.N : error("AbstractLinearMap objects have only 2 dimensions")))
Base.size(A::AbstractFunctionMap)=(A.M,A.N)
Base.issym(A::AbstractFunctionMap)=A._issym
Base.ishermitian(A::AbstractFunctionMap)=A._ishermitian
Base.isposdef(A::AbstractFunctionMap)=A._isposdef

# multiplication with vector
Base.A_mul_B!(y::AbstractVector,A::FunctionMap,x::AbstractVector)=begin
    (length(x)==A.N && length(y)==A.M) || throw(DimensionMismatch())
    copy!(y,A.f(x))
    return y
end

Base.A_mul_B!(y::AbstractVector,A::MutatingFunctionMap,x::AbstractVector)=begin
    (length(x)==A.N && length(y)==A.M) || throw(DimensionMismatch())
    A.f(y,x)
    return y
end

*(A::FunctionMap,x::AbstractVector)=begin
    length(x)==A.N || throw(DimensionMismatch())
    A.f(x)
end

*(A::MutatingFunctionMap,x::AbstractVector)=begin
    length(x)==A.N || throw(DimensionMismatch())
    A.f(similar(x, promote_type(eltype(A),eltype(x)), A.M), x)
end


Base.At_mul_B!(y::AbstractVector,A::FunctionMap,x::AbstractVector)=begin
    A._issym && return Base.A_mul_B!(y,A,x)
    (length(x)==A.M && length(y)==A.N) || throw(DimensionMismatch())
    if A._fT!=nothing
        copy!(y, A._fT(x))
    elseif A._fC!=nothing
        copy!(y, conj(A._fC(conj(x))))
    else
        error("transpose not implemented for $A")
    end
end

Base.At_mul_B!(y::AbstractVector,A::MutatingFunctionMap,x::AbstractVector)=begin
    A._issym && return Base.A_mul_B!(y,A,x)
    (length(x) == A.M && length(y) == A.N) || throw(DimensionMismatch())
    if A._fT != nothing
        return A._fT(y,x)
    elseif A._fC != nothing
        return conj!(A._fC(y, conj(x)))
    else
        error("transpose not implemented for $A")
    end
end

Base.At_mul_B(A::FunctionMap,x::AbstractVector)=begin
    A._issym && return A*x
    length(x)==A.M || throw(DimensionMismatch())
    if A._fT!=nothing
        return A._fT(x)
    elseif A._fC!=nothing
        return conj(A._fC(conj(x)))
    else
        error("transpose not implemented for $A")
    end
end

Base.At_mul_B(A::MutatingFunctionMap, x::AbstractVector)=begin
    A._issym && return A*x
    length(x)==A.M || throw(DimensionMismatch())
    if A._fT!=nothing
        return A._fT(similar(x,promote_type(eltype(A),eltype(x)),A.N),x)
    elseif A._fC!=nothing
        return conj!(A._fC(similar(x,promote_type(eltype(A),eltype(x)),A.N),conj(x)))
    else
        error("transpose not implemented for $A")
    end
end

Base.Ac_mul_B!(y::AbstractVector,A::FunctionMap,x::AbstractVector)=begin
    A._ishermitian && return Base.A_mul_B!(y,A,x)
    (length(x)==A.M && length(y)==A.N) || throw(DimensionMismatch())
    if A._fC!=nothing
        return copy!(y,A._fC(x))
    elseif A._fT!=nothing
        return copy!(y,conj(A._fT(conj(x))))
    else
        error("ctranspose not implemented for $A")
    end
end

Base.Ac_mul_B!(y::AbstractVector,A::MutatingFunctionMap,x::AbstractVector)=begin
    A._ishermitian && return Base.A_mul_B!(y,A,x)
    (length(x)==A.M && length(y)==A.N) || throw(DimensionMismatch())
    if A._fC!=nothing
        return A._fC(y,x)
    elseif A._fT!=nothing
        return conj!(A._fT(y, conj(x)))
    else
        error("ctranspose not implemented for $A")
    end
end

Base.Ac_mul_B(A::FunctionMap,x::AbstractVector)=begin
    A._ishermitian && return A*x
    length(x)==A.M || throw(DimensionMismatch())
    if A._fC!=nothing
        return A._fC(x)
    elseif A._fT!=nothing
        return conj(A._fT(conj(x)))
    else
        error("ctranspose not implemented for $A")
    end
end

Base.Ac_mul_B(A::MutatingFunctionMap,x::AbstractVector)=begin
    A._ishermitian && return A*x
    length(x)==A.M || throw(DimensionMismatch())
    if A._fC!=nothing
        return A._fC(similar(x,promote_type(eltype(A),eltype(x)),A.N),x)
    elseif A._fT!=nothing
        return conj!(A._fT(similar(x,promote_type(eltype(A),eltype(x)),A.N),conj(x)))
    else
        error("ctranspose not implemented for $A")
    end
end
