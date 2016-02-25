typealias OptionalFunction Union{Base.Callable,Void}

abstract AbstractFunctionMap{T} <: AbstractLinearMap{T}

function sanitycheck(M::Int, N::Int, isreal::Bool, issym::Bool, ishermitian::Bool, isposdef::Bool)
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
        sanitycheck(M, N, T<:Real, issym, ishermitian,isposdef)
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
        sanitycheck(M, N, T<:Real, issym, ishermitian,isposdef)
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
    y
end

Base.A_mul_B!(y::AbstractVector,A::MutatingFunctionMap,x::AbstractVector)=begin
    (length(x)==A.N && length(y)==A.M) || throw(DimensionMismatch())
    A.f(y,x)
    y
end

*(A::FunctionMap,x::AbstractVector)=begin
    length(x)==A.N || throw(DimensionMismatch())
    A.f(x)
end

*(A::MutatingFunctionMap,x::AbstractVector)=begin
    length(x)==A.N || throw(DimensionMismatch())
    A.f(similar(x, promote_type(eltype(A),eltype(x)), A.M), x)
end

# At_mul_B!
function Base.At_mul_B!(y::AbstractVector, A::FunctionMap, x::AbstractVector)
    (length(x)==A.M && length(y)==A.N) || throw(DimensionMismatch())
    copy!(y, A._fT(x))
end

function Base.At_mul_B!{T,F,Fc}(y::AbstractVector, A::FunctionMap{T,F,Void,Fc}, x::AbstractVector)
    (length(x)==A.M && length(y)==A.N) || throw(DimensionMismatch())
    copy!(y, conj(A._fC(conj(x))))
end

function Base.At_mul_B!{T,F}(y::AbstractVector,A::FunctionMap{T,F,Void,Void},x::AbstractVector)
    A._issym && return Base.A_mul_B!(y,A,x)
    error("transpose not implemented for $A")
end

function Base.At_mul_B!(y::AbstractVector,A::MutatingFunctionMap,x::AbstractVector)
    (length(x) == A.M && length(y) == A.N) || throw(DimensionMismatch())
    A._fT(y,x)
end

function Base.At_mul_B!{T,F,Fc}(y::AbstractVector,A::MutatingFunctionMap{T,F,Void,Fc},x::AbstractVector)
    (length(x) == A.M && length(y) == A.N) || throw(DimensionMismatch())
    conj!(A._fC(y, conj(x)))
end

function Base.At_mul_B!{T,F}(y::AbstractVector,A::MutatingFunctionMap{T,F,Void,Void},x::AbstractVector)
    A._issym && return Base.A_mul_B!(y,A,x)
    error("transpose not implemented for $A")
end

# At_mul_B
function Base.At_mul_B(A::FunctionMap,x::AbstractVector)
    length(x)==A.M || throw(DimensionMismatch())
    A._fT(x)
end

function Base.At_mul_B{T,F,Fc}(A::FunctionMap{T,F,Void,Fc},x::AbstractVector)
    length(x)==A.M || throw(DimensionMismatch())
    conj(A._fC(conj(x)))
end

function Base.At_mul_B{T,F}(A::FunctionMap{T,F,Void,Void},x::AbstractVector)
    A._issym && return A*x
    error("transpose not implemented for $A")
end

function Base.At_mul_B(A::MutatingFunctionMap,x::AbstractVector)
    length(x)==A.M || throw(DimensionMismatch())
    A._fT(similar(x,promote_type(eltype(A),eltype(x)),A.N),x)
end

function Base.At_mul_B{T,F,Fc}(A::MutatingFunctionMap{T,F,Void,Fc},x::AbstractVector)
    length(x)==A.M || throw(DimensionMismatch())
    conj!(A._fC(similar(x,promote_type(eltype(A),eltype(x)),A.N),conj(x)))
end

function Base.At_mul_B{T,F}(A::MutatingFunctionMap{T,F,Void,Void},x::AbstractVector)
    A._issym && return A*x
    error("transpose not implemented for $A")
end

# Ac_mul_B!
function Base.Ac_mul_B!(y::AbstractVector,A::FunctionMap,x::AbstractVector)
    (length(x)==A.M && length(y)==A.N) || throw(DimensionMismatch())
    copy!(y,A._fC(x))
end

function Base.Ac_mul_B!{T,F,Ft}(y::AbstractVector,A::FunctionMap{T,F,Ft,Void},x::AbstractVector)
    (length(x)==A.M && length(y)==A.N) || throw(DimensionMismatch())
    copy!(y,conj(A._fT(conj(x))))
end

function Base.Ac_mul_B!{T,F}(y::AbstractVector,A::FunctionMap{T,F,Void,Void},x::AbstractVector)
    A._ishermitian && return Base.A_mul_B!(y,A,x)
    error("ctranspose not implemented for $A")
end

function Base.Ac_mul_B!(y::AbstractVector,A::MutatingFunctionMap,x::AbstractVector)
    (length(x)==A.M && length(y)==A.N) || throw(DimensionMismatch())
    A._fC(y,x)
end

function Base.Ac_mul_B!{T,F,Ft}(y::AbstractVector,A::MutatingFunctionMap{T,F,Ft,Void},x::AbstractVector)
    (length(x)==A.M && length(y)==A.N) || throw(DimensionMismatch())
    conj!(A._fT(y, conj(x)))
end

function Base.Ac_mul_B!{T,F}(y::AbstractVector,A::MutatingFunctionMap{T,F,Void,Void},x::AbstractVector)
    A._ishermitian && return Base.A_mul_B!(y,A,x)
    error("ctranspose not implemented for $A")
end

# Ac_mul_B
function Base.Ac_mul_B(A::FunctionMap,x::AbstractVector)
    length(x)==A.M || throw(DimensionMismatch())
    A._fC(x)
end

function Base.Ac_mul_B{T,F,Ft}(A::FunctionMap{T,F,Ft,Void},x::AbstractVector)
    length(x)==A.M || throw(DimensionMismatch())
    conj(A._fT(conj(x)))
end

function Base.Ac_mul_B{T,F}(A::FunctionMap{T,F,Void,Void},x::AbstractVector)
    A._ishermitian && return A*x
    error("ctranspose not implemented for $A")
end

function Base.Ac_mul_B(A::MutatingFunctionMap,x::AbstractVector)
    length(x)==A.M || throw(DimensionMismatch())
    A._fC(similar(x,promote_type(eltype(A),eltype(x)),A.N),x)
end

function Base.Ac_mul_B{T,F,Ft}(A::MutatingFunctionMap{T,F,Ft,Void},x::AbstractVector)
    length(x)==A.M || throw(DimensionMismatch())
    conj!(A._fT(similar(x,promote_type(eltype(A),eltype(x)),A.N),conj(x)))
end

function Base.Ac_mul_B{T,F}(A::MutatingFunctionMap{T,F,Void,Void},x::AbstractVector)
    A._ishermitian && return A*x
    error("ctranspose not implemented for $A")
end
