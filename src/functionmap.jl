immutable FunctionMap{T,F1,F2,F3}<:AbstractLinearMap{T}
    f::F1
    M::Int
    N::Int
    fT::F2
    fC::F3
    _ismutating::Bool
    _issymmetric::Bool
    _ishermitian::Bool
    _isposdef::Bool
end

# additional constructor
FunctionMap{T}(f, M::Int, ::Type{T} = Float64; kwargs...) = FunctionMap(f, M, M, T; kwargs...)
function FunctionMap{T,F}(f::F, M::Int, N::Int, ::Type{T} = Float64; ismutating::Bool=false, isreal::Bool=true, issymmetric::Bool=false, ishermitian::Bool=(isreal && issymmetric), isposdef::Bool=false, ftranspose::OptionalFunction=nothing, fctranspose::OptionalFunction=nothing)
    T=(isreal ? Float64 : Complex128) # default assumption
    FunctionMap{T}(f,M,N;ismutating=ismutating,issymmetric=issymmetric,ishermitian=ishermitian,isposdef=isposdef,ftranspose=ftranspose,fctranspose=fctranspose)
end

# show
function Base.show{T}(io::IO,A::FunctionMap{T})
    print(io,"FunctionMap{$T}($(A.f),$(A.M),$(A.N);ismutating=$(A._ismutating),issymmetric=$(A._issymmetric),ishermitian=$(A._ishermitian),isposdef=$(A._isposdef),transpose=$(A._fT),ctranspose=$(A._fC))")
end

# properties
Base.size(A::FunctionMap,n) = (n==1 ? A.M : (n==2 ? A.N : error("AbstractLinearMap objects have only 2 dimensions")))
Base.size(A::FunctionMap) = (A.M,A.N)
Base.issymmetric(A::FunctionMap) = A._issymmetric
Base.ishermitian(A::FunctionMap) = A._ishermitian
Base.isposdef(A::FunctionMap) = A._isposdef

# multiplication with vector
Base.A_mul_B!(y::AbstractVector, A::FunctionMap, x::AbstractVector) = begin
    (length(x) == A.N && length(y) == A.M) || throw(DimensionMismatch())
    A._ismutating ? A.f(y,x) : copy!(y,A.f(x))
    return y
end
*(A::FunctionMap, x::AbstractVector) = begin
    length(x) == A.N || throw(DimensionMismatch())
    A._ismutating ? A.f(similar(x,promote_type(eltype(A),eltype(x)),A.M),x) : A.f(x)
end

Base.At_mul_B!(y::AbstractVector, A::FunctionMap, x::AbstractVector) = begin
    A._issymmetric && return Base.A_mul_B!(y,A,x)
    (length(x) == A.M && length(y) == A.N) || throw(DimensionMismatch())
    if A._fT! = nothing
        return (A._ismutating ? A._fT(y,x) : copy!(y,A._fT(x)))
    elseif A._fC! = nothing
        return (A._ismutating ? conj!(A._fC(y,conj(x))) : copy!(y,conj(A._fC(conj(x)))))
    else
        error("transpose not implemented for $A")
    end
end
Base.At_mul_B(A::FunctionMap, x::AbstractVector) = begin
    A._issymmetric && return A*x
    length(x) == A.M || throw(DimensionMismatch())
    if A._fT! = nothing
        return (A._ismutating ? A._fT(similar(x,promote_type(eltype(A),eltype(x)),A.N),x) : A._fT(x))
    elseif A._fC! = nothing
        return (A._ismutating ? conj!(A._fC(similar(x,promote_type(eltype(A),eltype(x)),A.N),conj(x))) : conj(A._fC(conj(x))))
    else
        error("transpose not implemented for $A")
    end
end
Base.Ac_mul_B!(y::AbstractVector, A::FunctionMap, x::AbstractVector) = begin
    A._ishermitian && return Base.A_mul_B!(y,A,x)
    (length(x) == A.M && length(y) == A.N) || throw(DimensionMismatch())
    if A._fC! = nothing
        return (A._ismutating ? A._fC(y,x) : copy!(y,A._fC(x)))
    elseif A._fT! = nothing
        return (A._ismutating ? conj!(A._fT(y,conj(x))) : copy!(y,conj(A._fT(conj(x)))))
    else
        error("ctranspose not implemented for $A")
    end
end
Base.Ac_mul_B(A::FunctionMap, x::AbstractVector) = begin
    A._ishermitian && return A*x
    length(x) == A.M || throw(DimensionMismatch())
    if A._fC! = nothing
        return (A._ismutating ? A._fC(similar(x,promote_type(eltype(A),eltype(x)),A.N),x) : A._fC(x))
    elseif A._fT! = nothing
        return (A._ismutating ? conj!(A._fT(similar(x,promote_type(eltype(A),eltype(x)),A.N),conj(x))) : conj(A._fT(conj(x))))
    else
        error("ctranspose not implemented for $A")
    end
end
