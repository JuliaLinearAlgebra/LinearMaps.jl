type CompositeMap{T}<:AbstractLinearMap{T}
#    t::Type{T} # just stores T, trick to make CompositeMap{T} actually depend on T
    maps::Vector{AbstractLinearMap} # stored in order of application to vector
    function CompositeMap(maps::Vector{AbstractLinearMap})
        N=length(maps)
        for n=2:N
            size(maps[n],2)==size(maps[n-1],1) || throw(DimensionMismatch("CompositeMap"))
        end
        for n=1:N
            promote_type(T,eltype(maps[n]))==T || throw(InexactError())
        end
        new(maps)
    end
end

# basic methods
Base.size(A::CompositeMap,n)=(n==1 ? size(A.maps[end],1) : (n==2 ? size(A.maps[1],2) : error("AbstractLinearMap objects have only 2 dimensions")))
Base.size(A::CompositeMap)=(size(A,1),size(A,2))
Base.isreal(A::CompositeMap)=all(isreal,A.maps) # sufficient but not necessary
# the following rules are sufficient but not necessary
function Base.issym(A::CompositeMap)
    N=length(A.maps)
    if isodd(N)
        issym(A.maps[div(N+1,2)]) || return false
    end
    for n=1:div(N,2)
        A.maps[n]==transpose(A.maps[N-n+1]) || return false
    end
    return true
end
function Base.ishermitian(A::CompositeMap)
    N=length(A.maps)
    if isodd(N)
        ishermitian(A.maps[div(N+1,2)]) || return false
    end
    for n=1:div(N,2)
        A.maps[n]==ctranspose(A.maps[N-n+1]) || return false
    end
    return true
end
function Base.isposdef(A::CompositeMap)
    N=length(A.maps)
    if isodd(N)
        isposdef(A.maps[div(N+1,2)]) || return false
    end
    for n=1:div(N,2)
        A.maps[n]==ctranspose(A.maps[N-n+1]) || return false
    end
    return true
end

# composition of linear maps
function *(A1::CompositeMap,A2::CompositeMap)
    size(A1,2)==size(A2,1) || throw(DimensionMismatch("*"))
    T=promote_type(eltype(A1),eltype(A2))
    return CompositeMap{T}(AbstractLinearMap[A2.maps...,A1.maps...])
end
function *(A1::AbstractLinearMap,A2::CompositeMap)
    size(A1,2)==size(A2,1) || throw(DimensionMismatch("*"))
    T=promote_type(eltype(A1),eltype(A2))
    return CompositeMap{T}(AbstractLinearMap[A2.maps...,A1])
end
function *(A1::CompositeMap,A2::AbstractLinearMap)
    size(A1,2)==size(A2,1) || throw(DimensionMismatch("*"))
    T=promote_type(eltype(A1),eltype(A2))
    return CompositeMap{T}(AbstractLinearMap[A2,A1.maps...])
end
function *(A1::AbstractLinearMap,A2::AbstractLinearMap)
    size(A1,2)==size(A2,1) || throw(DimensionMismatch("*"))
    T=promote_type(eltype(A1),eltype(A2))
    return CompositeMap{T}(AbstractLinearMap[A2,A1])
end

# comparison of CompositeMap objects
==(A::CompositeMap,B::CompositeMap)=(eltype(A)==eltype(B) && A.maps==B.maps)

# special transposition behavior
Base.transpose(A::CompositeMap)=CompositeMap{eltype(A)}(AbstractLinearMap[transpose(M) for M in reverse(A.maps)])
Base.ctranspose(A::CompositeMap)=CompositeMap{eltype(A)}(AbstractLinearMap[ctranspose(M) for M in reverse(A.maps)])

# multiplication with vectors
function Base.A_mul_B!(y::AbstractVector,A::CompositeMap,x::AbstractVector)
    # no size checking, will be done by individual maps
    N=length(A.maps)
    if N==1
        Base.A_mul_B!(y,A.maps[1],x)
    else
        T=promote_type(eltype(A),eltype(x))
        dest=Array(T,size(A.maps[1],1))
        Base.A_mul_B!(dest,A.maps[1],x)
        source=dest
        if N>2
            dest=Array(T,size(A.maps[2],1))
        end
        for n=2:N-1
            resize!(dest,size(A.maps[n],1))
            Base.A_mul_B!(dest,A.maps[n],source)
            dest,source=source,dest # alternate dest and source
        end
        Base.A_mul_B!(y,A.maps[N],source)
    end
    return y
end
function Base.At_mul_B!(y::AbstractVector,A::CompositeMap,x::AbstractVector)
    # no size checking, will be done by individual maps
    N=length(A.maps)
    if N==1
        Base.At_mul_B!(y,A.maps[1],x)
    else
        T=promote_type(eltype(A),eltype(x))
        dest=Array(T,size(A.maps[N],2))
        Base.At_mul_B!(dest,A.maps[N],x)
        source=dest
        if N>2
            dest=Array(T,size(A.maps[N-1],2))
        end
        for n=N-1:-1:2
            resize!(dest,size(A.maps[n],2))
            Base.At_mul_B!(dest,A.maps[n],source)
            dest,source=source,dest # alternate dest and source
        end
        Base.At_mul_B!(y,A.maps[1],source)
    end
    return y
end
function Base.Ac_mul_B!(y::AbstractVector,A::CompositeMap,x::AbstractVector)
    # no size checking, will be done by individual maps
    N=length(A.maps)
    if N==1
        Base.Ac_mul_B!(y,A.maps[1],x)
    else
        T=promote_type(eltype(A),eltype(x))
        dest=Array(T,size(A.maps[N],2))
        Base.Ac_mul_B!(dest,A.maps[N],x)
        source=dest
        if N>2
            dest=Array(T,size(A.maps[N-1],2))
        end
        for n=N-1:-1:2
            resize!(dest,size(A.maps[n],2))
            Base.Ac_mul_B!(dest,A.maps[n],source)
            dest,source=source,dest # alternate dest and source
        end
        Base.Ac_mul_B!(y,A.maps[1],source)
    end
    return y
end
