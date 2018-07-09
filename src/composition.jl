struct CompositeMap{T, As<:Tuple{Vararg{LinearMap}}} <: LinearMap{T}
    maps::As # stored in order of application to vector
    function CompositeMap{T, As}(maps::As) where {T, As}
        N = length(maps)
        for n = 2:N
            size(maps[n],2) == size(maps[n-1],1) || throw(DimensionMismatch("CompositeMap"))
        end
        for n = 1:N
            promote_type(T, eltype(maps[n])) == T || throw(InexactError())
        end
        new{T,As}(maps)
    end
end
CompositeMap{T}(maps::As) where {T, As<:Tuple{Vararg{LinearMap}}} = CompositeMap{T, As}(maps)

# basic methods
Base.size(A::CompositeMap) = (size(A.maps[end], 1), size(A.maps[1], 2))
Base.isreal(A::CompositeMap) = all(isreal, A.maps) # sufficient but not necessary

# the following rules are sufficient but not necessary
function LinearAlgebra.issymmetric(A::CompositeMap)
    N = length(A.maps)
    if isodd(N)
        issymmetric(A.maps[div(N+1,2)]) || return false
    end
    for n = 1:div(N,2)
        A.maps[n] == transpose(A.maps[N-n+1]) || return false
    end
    return true
end
function LinearAlgebra.ishermitian(A::CompositeMap)
    N = length(A.maps)
    if isodd(N)
        ishermitian(A.maps[div(N+1,2)]) || return false
    end
    for n = 1:div(N,2)
        A.maps[n] == adjoint(A.maps[N-n+1]) || return false
    end
    return true
end
function LinearAlgebra.isposdef(A::CompositeMap)
    N = length(A.maps)
    if isodd(N)
        isposdef(A.maps[div(N+1,2)]) || return false
    end
    for n = 1:div(N,2)
        A.maps[n] == adjoint(A.maps[N-n+1]) || return false
    end
    return true
end

# composition of linear maps
function Base.:(*)(A1::CompositeMap, A2::CompositeMap)
    size(A1,2) == size(A2,1) || throw(DimensionMismatch("*"))
    T = promote_type(eltype(A1),eltype(A2))
    return CompositeMap{T}(tuple(A2.maps..., A1.maps...))
end
function Base.:(*)(A1::LinearMap, A2::CompositeMap)
    size(A1,2) == size(A2,1) || throw(DimensionMismatch("*"))
    T = promote_type(eltype(A1),eltype(A2))
    return CompositeMap{T}(tuple(A2.maps..., A1))
end
function Base.:(*)(A1::CompositeMap, A2::LinearMap)
    size(A1,2) == size(A2,1) || throw(DimensionMismatch("*"))
    T = promote_type(eltype(A1),eltype(A2))
    return CompositeMap{T}(tuple(A2, A1.maps...))
end
function Base.:(*)(A1::LinearMap, A2::LinearMap)
    size(A1,2) == size(A2,1) || throw(DimensionMismatch("*"))
    T = promote_type(eltype(A1),eltype(A2))
    return CompositeMap{T}(tuple(A2, A1))
end

# special transposition behavior
LinearAlgebra.transpose(A::CompositeMap{T}) where {T} = CompositeMap{T}(map(transpose, reverse(A.maps)))
LinearAlgebra.adjoint(A::CompositeMap{T}) where {T}   = CompositeMap{T}(map(adjoint, reverse(A.maps)))

# multiplication with vectors
function A_mul_B!(y::AbstractVector, A::CompositeMap, x::AbstractVector)
    # no size checking, will be done by individual maps
    N = length(A.maps)
    if N==1
        A_mul_B!(y, A.maps[1], x)
    else
        T = eltype(y)
        dest = Array{T}(undef, size(A.maps[1], 1))
        A_mul_B!(dest, A.maps[1], x)
        source = dest
        if N>2
            dest = Array{T}(undef, size(A.maps[2],1))
        end
        for n=2:N-1
            resize!(dest, size(A.maps[n],1))
            A_mul_B!(dest, A.maps[n], source)
            dest, source = source, dest # alternate dest and source
        end
        A_mul_B!(y, A.maps[N], source)
    end
    return y
end
function At_mul_B!(y::AbstractVector, A::CompositeMap, x::AbstractVector)
    # no size checking, will be done by individual maps
    N = length(A.maps)
    if N == 1
        At_mul_B!(y, A.maps[1], x)
    else
        T = eltype(y)
        dest = Array{T}(size(A.maps[N], 2))
        At_mul_B!(dest, A.maps[N], x)
        source = dest
        if N>2
            dest = Array{T}(size(A.maps[N-1], 2))
        end
        for n = N-1:-1:2
            resize!(dest, size(A.maps[n], 2))
            At_mul_B!(dest, A.maps[n], source)
            dest, source = source, dest # alternate dest and source
        end
        At_mul_B!(y, A.maps[1], source)
    end
    return y
end
function Ac_mul_B!(y::AbstractVector, A::CompositeMap, x::AbstractVector)
    # no size checking, will be done by individual maps
    N = length(A.maps)
    if N == 1
        Ac_mul_B!(y, A.maps[1], x)
    else
        T = eltype(y)
        dest = Array{T}(size(A.maps[N], 2))
        Ac_mul_B!(dest, A.maps[N], x)
        source = dest
        if N>2
            dest = Array{T}(size(A.maps[N-1], 2))
        end
        for n = N-1:-1:2
            resize!(dest, size(A.maps[n], 2))
            Ac_mul_B!(dest, A.maps[n], source)
            dest, source = source, dest # alternate dest and source
        end
        Ac_mul_B!(y, A.maps[1], source)
    end
    return y
end
