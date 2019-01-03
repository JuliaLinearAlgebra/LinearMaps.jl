struct CompositeMap{T, As<:Tuple{Vararg{LinearMap}}} <: LinearMap{T}
    maps::As # stored in order of application to vector
    function CompositeMap{T, As}(maps::As) where {T, As}
        N = length(maps)
        for n = 2:N
            size(maps[n], 2) == size(maps[n-1], 1) || throw(DimensionMismatch("CompositeMap"))
        end
        for n = 1:N
            promote_type(T, eltype(maps[n])) == T || throw(InexactError())
        end
        new{T, As}(maps)
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
        issymmetric(A.maps[div(N+1, 2)]) || return false
    end
    for n = 1:div(N, 2)
        A.maps[n] == transpose(A.maps[N-n+1]) || return false
    end
    return true
end
function LinearAlgebra.ishermitian(A::CompositeMap)
    N = length(A.maps)
    if isodd(N)
        ishermitian(A.maps[div(N+1, 2)]) || return false
    end
    for n = 1:div(N, 2)
        A.maps[n] == adjoint(A.maps[N-n+1]) || return false
    end
    return true
end
function LinearAlgebra.isposdef(A::CompositeMap)
    N = length(A.maps)
    if isodd(N)
        isposdef(A.maps[div(N+1, 2)]) || return false
    end
    for n = 1:div(N, 2)
        A.maps[n] == adjoint(A.maps[N-n+1]) || return false
    end
    return true
end

# scalar multiplication and division
function Base.:(*)(α::Number, A::LinearMap)
    T = promote_type(eltype(α), eltype(A))
    return CompositeMap{T}(tuple(A, UniformScalingMap(α, size(A, 1))))
end
function Base.:(*)(α::Number, A::CompositeMap)
    T = promote_type(eltype(α), eltype(A))
    Alast = last(A.maps)
    if Alast isa UniformScalingMap
        return CompositeMap{T}(tuple(A.maps[1:end-1]..., UniformScalingMap(α * Alast.λ, size(Alast, 1))))
    else
        return CompositeMap{T}(tuple(A, UniformScalingMap(α, size(A, 1))))
    end
end
function Base.:(*)(A::LinearMap, α::Number)
    T = promote_type(eltype(α), eltype(A))
    return CompositeMap{T}(tuple(UniformScalingMap(α, size(A, 2)), A))
end
function Base.:(*)(A::CompositeMap, α::Number)
    T = promote_type(eltype(α), eltype(A))
    Afirst = first(A.maps)
    if Afirst isa UniformScalingMap
        return CompositeMap{T}(tuple(UniformScalingMap(Afirst.λ * α, size(Afirst, 1)), A.maps[2:end]...))
    else
        return CompositeMap{T}(tuple(UniformScalingMap(α, size(A, 2)), A))
    end
end
Base.:(\)(α::Number, A::LinearMap) = inv(α) * A
Base.:(/)(A::LinearMap, α::Number) = A * inv(α)
Base.:(-)(A::LinearMap) = -1 * A

# composition of linear maps
function Base.:(*)(A1::CompositeMap, A2::CompositeMap)
    size(A1, 2) == size(A2, 1) || throw(DimensionMismatch("*"))
    T = promote_type(eltype(A1), eltype(A2))
    return CompositeMap{T}(tuple(A2.maps..., A1.maps...))
end
function Base.:(*)(A1::LinearMap, A2::CompositeMap)
    size(A1, 2) == size(A2, 1) || throw(DimensionMismatch("*"))
    T = promote_type(eltype(A1), eltype(A2))
    return CompositeMap{T}(tuple(A2.maps..., A1))
end
function Base.:(*)(A1::CompositeMap, A2::LinearMap)
    size(A1, 2) == size(A2, 1) || throw(DimensionMismatch("*"))
    T = promote_type(eltype(A1), eltype(A2))
    return CompositeMap{T}(tuple(A2, A1.maps...))
end
function Base.:(*)(A1::LinearMap, A2::LinearMap)
    size(A1, 2) == size(A2, 1) || throw(DimensionMismatch("*"))
    T = promote_type(eltype(A1),eltype(A2))
    return CompositeMap{T}(tuple(A2, A1))
end
Base.:(*)(A1::LinearMap, A2::UniformScaling{T}) where {T} = A1 * A2[1,1]
Base.:(*)(A1::UniformScaling{T}, A2::LinearMap) where {T} = A1[1,1] * A2

# special transposition behavior
LinearAlgebra.transpose(A::CompositeMap{T}) where {T} = CompositeMap{T}(map(transpose, reverse(A.maps)))
LinearAlgebra.adjoint(A::CompositeMap{T}) where {T}   = CompositeMap{T}(map(adjoint, reverse(A.maps)))

# comparison of LinearCombination objects
Base.:(==)(A::CompositeMap, B::CompositeMap) = (eltype(A) == eltype(B) && A.maps == B.maps)

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
            dest = Array{T}(undef, size(A.maps[2], 1))
        end
        for n=2:N-1
            try
                resize!(dest, size(A.maps[n], 1))
            catch err
                if err == ErrorException("cannot resize array with shared data")
                    dest = Array{T}(undef, size(A.maps[n], 1))
                else
                    rethrow(err)
                end
            end
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
        dest = Array{T}(undef, size(A.maps[N], 2))
        At_mul_B!(dest, A.maps[N], x)
        source = dest
        if N>2
            dest = Array{T}(undef, size(A.maps[N-1], 2))
        end
        for n = N-1:-1:2
            try
                resize!(dest, size(A.maps[n], 2))
            catch err
                if err == ErrorException("cannot resize array with shared data")
                    dest = Array{T}(undef, size(A.maps[n], 2))
                else
                    rethrow(err)
                end
            end
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
        dest = Array{T}(undef, size(A.maps[N], 2))
        Ac_mul_B!(dest, A.maps[N], x)
        source = dest
        if N>2
            dest = Array{T}(undef, size(A.maps[N-1], 2))
        end
        for n = N-1:-1:2
            try
                resize!(dest, size(A.maps[n], 2))
            catch err
                if err == ErrorException("cannot resize array with shared data")
                    dest = Array{T}(undef, size(A.maps[n], 2))
                else
                    rethrow(err)
                end
            end
            Ac_mul_B!(dest, A.maps[n], source)
            dest, source = source, dest # alternate dest and source
        end
        Ac_mul_B!(y, A.maps[1], source)
    end
    return y
end
