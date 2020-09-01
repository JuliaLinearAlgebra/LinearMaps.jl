# helper function
check_dim_mul(A, B) = size(A, 2) == size(B, 1)

struct CompositeMap{T, As<:Tuple{Vararg{LinearMap}}} <: LinearMap{T}
    maps::As # stored in order of application to vector
    function CompositeMap{T, As}(maps::As) where {T, As}
        N = length(maps)
        for n in 2:N
            check_dim_mul(maps[n], maps[n-1]) || throw(DimensionMismatch("CompositeMap"))
        end
        for n in 1:N
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
for (f, _f, g) in ((:issymmetric, :_issymmetric, :transpose),
                    (:ishermitian, :_ishermitian, :adjoint),
                    (:isposdef, :_isposdef, :adjoint))
    @eval begin
        LinearAlgebra.$f(A::CompositeMap) = $_f(A.maps)
        $_f(maps::Tuple{}) = true
        $_f(maps::Tuple{<:LinearMap}) = $f(maps[1])
        $_f(maps::Tuple{Vararg{<:LinearMap}}) = maps[end] == $g(maps[1]) && $_f(Base.front(Base.tail(maps)))
        # since the introduction of ScaledMap, the following cases cannot occur
        # function $_f(maps::Tuple{Vararg{<:LinearMap}}) # length(maps) >= 2
            # if maps[1] isa UniformScalingMap{<:RealOrComplex}
            #     return $f(maps[1]) && $_f(Base.tail(maps))
            # elseif maps[end] isa UniformScalingMap{<:RealOrComplex}
            #     return $f(maps[end]) && $_f(Base.front(maps))
            # else
                # return maps[end] == $g(maps[1]) && $_f(Base.front(Base.tail(maps)))
            # end
        # end
    end
end

# scalar multiplication and division (non-commutative case)
function Base.:(*)(α::Number, A::LinearMap)
    T = promote_type(typeof(α), eltype(A))
    return CompositeMap{T}(tuple(A, UniformScalingMap(α, size(A, 1))))
end
function Base.:(*)(α::Number, A::CompositeMap)
    T = promote_type(typeof(α), eltype(A))
    Alast = last(A.maps)
    if Alast isa UniformScalingMap
        return CompositeMap{T}(tuple(Base.front(A.maps)..., α * Alast))
    else
        return CompositeMap{T}(tuple(A.maps..., UniformScalingMap(α, size(A, 1))))
    end
end
# needed for disambiguation
function Base.:(*)(α::RealOrComplex, A::CompositeMap)
    T = Base.promote_op(*, typeof(α), eltype(A))
    return ScaledMap{T}(α, A)
end
function Base.:(*)(A::LinearMap, α::Number)
    T = promote_type(typeof(α), eltype(A))
    return CompositeMap{T}(tuple(UniformScalingMap(α, size(A, 2)), A))
end
function Base.:(*)(A::CompositeMap, α::Number)
    T = promote_type(typeof(α), eltype(A))
    Afirst = first(A.maps)
    if Afirst isa UniformScalingMap
        return CompositeMap{T}(tuple(Afirst * α, Base.tail(A.maps)...))
    else
        return CompositeMap{T}(tuple(UniformScalingMap(α, size(A, 2)), A.maps...))
    end
end
# needed for disambiguation
function Base.:(*)(A::CompositeMap, α::RealOrComplex)
    T = Base.promote_op(*, typeof(α), eltype(A))
    return ScaledMap{T}(α, A)
end

Base.:(\)(α::Number, A::LinearMap) = inv(α) * A
Base.:(/)(A::LinearMap, α::Number) = A * inv(α)

# composition of linear maps
"""
    *(A::LinearMap, B::LinearMap)

Construct a `CompositeMap <: LinearMap`, a (lazy) representation of the product
of the two operators. Products of `LinearMap`/`CompositeMap` objects and
`LinearMap`/`CompositeMap` objects are reduced to a single `CompositeMap`.
In products of `LinearMap`s and `AbstractMatrix`/`UniformScaling` objects, the
latter get promoted to `LinearMap`s automatically.

# Examples
```jldoctest; setup=(using LinearAlgebra, LinearMaps)
julia> CS = LinearMap{Int}(cumsum, 3)::LinearMaps.FunctionMap;

julia> LinearMap(ones(Int, 3, 3)) * CS * I * rand(3, 3);
```
"""
function Base.:(*)(A₁::LinearMap, A₂::LinearMap)
    check_dim_mul(A₁, A₂) || throw(DimensionMismatch("*"))
    T = promote_type(eltype(A₁), eltype(A₂))
    return CompositeMap{T}(tuple(A₂, A₁))
end
function Base.:(*)(A₁::LinearMap, A₂::CompositeMap)
    check_dim_mul(A₁, A₂) || throw(DimensionMismatch("*"))
    T = promote_type(eltype(A₁), eltype(A₂))
    return CompositeMap{T}(tuple(A₂.maps..., A₁))
end
function Base.:(*)(A₁::CompositeMap, A₂::LinearMap)
    check_dim_mul(A₁, A₂) || throw(DimensionMismatch("*"))
    T = promote_type(eltype(A₁), eltype(A₂))
    return CompositeMap{T}(tuple(A₂, A₁.maps...))
end
function Base.:(*)(A₁::CompositeMap, A₂::CompositeMap)
    check_dim_mul(A₁, A₂) || throw(DimensionMismatch("*"))
    T = promote_type(eltype(A₁), eltype(A₂))
    return CompositeMap{T}(tuple(A₂.maps..., A₁.maps...))
end

# special transposition behavior
LinearAlgebra.transpose(A::CompositeMap{T}) where {T} = CompositeMap{T}(map(transpose, reverse(A.maps)))
LinearAlgebra.adjoint(A::CompositeMap{T}) where {T}   = CompositeMap{T}(map(adjoint, reverse(A.maps)))

# comparison of CompositeMap objects
Base.:(==)(A::CompositeMap, B::CompositeMap) = (eltype(A) == eltype(B) && A.maps == B.maps)

# multiplication with vectors
function mul!(y::AbstractVecOrMat, A::CompositeMap{<:Any,<:Tuple{LinearMap}}, x::AbstractVector)
    check_dim_mul(y, A, x)
    return mul!(y, A.maps[1], x)
end
function mul!(y::AbstractVecOrMat, A::CompositeMap{<:Any,<:Tuple{LinearMap,LinearMap}}, x::AbstractVector)
    check_dim_mul(y, A, x)
    _compositemul!(y, A, x, similar(y, size(A.maps[1], 1)))
end
function mul!(y::AbstractVecOrMat, A::CompositeMap{<:Any,<:Tuple{Vararg{LinearMap}}}, x::AbstractVector)
    check_dim_mul(y, A, x)
    _compositemul!(y, A, x, similar(y, size(A.maps[1], 1)), similar(y, size(A.maps[2], 1)))
end

function _compositemul!(y::AbstractVecOrMat, A::CompositeMap{T,<:Tuple{LinearMap,LinearMap}}, x::AbstractVector, z::AbstractVector) where {T}
    # no size checking, will be done by individual maps
    mul!(z, A.maps[1], x)
    mul!(y, A.maps[2], z)
    return y
end
function _compositemul!(y::AbstractVecOrMat, A::CompositeMap{T,<:Tuple{Vararg{LinearMap}}}, x::AbstractVector, source::AbstractVector, dest::AbstractVector) where {T}
    # no size checking, will be done by individual maps
    N = length(A.maps)
    mul!(source, A.maps[1], x)
    for n in 2:N-1
        try
            resize!(dest, size(A.maps[n], 1))
        catch err
            if err == ErrorException("cannot resize array with shared data")
                dest = similar(y, size(A.maps[n], 1))
            else
                rethrow(err)
            end
        end
        mul!(dest, A.maps[n], source)
        dest, source = source, dest # alternate dest and source
    end
    mul!(y, A.maps[N], source)
    return y
end
