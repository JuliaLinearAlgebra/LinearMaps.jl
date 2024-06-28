struct CompositeMap{T, As<:LinearMapTupleOrVector} <: LinearMap{T}
    maps::As # stored in order of application to vector
    function CompositeMap{T, As}(maps::As) where {T, As}
        N = length(maps)
        for n in 2:N
            check_dim_mul(maps[n], maps[n-1])
        end
        for TA in Base.Iterators.map(eltype, maps)
            promote_type(T, TA) == T ||
                error("eltype $TA cannot be promoted to $T in CompositeMap constructor")
        end
        new{T, As}(maps)
    end
end
CompositeMap{T}(maps::As) where {T, As<:LinearMapTupleOrVector} = CompositeMap{T, As}(maps)

Base.mapreduce(::typeof(identity), ::typeof(Base.mul_prod), maps::LinearMapTupleOrVector) =
    CompositeMap{promote_type(map(eltype, maps)...)}(_reverse!(maps))
Base.mapreduce(::typeof(identity), ::typeof(Base.mul_prod), maps::AbstractVector{<:LinearMap{T}}) where {T} =
    CompositeMap{T}(reverse!(maps))

MulStyle(A::CompositeMap) = MulStyle(A.maps...) === TwoArg() ? TwoArg() : ThreeArg()

# basic methods
Base.size(A::CompositeMap) = (size(last(A.maps), 1), size(first(A.maps), 2))
Base.axes(A::CompositeMap) = (axes(last(A.maps))[1], axes(first(A.maps))[2])
Base.isreal(A::CompositeMap) = all(isreal, A.maps) # sufficient but not necessary

# the following rules are sufficient but not necessary
for (f, _f, g) in ((:issymmetric, :_issymmetric, :transpose),
                    (:ishermitian, :_ishermitian, :adjoint))
    @eval begin
        LinearAlgebra.$f(A::CompositeMap) = $_f(A.maps)
        $_f(maps::Tuple{}) = true
        $_f(maps::Tuple{<:LinearMap}) = $f(first(maps))
        $_f(maps::LinearMapTuple) =
            maps[end] == $g(first(maps)) && $_f(Base.front(Base.tail(maps)))
        function $_f(maps::LinearMapVector)
            n = length(maps)
            if n == 0
                return true
            elseif n == 1
                return ($f(first(maps)))::Bool
            else
                return ((last(maps) == $g(first(maps)))::Bool && $_f(@views maps[begin+1:end-1]))
            end
        end
        # since the introduction of ScaledMap, the following cases cannot occur
        # function $_f(maps::Tuple{Vararg{LinearMap}}) # length(maps) >= 2
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

# A * B * A and A * B * A' are positive definite if (sufficient condition) A & B are positive definite
LinearAlgebra.isposdef(A::CompositeMap) = _isposdef(A.maps)
_isposdef(maps::Tuple{}) = true # empty product is equivalent to "I" which is pos. def.
_isposdef(maps::Tuple{<:LinearMap}) = isposdef(maps[1])
function _isposdef(maps::LinearMapTuple)
    (maps[end] == adjoint(maps[1]) || maps[end] == maps[1]) &&
        isposdef(maps[1]) && _isposdef(Base.front(Base.tail(maps)))
end
function _isposdef(maps::LinearMapVector)
    n = length(maps)
    if n == 0
        return true
    elseif n == 1
        return isposdef(first(maps))
    else
        return (last(maps) == adjoint(first(maps)) || last(maps) == first(maps)) &&
            isposdef(first(maps)) && _isposdef(maps[begin+1:end-1])
    end
end

# scalar multiplication and division (non-commutative case)
function Base.:(*)(α::Number, A::LinearMap)
    T = promote_type(typeof(α), eltype(A))
    return CompositeMap{T}(_combine(A, UniformScalingMap(α, size(A, 1))))
end
function Base.:(*)(α::Number, A::CompositeMap)
    T = promote_type(typeof(α), eltype(A))
    Alast = last(A.maps)
    if Alast isa UniformScalingMap
        return CompositeMap{T}(_combine(_front(A.maps), α * Alast))
    else
        return CompositeMap{T}(_combine(A.maps, UniformScalingMap(α, size(A, 1))))
    end
end
# needed for disambiguation
function Base.:(*)(α::RealOrComplex, A::CompositeMap{<:RealOrComplex})
    T = Base.promote_op(*, typeof(α), eltype(A))
    return ScaledMap{T}(α, A)
end
function Base.:(*)(A::LinearMap, α::Number)
    T = promote_type(typeof(α), eltype(A))
    return CompositeMap{T}(_combine(UniformScalingMap(α, size(A, 2)), A))
end
function Base.:(*)(A::CompositeMap, α::Number)
    T = promote_type(typeof(α), eltype(A))
    Afirst = first(A.maps)
    if Afirst isa UniformScalingMap
        return CompositeMap{T}(_combine(Afirst * α, _tail(A.maps)))
    else
        return CompositeMap{T}(_combine(UniformScalingMap(α, size(A, 2)), A.maps))
    end
end
# needed for disambiguation
function Base.:(*)(A::CompositeMap{<:RealOrComplex}, α::RealOrComplex)
    T = Base.promote_op(*, typeof(α), eltype(A))
    return ScaledMap{T}(α, A)
end

Base.:(\)(α::Number, A::LinearMap) = inv(α) * A
Base.:(/)(A::LinearMap, α::Number) = A * inv(α)

# composition of linear maps
"""
    *(A::LinearMap, B::LinearMap)::CompositeMap

Construct a (lazy) representation of the product of the two operators.
Products of `LinearMap`/`CompositeMap` objects and `LinearMap`/`CompositeMap`
objects are reduced to a single `CompositeMap`. In products of `LinearMap`s and
`AbstractMatrix`/`UniformScaling` objects, the latter get promoted to `LinearMap`s
automatically.

# Examples
```jldoctest; setup=(using LinearAlgebra, LinearMaps)
julia> CS = LinearMap{Int}(cumsum, 3)::LinearMaps.FunctionMap;

julia> LinearMap(ones(Int, 3, 3)) * CS * I * rand(3, 3);
```
"""
function Base.:(*)(A₁::LinearMap, A₂::LinearMap)
    T = promote_type(eltype(A₁), eltype(A₂))
    return CompositeMap{T}(_combine(A₂, A₁))
end
function Base.:(*)(A₁::LinearMap, A₂::CompositeMap)
    T = promote_type(eltype(A₁), eltype(A₂))
    return CompositeMap{T}(_combine(A₂.maps, A₁))
end
function Base.:(*)(A₁::CompositeMap, A₂::LinearMap)
    T = promote_type(eltype(A₁), eltype(A₂))
    return CompositeMap{T}(_combine(A₂, A₁.maps))
end
function Base.:(*)(A₁::CompositeMap, A₂::CompositeMap)
    T = promote_type(eltype(A₁), eltype(A₂))
    return CompositeMap{T}(_combine(A₂.maps, A₁.maps))
end
# needed for disambiguation
Base.:(*)(A₁::ScaledMap, A₂::CompositeMap) = A₁.λ * (A₁.lmap * A₂)
Base.:(*)(A₁::CompositeMap, A₂::ScaledMap) = (A₁ * A₂.lmap) * A₂.λ

# special transposition behavior
LinearAlgebra.transpose(A::CompositeMap{T}) where {T} =
    CompositeMap{T}(map(transpose, _reverse!(A.maps)))
LinearAlgebra.adjoint(A::CompositeMap{T}) where {T} =
    CompositeMap{T}(map(adjoint, _reverse!(A.maps)))

# comparison of CompositeMap objects
Base.:(==)(A::CompositeMap, B::CompositeMap) =
    (eltype(A) == eltype(B) && all(A.maps .== B.maps))

# multiplication with vectors/matrices
function Base.:(*)(A::CompositeMap, x::AbstractVector)
    MulStyle(A) === TwoArg() ?
        foldr(*, _reverse!(A.maps), init=x) :
        invoke(*, Tuple{LinearMap, AbstractVector}, A, x)
end

function _unsafe_mul!(y, A::CompositeMap, x::AbstractVector)
    MulStyle(A) === TwoArg() ?
        copyto!(y, A*x) :
        _compositemul!(y, A, x)
    return y
end
_unsafe_mul!(y, A::CompositeMap, x::AbstractMatrix) = _compositemul!(y, A, x)
_unsafe_mul!(y, A::CompositeMap{<:Any,<:Tuple{LinearMap}}, x::AbstractVector) =
    _unsafe_mul!(y, A.maps[1], x)
_unsafe_mul!(y, A::CompositeMap{<:Any,<:Tuple{LinearMap}}, X::AbstractMatrix) =
    _unsafe_mul!(y, A.maps[1], X)

function _compositemul!(y, A::CompositeMap{<:Any,<:Tuple{LinearMap,LinearMap}}, x,
                        source = nothing,
                        dest = nothing)
    if isnothing(source)
        z = convert(AbstractArray, A.maps[1] * x)
        _unsafe_mul!(y, A.maps[2], z)
        return y
    else
        _unsafe_mul!(source, A.maps[1], x)
        _unsafe_mul!(y, A.maps[2], source)
        return y
    end
end
_compositemul!(y, A::CompositeMap{<:Any,<:LinearMapTuple}, x, s = nothing, d = nothing) =
    _compositemulN!(y, A, x, s, d)
function _compositemul!(y, A::CompositeMap{<:Any,<:LinearMapVector}, x,
                        source = nothing,
                        dest = nothing)
    N = length(A.maps)
    if N == 1
        return _unsafe_mul!(y, A.maps[begin], x)
    elseif N == 2
        return _unsafe_mul!(y, A.maps[end] * A.maps[begin], x)
    else
        return _compositemulN!(y, A, x, source, dest)
    end
end

function _compositemulN!(y, A::CompositeMap, x,
                         src = nothing,
                         dst = nothing)
    N = length(A.maps) # ≥ 3
    n = n0 = firstindex(A.maps)
    source = isnothing(src) ?
        convert(AbstractArray, A.maps[n] * x) :
        _unsafe_mul!(src, A.maps[n], x)
    n += 1
    dest = isnothing(dst) ?
        convert(AbstractArray, A.maps[n] * source) :
        _unsafe_mul!(dst, A.maps[n], source)
    dest, source = source, dest # alternate dest and source
    for n in (n0+2):N-1
        dest = _resize(dest, (size(A.maps[n], 1), size(x)[2:end]...))
        _unsafe_mul!(dest, A.maps[n], source)
        dest, source = source, dest # alternate dest and source
    end
    _unsafe_mul!(y, last(A.maps), source)
    return y
end

function _resize(dest::AbstractVector, sz::Tuple{<:Integer})
    try
        resize!(dest, sz[1])
    catch err
        dest = similar(dest, sz)
    end
    dest
end
function _resize(dest::AbstractMatrix, sz::Tuple{<:Integer,<:Integer})
    size(dest) == sz && return dest
    similar(dest, sz)
end
