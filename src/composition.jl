struct CompositeMap{T, As<:Tuple{Vararg{LinearMap}}} <: LinearMap{T}
    maps::As # stored in order of application to vector
    function CompositeMap{T, As}(maps::As) where {T, As}
        N = length(maps)
        for n in 2:N
            size(maps[n], 2) == size(maps[n-1], 1) || throw(DimensionMismatch("CompositeMap"))
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
const RealOrComplex = Union{Real,Complex}
for (f, _f, g) in ((:issymmetric, :_issymmetric, :transpose),
                    (:ishermitian, :_ishermitian, :adjoint),
                    (:isposdef, :_isposdef, :adjoint))
    @eval begin
        LinearAlgebra.$f(A::CompositeMap) = $_f(A.maps)
        $_f(maps::Tuple{}) = true
        $_f(maps::Tuple{<:LinearMap}) = $f(maps[1])
        function $_f(maps::Tuple{Vararg{<:LinearMap}}) # length(maps) >= 2
            if maps[1] isa UniformScalingMap{<:RealOrComplex}
                return $f(maps[1]) && $_f(Base.tail(maps))
            elseif maps[end] isa UniformScalingMap{<:RealOrComplex}
                return $f(maps[end]) && $_f(Base.front(maps))
            else
                return maps[end] == $g(maps[1]) && $_f(Base.front(Base.tail(maps)))
            end
        end
    end
end

# scalar multiplication and division
function Base.:(*)(α::Number, A::LinearMap)
    T = promote_type(typeof(α), eltype(A))
    return CompositeMap{T}(tuple(A, UniformScalingMap(α, size(A, 1))))
end
function Base.:(*)(α::Number, A::CompositeMap)
    T = promote_type(typeof(α), eltype(A))
    Alast = last(A.maps)
    if Alast isa UniformScalingMap
        return CompositeMap{T}(tuple(Base.front(A.maps)..., UniformScalingMap(α * Alast.λ, size(Alast, 1))))
    else
        return CompositeMap{T}(tuple(A.maps..., UniformScalingMap(α, size(A, 1))))
    end
end
function Base.:(*)(A::LinearMap, α::Number)
    T = promote_type(typeof(α), eltype(A))
    return CompositeMap{T}(tuple(UniformScalingMap(α, size(A, 2)), A))
end
function Base.:(*)(A::CompositeMap, α::Number)
    T = promote_type(typeof(α), eltype(A))
    Afirst = first(A.maps)
    if Afirst isa UniformScalingMap
        return CompositeMap{T}(tuple(UniformScalingMap(Afirst.λ * α, size(Afirst, 1)), Base.tail(A.maps)...))
    else
        return CompositeMap{T}(tuple(UniformScalingMap(α, size(A, 2)), A.maps...))
    end
end
Base.:(\)(α::Number, A::LinearMap) = inv(α) * A
Base.:(/)(A::LinearMap, α::Number) = A * inv(α)
Base.:(-)(A::LinearMap) = -1 * A

# composition of linear maps
"""
    A::LinearMap * B::LinearMap

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
    size(A₁, 2) == size(A₂, 1) || throw(DimensionMismatch("*"))
    T = promote_type(eltype(A₁), eltype(A₂))
    return CompositeMap{T}(tuple(A₂, A₁))
end
function Base.:(*)(A₁::LinearMap, A₂::CompositeMap)
    size(A₁, 2) == size(A₂, 1) || throw(DimensionMismatch("*"))
    T = promote_type(eltype(A₁), eltype(A₂))
    return CompositeMap{T}(tuple(A₂.maps..., A₁))
end
function Base.:(*)(A₁::CompositeMap, A₂::LinearMap)
    size(A₁, 2) == size(A₂, 1) || throw(DimensionMismatch("*"))
    T = promote_type(eltype(A₁), eltype(A₂))
    return CompositeMap{T}(tuple(A₂, A₁.maps...))
end
function Base.:(*)(A₁::CompositeMap, A₂::CompositeMap)
    size(A₁, 2) == size(A₂, 1) || throw(DimensionMismatch("*"))
    T = promote_type(eltype(A₁), eltype(A₂))
    return CompositeMap{T}(tuple(A₂.maps..., A₁.maps...))
end
Base.:(*)(A₁::LinearMap, A₂::UniformScaling) = A₁ * A₂.λ
Base.:(*)(A₁::UniformScaling, A₂::LinearMap) = A₁.λ * A₂

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
        for n in 2:N-1
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

At_mul_B!(y::AbstractVector, A::CompositeMap, x::AbstractVector) = A_mul_B!(y, transpose(A), x)

Ac_mul_B!(y::AbstractVector, A::CompositeMap, x::AbstractVector) = A_mul_B!(y, adjoint(A), x)
