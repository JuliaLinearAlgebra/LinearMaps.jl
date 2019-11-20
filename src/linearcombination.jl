struct LinearCombination{T, MS<:MulStyle, As<:Tuple{Vararg{LinearMap}}} <: LinearMap{T}
    maps::As
    function LinearCombination{T, MS, As}(maps::As) where {T, MS<:MulStyle, As}
        N = length(maps)
        sz = size(maps[1])
        for Ai in maps
            size(Ai) == sz || throw(DimensionMismatch("LinearCombination"))
            promote_type(T, eltype(Ai)) == T || throw(InexactError())
        end
        MS === FiveArg && mulstyle(maps...) === ThreeArg && throw("wrong mulstyle in constructor")
        new{T, MS, As}(maps)
    end
end

LinearCombination{T,MS}(maps::As) where {T, MS<:MulStyle, As} = LinearCombination{T, mulstyle(maps...), As}(maps)

mulstyle(::LinearCombination{T,MS}) where {T, MS<:MulStyle} = MS

# basic methods
Base.size(A::LinearCombination) = size(A.maps[1])
LinearAlgebra.issymmetric(A::LinearCombination) = all(issymmetric, A.maps) # sufficient but not necessary
LinearAlgebra.ishermitian(A::LinearCombination) = all(ishermitian, A.maps) # sufficient but not necessary
LinearAlgebra.isposdef(A::LinearCombination) = all(isposdef, A.maps) # sufficient but not necessary

# adding linear maps
"""
    A::LinearMap + B::LinearMap

Construct a `LinearCombination <: LinearMap`, a (lazy) representation of the sum
of the two operators. Sums of `LinearMap`/`LinearCombination` objects and
`LinearMap`/`LinearCombination` objects are reduced to a single `LinearCombination`.
In sums of `LinearMap`s and `AbstractMatrix`/`UniformScaling` objects, the latter
get promoted to `LinearMap`s automatically.

# Examples
```jldoctest; setup=(using LinearAlgebra, LinearMaps)
julia> CS = LinearMap{Int}(cumsum, 3)::LinearMaps.FunctionMap;

julia> LinearMap(ones(Int, 3, 3)) + CS + I + rand(3, 3);
```
"""
function Base.:(+)(A₁::LinearMap, A₂::LinearMap)
    size(A₁) == size(A₂) || throw(DimensionMismatch("+"))
    T = promote_type(eltype(A₁), eltype(A₂))
    return LinearCombination{T, mulstyle(A₁, A₂)}(tuple(A₁, A₂))
end
function Base.:(+)(A₁::LinearMap, A₂::LinearCombination)
    size(A₁) == size(A₂) || throw(DimensionMismatch("+"))
    T = promote_type(eltype(A₁), eltype(A₂))
    return LinearCombination{T, mulstyle(A₁, A₂)}(tuple(A₁, A₂.maps...))
end
Base.:(+)(A₁::LinearCombination, A₂::LinearMap) = +(A₂, A₁)
function Base.:(+)(A₁::LinearCombination, A₂::LinearCombination)
    size(A₁) == size(A₂) || throw(DimensionMismatch("+"))
    T = promote_type(eltype(A₁), eltype(A₂))
    return LinearCombination{T, mulstyle(A₁, A₂)}(tuple(A₁.maps..., A₂.maps...))
end
Base.:(-)(A₁::LinearMap, A₂::LinearMap) = +(A₁, -A₂)

# comparison of LinearCombination objects, sufficient but not necessary
Base.:(==)(A::LinearCombination, B::LinearCombination) = (eltype(A) == eltype(B) && A.maps == B.maps)

# special transposition behavior
LinearAlgebra.transpose(A::LinearCombination) = LinearCombination{eltype(A), mulstyle(A)}(map(transpose, A.maps))
LinearAlgebra.adjoint(A::LinearCombination)   = LinearCombination{eltype(A), mulstyle(A)}(map(adjoint, A.maps))

# multiplication with vectors
for Atype in (AbstractVector, AbstractMatrix)
    @eval Base.@propagate_inbounds function LinearAlgebra.mul!(y::$Atype, A::LinearCombination, x::$Atype,
                             α::Number=true, β::Number=false)
        @boundscheck check_dim_mul(y, A, x)
        return _lincombmul!(y, A, x, α, β)
    end
end

@inline function _lincombmul!(y, A::LinearCombination{<:Any,FiveArg}, x, α::Number, β::Number)
    if iszero(α) # trivial cases
        iszero(β) && (fill!(y, zero(eltype(y))); return y)
        isone(β) && return y
        # β != 0, 1
        rmul!(y, β)
        return y
    else
        mul!(y, first(A.maps), x, α, β)
        @inbounds for An in Base.tail(A.maps)
            mul!(y, An, x, α, true)
        end
        return y
    end
end

@inline function _lincombmul!(y, A::LinearCombination{<:Any,ThreeArg}, x, α::Number, β::Number)
    if iszero(α)
        iszero(β) && (fill!(y, zero(eltype(y))); return y)
        isone(β) && return y
        # β != 0, 1
        rmul!(y, β)
        return y
    else
        mul!(y, first(A.maps), x, α, β)
        l = length(A.maps)
        if l>1
            z = similar(y)
            @inbounds for n in 2:l
                An = A.maps[n]
                muladd!(mulstyle(An), y, An, x, α, z)
            end
        end
        return y
    end
end

@inline muladd!(::Type{FiveArg}, y, A, x, α, _) = mul!(y, A, x, α, true)
@inline function muladd!(::Type{ThreeArg}, y, A, x, α, z)
    A_mul_B!(z, A, x)
    y .+= isone(α) ? z : z .* α
end

A_mul_B!(y::AbstractVector, A::LinearCombination, x::AbstractVector) = mul!(y, A, x)

At_mul_B!(y::AbstractVector, A::LinearCombination, x::AbstractVector) = mul!(y, transpose(A), x)

Ac_mul_B!(y::AbstractVector, A::LinearCombination, x::AbstractVector) = mul!(y, adjoint(A), x)
