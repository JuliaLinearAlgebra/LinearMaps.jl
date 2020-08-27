struct LinearCombination{T, As<:Tuple{Vararg{LinearMap}}} <: LinearMap{T}
    maps::As
    function LinearCombination{T, As}(maps::As) where {T, As}
        N = length(maps)
        sz = size(maps[1])
        for n in 1:N
            size(maps[n]) == sz || throw(DimensionMismatch("LinearCombination"))
            promote_type(T, eltype(maps[n])) == T || throw(InexactError())
        end
        new{T, As}(maps)
    end
end

LinearCombination{T}(maps::As) where {T, As} = LinearCombination{T, As}(maps)

MulStyle(A::LinearCombination) = MulStyle(A.maps...)

# basic methods
Base.size(A::LinearCombination) = size(A.maps[1])
LinearAlgebra.issymmetric(A::LinearCombination) = all(issymmetric, A.maps) # sufficient but not necessary
LinearAlgebra.ishermitian(A::LinearCombination) = all(ishermitian, A.maps) # sufficient but not necessary
LinearAlgebra.isposdef(A::LinearCombination) = all(isposdef, A.maps) # sufficient but not necessary

# adding linear maps
"""
    +(A::LinearMap, B::LinearMap)

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
    return LinearCombination{T}(tuple(A₁, A₂))
end
function Base.:(+)(A₁::LinearMap, A₂::LinearCombination)
    size(A₁) == size(A₂) || throw(DimensionMismatch("+"))
    T = promote_type(eltype(A₁), eltype(A₂))
    return LinearCombination{T}(tuple(A₁, A₂.maps...))
end
Base.:(+)(A₁::LinearCombination, A₂::LinearMap) = +(A₂, A₁)
function Base.:(+)(A₁::LinearCombination, A₂::LinearCombination)
    size(A₁) == size(A₂) || throw(DimensionMismatch("+"))
    T = promote_type(eltype(A₁), eltype(A₂))
    return LinearCombination{T}(tuple(A₁.maps..., A₂.maps...))
end
Base.:(-)(A₁::LinearMap, A₂::LinearMap) = +(A₁, -A₂)

# comparison of LinearCombination objects, sufficient but not necessary
Base.:(==)(A::LinearCombination, B::LinearCombination) = (eltype(A) == eltype(B) && A.maps == B.maps)

# special transposition behavior
LinearAlgebra.transpose(A::LinearCombination) = LinearCombination{eltype(A)}(map(transpose, A.maps))
LinearAlgebra.adjoint(A::LinearCombination)   = LinearCombination{eltype(A)}(map(adjoint, A.maps))

# multiplication with vectors & matrices
for (intype, outtype) in Any[Any[AbstractVector, AbstractVecOrMat], Any[AbstractMatrix, AbstractMatrix]]
    @eval begin
        @propagate_inbounds function mul!(y::$outtype, A::LinearCombination, x::$intype)
            @boundscheck check_dim_mul(y, A, x)
            @inbounds mul!(y, first(A.maps), x)
            @inbounds _mul!(MulStyle(A), y, A, x, true)
            return y
        end

        @propagate_inbounds function mul!(y::$outtype, A::LinearCombination, x::$intype,
                                α::Number, β::Number)
            @boundscheck check_dim_mul(y, A, x)
            @inbounds if iszero(α) # trivial cases
                iszero(β) && return fill!(y, zero(eltype(y)))
                isone(β) && return y
                return rmul!(y, β)
            else
                A1 = first(A.maps)
                if MulStyle(A1) === ThreeArg() && !iszero(β)
                    # if we need an intermediate vector, allocate here and reuse in
                    # LinearCombination multiplication
                    !isone(β) && rmul!(y, β)
                    z = similar(y)
                    muladd!(ThreeArg(), y, A1, x, α, z)
                    __mul!(y, Base.tail(A.maps), x, α, z)
                else # MulStyle(A1) === FiveArg() || β == 0
                    # this is allocation-free
                    mul!(y, A1, x, α, β)
                    # let _mul! decide whether an intermediate vector needs to be allocated
                    _mul!(MulStyle(A), y, A, x, α)
                end
                return y
            end
        end
    end
end

@propagate_inbounds function _mul!(::FiveArg, y, A::LinearCombination, x, α)
    return __mul!(y, Base.tail(A.maps), x, α, nothing)
end
@propagate_inbounds function _mul!(::ThreeArg, y, A::LinearCombination, x, α)
    return __mul!(y, Base.tail(A.maps), x, α, similar(y))
end

@propagate_inbounds __mul!(y, As::Tuple{Vararg{LinearMap}}, x, α, z) =
    __mul!(__mul!(y, first(As), x, α, z), Base.tail(As), x, α, z)
@propagate_inbounds __mul!(y, A::Tuple{LinearMap}, x, α, z) = __mul!(y, first(A), x, α, z)
@propagate_inbounds __mul!(y, A::LinearMap, x, α, z) = muladd!(MulStyle(A), y, A, x, α, z)

@propagate_inbounds muladd!(::FiveArg, y, A, x, α, _) = mul!(y, A, x, α, true)
@propagate_inbounds function muladd!(::ThreeArg, y, A, x, α, z)
    mul!(z, A, x)
    if isone(α)
        y .+= z
    else
        y .+= z .* α
    end
    return y
end
