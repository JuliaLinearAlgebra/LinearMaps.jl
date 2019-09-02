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

# basic methods
Base.size(A::LinearCombination) = size(A.maps[1])
LinearAlgebra.issymmetric(A::LinearCombination) = all(issymmetric, A.maps) # sufficient but not necessary
LinearAlgebra.ishermitian(A::LinearCombination) = all(ishermitian, A.maps) # sufficient but not necessary
LinearAlgebra.isposdef(A::LinearCombination) = all(isposdef, A.maps) # sufficient but not necessary

# adding linear maps
function Base.:(+)(A₁::LinearCombination, A₂::LinearCombination)
    size(A₁) == size(A₂) || throw(DimensionMismatch("+"))
    T = promote_type(eltype(A₁), eltype(A₂))
    return LinearCombination{T}(tuple(A₁.maps..., A₂.maps...))
end
function Base.:(+)(A₁::LinearMap, A₂::LinearCombination)
    size(A₁) == size(A₂) || throw(DimensionMismatch("+"))
    T = promote_type(eltype(A₁), eltype(A₂))
    return LinearCombination{T}(tuple(A₁, A₂.maps...))
end
Base.:(+)(A₁::LinearCombination, A₂::LinearMap) = +(A₂, A₁)
function Base.:(+)(A₁::LinearMap, A₂::LinearMap)
    size(A₁) == size(A₂) || throw(DimensionMismatch("+"))
    T = promote_type(eltype(A₁), eltype(A₂))
    return LinearCombination{T}(tuple(A₁, A₂))
end
Base.:(-)(A₁::LinearMap, A₂::LinearMap) = +(A₁, -A₂)

# comparison of LinearCombination objects, sufficient but not necessary
Base.:(==)(A::LinearCombination, B::LinearCombination) = (eltype(A) == eltype(B) && A.maps == B.maps)

# special transposition behavior
LinearAlgebra.transpose(A::LinearCombination) = LinearCombination{eltype(A)}(map(transpose, A.maps))
LinearAlgebra.adjoint(A::LinearCombination)   = LinearCombination{eltype(A)}(map(adjoint, A.maps))

# multiplication with vectors
if VERSION < v"1.3.0-alpha.115"

function A_mul_B!(y::AbstractVector, A::LinearCombination, x::AbstractVector)
    # no size checking, will be done by individual maps
    A_mul_B!(y, A.maps[1], x)
    l = length(A.maps)
    if l>1
        z = similar(y)
        for n in 2:l
            A_mul_B!(z, A.maps[n], x)
            y .+= z
        end
    end
    return y
end

else # 5-arg mul! is available for matrices

function A_mul_B!(y::AbstractVector, A::LinearCombination, x::AbstractVector)
    # no size checking, will be done by individual maps
    A_mul_B!(y, A.maps[1], x)
    l = length(A.maps)
    if l>1
        z = similar(y)
        for n in 2:l
            An = A.maps[n]
            if An isa MatrixMap
                mul!(y, An, x, true, true)
            else
                A_mul_B!(z, A.maps[n], x)
                y .+= z
            end
        end
    end
    return y
end

end # VERSION

At_mul_B!(y::AbstractVector, A::LinearCombination, x::AbstractVector) = A_mul_B!(y, transpose(A), x)

Ac_mul_B!(y::AbstractVector, A::LinearCombination, x::AbstractVector) = A_mul_B!(y, adjoint(A), x)
