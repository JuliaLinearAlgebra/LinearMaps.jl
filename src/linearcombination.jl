struct LinearCombination{T, As<:Tuple{Vararg{LinearMap}}} <: LinearMap{T}
    maps::As
    function LinearCombination{T, As}(maps::As) where {T, As}
        N = length(maps)
        sz = size(maps[1])
        for n = 1:N
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
function Base.:(+)(A1::LinearCombination, A2::LinearCombination)
    size(A1) == size(A2) || throw(DimensionMismatch("+"))
    T = promote_type(eltype(A1), eltype(A2))
    return LinearCombination{T}(tuple(A1.maps..., A2.maps...))
end
function Base.:(+)(A1::LinearMap, A2::LinearCombination)
    size(A1) == size(A2) || throw(DimensionMismatch("+"))
    T = promote_type(eltype(A1), eltype(A2))
    return LinearCombination{T}(tuple(A1, A2.maps...))
end
Base.:(+)(A1::LinearCombination, A2::LinearMap) = +(A2, A1)
function Base.:(+)(A1::LinearMap, A2::LinearMap)
    size(A1)==size(A2) || throw(DimensionMismatch("+"))
    T = promote_type(eltype(A1), eltype(A2))
    return LinearCombination{T}(tuple(A1, A2))
end
Base.:(-)(A1::LinearMap, A2::LinearMap) = +(A1, -A2)

# comparison of LinearCombination objects, sufficient but not necessary
Base.:(==)(A::LinearCombination, B::LinearCombination) = (eltype(A) == eltype(B) && A.maps == B.maps)

# special transposition behavior
LinearAlgebra.transpose(A::LinearCombination) = LinearCombination{eltype(A)}(map(transpose, A.maps))
LinearAlgebra.adjoint(A::LinearCombination)   = LinearCombination{eltype(A)}(map(adjoint, A.maps))

# multiplication with vectors
function A_mul_B!(y::AbstractVector, A::LinearCombination, x::AbstractVector)
    # no size checking, will be done by individual maps
    A_mul_B!(y, A.maps[1], x)
    l = length(A.maps)
    if l>1
        z = similar(y)
        for n=2:l
            A_mul_B!(z, A.maps[n], x)
            y .+= z
        end
    end
    return y
end
function At_mul_B!(y::AbstractVector, A::LinearCombination, x::AbstractVector)
    # no size checking, will be done by individual maps
    At_mul_B!(y, A.maps[1], x)
    l = length(A.maps)
    if l>1
        z = similar(y)
        for n = 2:l
            At_mul_B!(z, A.maps[n], x)
            y .+= z
        end
    end
    return y
end
function Ac_mul_B!(y::AbstractVector, A::LinearCombination, x::AbstractVector)
    # no size checking, will be done by individual maps
    Ac_mul_B!(y, A.maps[1], x)
    l = length(A.maps)
    if l>1
        z = similar(y)
        for n=2:l
            Ac_mul_B!(z, A.maps[n], x)
            y .+= z
        end
    end
    return y
end
