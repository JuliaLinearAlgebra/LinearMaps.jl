struct LinearCombination{T, As<:Tuple{Vararg{LinearMap}}, Ts<:Tuple} <: LinearMap{T}
    maps::As
    coeffs::Ts
    function LinearCombination{T,As,Ts}(maps::As, coeffs::Ts) where {T,As,Ts}
        N = length(maps)
        N == length(coeffs) || error("Number of coefficients doesn't match number of terms")
        sz = size(maps[1])
        for n = 1:N
            size(maps[n]) == sz || throw(DimensionMismatch("LinearCombination"))
            promote_type(T, eltype(maps[n]), typeof(coeffs[n])) == T || throw(InexactError())
        end
        new{T,As,Ts}(maps, coeffs)
    end
end

(::Type{LinearCombination{T}})(maps::As, coeffs::Ts) where {T,As,Ts} = LinearCombination{T,As,Ts}(maps, coeffs)

# basic methods
Base.size(A::LinearCombination) = size(A.maps[1])
LinearAlgebra.issymmetric(A::LinearCombination) = all(issymmetric, A.maps) # sufficient but not necessary
LinearAlgebra.ishermitian(A::LinearCombination) = all(ishermitian, A.maps) && all(isreal, A.coeffs) # sufficient but not necessary
LinearAlgebra.isposdef(A::LinearCombination) = all(isposdef, A.maps) && all(isposdef, A.coeffs) # sufficient but not necessary

# adding linear maps
function Base.:(+)(A1::LinearCombination, A2::LinearCombination)
    size(A1) == size(A2) || throw(DimensionMismatch("+"))
    T = promote_type(eltype(A1), eltype(A2))
    return LinearCombination{T}(tuple(A1.maps..., A2.maps...), tuple(A1.coeffs..., A2.coeffs...))
end
function Base.:(+)(A1::LinearMap, A2::LinearCombination)
    size(A1) == size(A2) || throw(DimensionMismatch("+"))
    T = promote_type(eltype(A1), eltype(A2))
    return LinearCombination{T}(tuple(A1, A2.maps...), tuple(one(T), A2.coeffs...))
end
Base.:(+)(A1::LinearCombination, A2::LinearMap) = +(A2,A1)
function Base.:(+)(A1::LinearMap, A2::LinearMap)
    size(A1)==size(A2) || throw(DimensionMismatch("+"))
    T = promote_type(eltype(A1), eltype(A2))
    return LinearCombination{T}(tuple(A1, A2), tuple(one(T), one(T)))
end
function Base.:(-)(A1::LinearCombination, A2::LinearCombination)
    size(A1) == size(A2) || throw(DimensionMismatch("-"))
    T = promote_type(eltype(A1), eltype(A2))
    return LinearCombination{T}(tuple(A1.maps..., A2.maps...), tuple(A1.coeffs..., map(-, A2.coeffs)...))
end
function Base.:(-)(A1::LinearMap, A2::LinearCombination)
    size(A1) == size(A2) || throw(DimensionMismatch("-"))
    T = promote_type(eltype(A1), eltype(A2))
    return LinearCombination{T}(tuple(A1, A2.maps...), tuple(one(T), map(-, A2.coeffs)...))
end
function Base.:(-)(A1::LinearCombination, A2::LinearMap)
    size(A1) == size(A2) || throw(DimensionMismatch("-"))
    T = promote_type(eltype(A1), eltype(A2))
    return LinearCombination{T}(tuple(A1.maps..., A2), tuple(A1.coeffs..., -one(T)))
end
function Base.:(-)(A1::LinearMap,A2::LinearMap)
    size(A1) == size(A2) || throw(DimensionMismatch("-"))
    T = promote_type(eltype(A1), eltype(A2))
    return LinearCombination{T}(tuple(A1, A2), tuple(one(T), -one(T)))
end

# scalar multiplication
Base.:(-)(A::LinearMap) = LinearCombination{eltype(A)}(tuple(A), tuple(-one(eltype(A))))
Base.:(-)(A::LinearCombination) = LinearCombination{eltype(A)}(A.maps, map(-, A.coeffs))

function Base.:(*)(α::Number, A::LinearMap)
    T = promote_type(eltype(α), eltype(A))
    return LinearCombination{T}(tuple(A), tuple(α))
end
Base.:(*)(A::LinearMap, α::Number) = *(α,A)
function *(α::Number, A::LinearCombination)
    T = promote_type(eltype(α), eltype(A))
    return LinearCombination{T}(A.maps, map(x->α*x, A.coeffs))
end
Base.:(*)(A::LinearCombination, α::Number) = *(α, A)

function Base.:(\)(α::Number, A::LinearMap)
    T = promote_type(eltype(α), eltype(A))
    return LinearCombination{T}(tuple(A), tuple(1/α))
end
Base.:(/)(A::LinearMap, α::Number) = \(α, A)
function \(α::Number, A::LinearCombination)
    T = promote_type(eltype(α), eltype(A))
    return LinearCombination{T}(A.maps, map(x->α\x, A.coeffs))
end
Base.:(/)(A::LinearCombination, α::Number) = \(α,A)

# comparison of LinearCombination objects
Base.:(==)(A::LinearCombination, B::LinearCombination) = (eltype(A)==eltype(B) && A.maps==B.maps && A.coeffs==B.coeffs)

# special transposition behavior
LinearAlgebra.transpose(A::LinearCombination) = LinearCombination{eltype(A)}(map(transpose, A.maps), A.coeffs)
LinearAlgebra.adjoint(A::LinearCombination) = LinearCombination{eltype(A)}(map(adjoint, A.maps), map(conj, A.coeffs))

# multiplication with vectors
function A_mul_B!(y::AbstractVector, A::LinearCombination, x::AbstractVector)
    # no size checking, will be done by individual maps
    A_mul_B!(y, A.maps[1], x)
    A.coeffs[1] == 1 || lmul!(A.coeffs[1], y)
    z = similar(y)
    for n=2:length(A.maps)
        A_mul_B!(z, A.maps[n], x)
        axpy!(A.coeffs[n], z, y)
    end
    return y
end
function At_mul_B!(y::AbstractVector, A::LinearCombination, x::AbstractVector)
    # no size checking, will be done by individual maps
    At_mul_B!(y, A.maps[1], x)
    A.coeffs[1] == 1 || lmul!(A.coeffs[1], y)
    z = similar(y)
    for n = 2:length(A.maps)
        At_mul_B!(z, A.maps[n], x)
        axpy!(A.coeffs[n], z, y)
    end
    return y
end
function Ac_mul_B!(y::AbstractVector, A::LinearCombination, x::AbstractVector)
    # no size checking, will be done by individual maps
    Ac_mul_B!(y, A.maps[1], x)
    A.coeffs[1] == 1 || lmul!(conj(A.coeffs[1]), y)
    z = similar(y)
    for n=2:length(A.maps)
        Ac_mul_B!(z, A.maps[n], x)
        axpy!(conj(A.coeffs[n]), z, y)
    end
    return y
end
