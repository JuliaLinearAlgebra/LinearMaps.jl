type LinearCombination{T, As<:Tuple{Vararg{AbstractLinearMap}}, Ts<:Tuple} <: AbstractLinearMap{T}
    maps::As
    coeffs::Ts
    function LinearCombination(maps::As, coeffs::Ts)
        N = length(maps)
        N == length(coeffs) || error("Number of coefficients doesn't match number of terms")
        sz = size(maps[1])
        for n = 1:N
            size(maps[n]) == sz || throw(DimensionMismatch("LinearCombination"))
            promote_type(T, eltype(maps[n]), typeof(coeffs[n])) == T || throw(InexactError())
        end
        new(maps, coeffs)
    end
end

(::Type{LinearCombination{T}}){T,As,Ts}(maps::As, coeffs::Ts) = LinearCombination{T,As,Ts}(maps, coeffs)

# basic methods
Base.size(A::LinearCombination) = size(A.maps[1])
Base.isreal(A::LinearCombination) = all(isreal, A.maps) && all(isreal, A.coeffs) # sufficient but not necessary
Base.issymmetric(A::LinearCombination) = all(issymmetric, A.maps) # sufficient but not necessary
Base.ishermitian(A::LinearCombination) = all(ishermitian, A.maps) && all(isreal, A.coeffs) # sufficient but not necessary
Base.isposdef(A::LinearCombination) = all(isposdef, A.maps) && all(isposdef, A.coeffs) # sufficient but not necessary

# adding linear maps
function +(A1::LinearCombination, A2::LinearCombination)
    size(A1) == size(A2) || throw(DimensionMismatch("+"))
    T = promote_type(eltype(A1), eltype(A2))
    return LinearCombination{T}(tuple(A1.maps..., A2.maps...), tuple(A1.coeffs..., A2.coeffs...))
end
function +(A1::AbstractLinearMap, A2::LinearCombination)
    size(A1) == size(A2) || throw(DimensionMismatch("+"))
    T = promote_type(eltype(A1), eltype(A2))
    return LinearCombination{T}(tuple(A1, A2.maps...), tuple(one(T), A2.coeffs...))
end
+(A1::LinearCombination, A2::AbstractLinearMap) = +(A2,A1)
function +(A1::AbstractLinearMap, A2::AbstractLinearMap)
    size(A1)==size(A2) || throw(DimensionMismatch("+"))
    T = promote_type(eltype(A1), eltype(A2))
    return LinearCombination{T}(tuple(A1,A2), tuple(one(T),one(T)))
end
function -(A1::LinearCombination, A2::LinearCombination)
    size(A1) == size(A2) || throw(DimensionMismatch("-"))
    T = promote_type(eltype(A1), eltype(A2))
    return LinearCombination{T}(tuple(A1.maps..., A2.maps...), tuple(A1.coeffs..., map(-,A2.coeffs)...))
end
function -(A1::AbstractLinearMap, A2::LinearCombination)
    size(A1) == size(A2) || throw(DimensionMismatch("-"))
    T = promote_type(eltype(A1), eltype(A2))
    return LinearCombination{T}(tuple(A1, A2.maps...), tuple(one(T), map(-,A2.coeffs)...))
end
function -(A1::LinearCombination, A2::AbstractLinearMap)
    size(A1) == size(A2) || throw(DimensionMismatch("-"))
    T = promote_type(eltype(A1), eltype(A2))
    return LinearCombination{T}(tuple(A1.maps..., A2), tuple(A1.coeffs..., -one(T)))
end
function -(A1::AbstractLinearMap,A2::AbstractLinearMap)
    size(A1) == size(A2) || throw(DimensionMismatch("-"))
    T = promote_type(eltype(A1), eltype(A2))
    return LinearCombination{T}(tuple(A1, A2), tuple(one(T), -one(T)))
end

# scalar multiplication
-(A::AbstractLinearMap) = LinearCombination{eltype(A)}(tuple(A), tuple(-one(eltype(A))))
-(A::LinearCombination) = LinearCombination{eltype(A)}(A.maps, map(-, A.coeffs))

function *(α::Number, A::AbstractLinearMap)
    T = promote_type(eltype(α), eltype(A))
    return LinearCombination{T}(tuple(A), tuple(α))
end
*(A::AbstractLinearMap, α::Number) = *(α,A)
function *(α::Number, A::LinearCombination)
    T = promote_type(eltype(α), eltype(A))
    return LinearCombination{T}(A.maps, map(x->α*x, A.coeffs))
end
*(A::LinearCombination, α::Number) = *(α,A)

function \(α::Number, A::AbstractLinearMap)
    T = promote_type(eltype(α), eltype(A))
    return LinearCombination{T}(tuple(A), tuple(1/α))
end
/(A::AbstractLinearMap, α::Number) = \(α, A)
function \(α::Number, A::LinearCombination)
    T = promote_type(eltype(α), eltype(A))
    return LinearCombination{T}(A.maps, map(x->α\x, A.coeffs))
end
/(A::LinearCombination, α::Number) = \(α,A)

# comparison of LinearCombination objects
==(A::LinearCombination, B::LinearCombination) = (eltype(A)==eltype(B) && A.maps==B.maps && A.coeffs==B.coeffs)

# special transposition behavior
Base.transpose(A::LinearCombination) = LinearCombination{eltype(A)}(map(transpose, A.maps), A.coeffs)
Base.ctranspose(A::LinearCombination) = LinearCombination{eltype(A)}(map(ctranspose, A.maps), map(conj, A.coeffs))

# multiplication with vectors
function Base.A_mul_B!(y::AbstractVector, A::LinearCombination, x::AbstractVector)
    # no size checking, will be done by individual maps
    Base.A_mul_B!(y, A.maps[1], x)
    A.coeffs[1] == 1 || scale!(A.coeffs[1], y)
    z = similar(y)
    for n=2:length(A.maps)
        Base.A_mul_B!(z, A.maps[n], x)
        Base.axpy!(A.coeffs[n], z, y)
    end
    return y
end
function Base.At_mul_B!(y::AbstractVector, A::LinearCombination, x::AbstractVector)
    # no size checking, will be done by individual maps
    Base.At_mul_B!(y, A.maps[1], x)
    A.coeffs[1] == 1 || scale!(A.coeffs[1], y)
    z = similar(y)
    for n = 2:length(A.maps)
        Base.At_mul_B!(z, A.maps[n], x)
        Base.axpy!(A.coeffs[n], z, y)
    end
    return y
end
function Base.Ac_mul_B!(y::AbstractVector, A::LinearCombination, x::AbstractVector)
    # no size checking, will be done by individual maps
    Base.Ac_mul_B!(y, A.maps[1], x)
    A.coeffs[1] == 1 || scale!(conj(A.coeffs[1]), y)
    z = similar(y)
    for n=2:length(A.maps)
        Base.Ac_mul_B!(z, A.maps[n], x)
        Base.axpy!(conj(A.coeffs[n]), z, y)
    end
    return y
end
