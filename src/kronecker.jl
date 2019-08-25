struct KroneckerMap{T, As<:Tuple{LinearMap,LinearMap}} <: LinearMap{T}
    maps::As
    function KroneckerMap{T, As}(maps::As) where {T, As}
        for A in maps
            promote_type(T, eltype(A)) == T || throw(InexactError())
        end
        return new{T,As}(maps)
    end
end

KroneckerMap{T}(maps::As) where {T, As<:Tuple{LinearMap,LinearMap}} = KroneckerMap{T, As}(maps)
Base.kron(A::LinearMap{TA}, B::LinearMap{TB}) where {TA,TB} = KroneckerMap{promote_type(TA,TB)}((A, B))
Base.kron(A::LinearMap, B::AbstractArray) = kron(A, LinearMap(B))
Base.kron(A::AbstractArray, B::LinearMap) = kron(LinearMap(A), B)

Base.size(A::KroneckerMap) = size(A.maps[1]) .* size(A.maps[2])

LinearAlgebra.issymmetric(A::KroneckerMap) = all(issymmetric, A.maps)
LinearAlgebra.ishermitian(A::KroneckerMap{<:Real}) = all(issymmetric, A.maps)
LinearAlgebra.ishermitian(A::KroneckerMap) = all(ishermitian, A.maps)

LinearAlgebra.adjoint(A::KroneckerMap{T}) where {T} = KroneckerMap{T}(map(adjoint, A.maps))
LinearAlgebra.transpose(A::KroneckerMap{T}) where {T} = KroneckerMap{T}(map(transpose, A.maps))

Base.:(==)(A::KroneckerMap, B::KroneckerMap) = (eltype(A) == eltype(B) && A.maps == B.maps)

function LinearMaps.A_mul_B!(y::AbstractVector, L::KroneckerMap, x::AbstractVector)
    A, B = L.maps
    ma, na = size(A)
    mb, nb = size(B)
    (length(y) == size(L, 1) && length(x) == size(L, 2)) || throw(DimensionMismatch("kronecker product"))
    X = LinearMap(reshape(x, (nb, na)))
    M = B * X * transpose(A)
    T = eltype(L)
    v = zeros(T, ma)
    @views @inbounds for i in 1:ma
        v[i] = one(T)
        mul!(y[((i-1)*mb+1):i*mb], M, v)
        v[i] = zero(T)
    end
    return y
end

LinearMaps.At_mul_B!(y::AbstractVector, A::KroneckerMap, x::AbstractVector) = A_mul_B!(y, transpose(A), x)

LinearMaps.Ac_mul_B!(y::AbstractVector, A::KroneckerMap, x::AbstractVector) = A_mul_B!(y, adjoint(A), x)
