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
# promote AbstractMatrix arguments to LinearMaps, then take LinearMap-Kronecker product
for k in 3:8 # is 8 sufficient?
    Is = ntuple(n->:($(Symbol(:A,n))::AbstractMatrix), Val(k-1))
    # yields (:A1, :A2, :A3, ..., :A(k-1))
    L = :($(Symbol(:A,k))::LinearMap)
    # yields :Ak
    mapargs = ntuple(n -> :(LinearMap($(Symbol(:A,n)))), Val(k-1))
    # yields (:LinearMap(A1), :LinearMap(A2), ..., LinearMap(A(k-1)))

    @eval Base.kron($(Is...), $L, As::Union{LinearMap,AbstractMatrix}...) =
        kron($(mapargs...), $(Symbol(:A,k)), As...)
end

Base.size(A::KroneckerMap) = size(A.maps[1]) .* size(A.maps[2])

LinearAlgebra.issymmetric(A::KroneckerMap) = all(issymmetric, A.maps)
LinearAlgebra.ishermitian(A::KroneckerMap{<:Real}) = all(issymmetric, A.maps)
LinearAlgebra.ishermitian(A::KroneckerMap) = all(ishermitian, A.maps)

LinearAlgebra.adjoint(A::KroneckerMap{T}) where {T} = KroneckerMap{T}(map(adjoint, A.maps))
LinearAlgebra.transpose(A::KroneckerMap{T}) where {T} = KroneckerMap{T}(map(transpose, A.maps))

function Base.:(*)(A::KroneckerMap, B::KroneckerMap)
    if (size(A.maps[1], 2) == size(B.maps[1], 1) && size(A.maps[2], 2) == size(B.maps[2], 1))
        return kron(A.maps[1]*B.maps[1], A.maps[2]*B.maps[2])
    else
        return CompositeMap{T}(tuple(B, A))
    end
end

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
