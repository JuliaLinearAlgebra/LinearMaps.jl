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

"""
    kron(A::LinearMap, B::LinearMap)

Construct a `KroneckerMap <: LinearMap` object, a (lazy) representation of the
Kronecker product of two `LinearMap`s. One of the two factors can be an `AbstractMatrix`
object, which then is promoted to `LinearMap` automatically. To avoid fallback to
the generic [`Base.kron`](@ref), there must be a `LinearMap` object among the
first 8 arguments in usage like `kron(A, B, Cs...)`.

Note: If `A`, `B`, `C` and `D` are linear maps of such size that one can form the
matrix products `A*C` and `B*D`, then the mixed-product property `(A⊗B)*(C⊗D)`
holds. Upon multiplication of Kronecker products, this rule is checked for
applicability, which leads to type-instability with a union of two types.

# Examples
```jldoctest; setup=(using LinearAlgebra, SparseArrays, LinearMaps)
julia> J = LinearMap(I, 2) # 2×2 identity map
LinearMaps.UniformScalingMap{Bool}(true, 2)

julia> E = spdiagm(-1 => trues(1)); D = E + E' - 2I;

julia> Δ = kron(D, J) + kron(J, D); # discrete 2D-Laplace operator

julia> Matrix(Δ)
4×4 Array{Int64,2}:
 -4   1   1   0
  1  -4   0   1
  1   0  -4   1
  0   1   1  -4
```
"""
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

Base.size(A::KroneckerMap, i) = prod(size.(A.maps, i))
Base.size(A::KroneckerMap) = (size(A, 1), size(A, 2))

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
    # require_one_based_indexing(y)
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

struct KroneckerSumMap{T, As<:Tuple{LinearMap,LinearMap}} <: LinearMap{T}
    maps::As
    function KroneckerSumMap{T, As}(maps::As) where {T, As}
        for A in maps
            size(A, 1) == size(A, 2) || throw(ArgumentError("operators need to be square in Kronecker sums"))
            promote_type(T, eltype(A)) == T || throw(InexactError())
        end
        return new{T,As}(maps)
    end
end

KroneckerSumMap{T}(maps::As) where {T, As<:Tuple{LinearMap,LinearMap}} = KroneckerSumMap{T, As}(maps)

"""
    kronsum(A::LinearMap, B::LinearMap)

Construct a `KroneckerSumMap <: LinearMap` object, a (lazy) representation of the
Kronecker sum `A⊕B = kron(A, Ib) + kron(Ia, B)` of two square `LinearMap`s. Here,
`Ia` and `Ib` are identity One of the two factors can be an `AbstractMatrix`
object, which then is promoted to `LinearMap` automatically. To avoid fallback to
the generic [`Base.kron`](@ref), there must be a `LinearMap` object among the
first 8 arguments in usage like `kron(A, B, Cs...)`.

Note: If `A`, `B`, `C` and `D` are linear maps of such size that one can form the
matrix products `A*C` and `B*D`, then the mixed-product property `(A⊗B)*(C⊗D)`
holds. Upon multiplication of Kronecker products, this rule is checked for
applicability, which leads to type-instability with a union of two types.

# Examples
```jldoctest; setup=(using LinearAlgebra, SparseArrays, LinearMaps)
julia> J = LinearMap(I, 2) # 2×2 identity map
LinearMaps.UniformScalingMap{Bool}(true, 2)

julia> E = spdiagm(-1 => trues(1)); D = E + E' - 2I;

julia> Δ = kron(D, J) + kron(J, D); # discrete 2D-Laplace operator

julia> Matrix(Δ)
4×4 Array{Int64,2}:
 -4   1   1   0
  1  -4   0   1
  1   0  -4   1
  0   1   1  -4
```
"""
Base.kronsum(A::LinearMap{TA}, B::LinearMap{TB}) where {TA,TB} = KroneckerMap{promote_type(TA,TB)}((A, B))
Base.kronsum(A::LinearMap, Bs::LinearMap...) = kronsum(A, kronsum(Bs...))
Base.kronsum(A::LinearMap, B::AbstractArray) = kronsum(A, LinearMap(B))
Base.kronsum(A::AbstractArray, B::LinearMap) = kronsum(LinearMap(A), B)
# promote AbstractMatrix arguments to LinearMaps, then take LinearMap-Kronecker product
for k in 3:8 # is 8 sufficient?
    Is = ntuple(n->:($(Symbol(:A,n))::AbstractMatrix), Val(k-1))
    # yields (:A1, :A2, :A3, ..., :A(k-1))
    L = :($(Symbol(:A,k))::LinearMap)
    # yields :Ak
    mapargs = ntuple(n -> :(LinearMap($(Symbol(:A,n)))), Val(k-1))
    # yields (:LinearMap(A1), :LinearMap(A2), ..., LinearMap(A(k-1)))

    @eval Base.kronsum($(Is...), $L, As::Union{LinearMap,AbstractMatrix}...) =
        kron($(mapargs...), $(Symbol(:A,k)), As...)
end

Base.size(A::KroneckerSumMap, i) = prod(size.(A.maps, i))
Base.size(A::KroneckerSumMap) = (size(A, 1), size(A, 2))

LinearAlgebra.issymmetric(A::KroneckerSumMap) = all(issymmetric, A.maps)
LinearAlgebra.ishermitian(A::KroneckerSumMap{<:Real}) = all(issymmetric, A.maps)
LinearAlgebra.ishermitian(A::KroneckerSumMap) = all(ishermitian, A.maps)

LinearAlgebra.adjoint(A::KroneckerSumMap{T}) where {T} = KroneckerSumMap{T}(map(adjoint, A.maps))
LinearAlgebra.transpose(A::KroneckerSumMap{T}) where {T} = KroneckerSumMap{T}(map(transpose, A.maps))

Base.:(==)(A::KroneckerSumMap, B::KroneckerSumMap) = (eltype(A) == eltype(B) && A.maps == B.maps)

function LinearMaps.A_mul_B!(y::AbstractVector, L::KroneckerSumMap, x::AbstractVector)
    A, B = L.maps
    ma, na = size(A)
    mb, nb = size(B)
    (length(y) == size(L, 1) && length(x) == size(L, 2)) || throw(DimensionMismatch("kronecker product"))
    X = reshape(x, (nb, na))
    Y = reshape(y, (nb, na))
    mul!(Y, X, convert(AbstractMatrix{eltype(A)}, transpose(A)))
    mul!(Y, B, X, true, true)
    return y
end

LinearMaps.At_mul_B!(y::AbstractVector, A::KroneckerSumMap, x::AbstractVector) = A_mul_B!(y, transpose(A), x)

LinearMaps.Ac_mul_B!(y::AbstractVector, A::KroneckerSumMap, x::AbstractVector) = A_mul_B!(y, adjoint(A), x)
