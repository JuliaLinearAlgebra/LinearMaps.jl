struct KroneckerMap{T, As<:Tuple{Vararg{LinearMap}}} <: LinearMap{T}
    maps::As
    function KroneckerMap{T, As}(maps::As) where {T, As}
        for A in maps
            promote_type(T, eltype(A)) == T || throw(InexactError())
        end
        return new{T,As}(maps)
    end
end

KroneckerMap{T}(maps::As) where {T, As<:Tuple{Vararg{LinearMap}}} = KroneckerMap{T, As}(maps)

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
# Base.kron(A::LinearMap, B::LinearMap) = KroneckerMap{promote_type(eltype(A), eltype(B))}((A, B))
# Base.kron(A::LinearMap{TA}, B::LinearMap{TB}) where {TA,TB} = KroneckerMap{promote_type(TA,TB)}((A, B))
Base.kron(As::LinearMap...) = KroneckerMap{promote_type(map(eltype, As)...)}(tuple(As...))
Base.kron(A::LinearMap, B::AbstractArray) = kron(A, LinearMap(B))
Base.kron(A::AbstractArray, B::LinearMap) = kron(LinearMap(A), B)
# promote AbstractMatrix arguments to LinearMaps, then take LinearMap-Kronecker product
for k in 3:8 # is 8 sufficient?
    Is = ntuple(n->:($(Symbol(:A,n))::AbstractMatrix), Val(k-1))
    # yields (:A1, :A2, :A3, ..., :A(k-1))
    L = :($(Symbol(:A,k))::LinearMap)
    # yields :Ak
    mapargs = ntuple(n -> :(LinearMap($(Symbol(:A,n)))), Val(k-1))
    # yields (:LinearMap(A1), :LinearMap(A2), ..., :LinearMap(A(k-1)))

    @eval Base.kron($(Is...), $L, As::Union{LinearMap,AbstractMatrix}...) =
        kron($(mapargs...), $(Symbol(:A,k)), promote_to_lmaps(As...)...)
end

promote_to_lmaps_(A::AbstractMatrix) = LinearMap(A)
promote_to_lmaps_(A::LinearMap) = A
promote_to_lmaps() = ()
promote_to_lmaps(A) = (promote_to_lmaps_(A),)
@inline promote_to_lmaps(A, B, Cs...) =
    (promote_to_lmaps_(A), promote_to_lmaps_(B), promote_to_lmaps(Cs...)...)

Base.size(A::KroneckerMap) = (prod(size.(A.maps, 1)), prod(size.(A.maps, 2)))

LinearAlgebra.issymmetric(A::KroneckerMap) = all(issymmetric, A.maps)
LinearAlgebra.ishermitian(A::KroneckerMap{<:Real}) = all(issymmetric, A.maps)
LinearAlgebra.ishermitian(A::KroneckerMap) = all(ishermitian, A.maps)

LinearAlgebra.adjoint(A::KroneckerMap{T}) where {T} = KroneckerMap{T}(map(adjoint, A.maps))
LinearAlgebra.transpose(A::KroneckerMap{T}) where {T} = KroneckerMap{T}(map(transpose, A.maps))

function Base.:(*)(A::KroneckerMap, B::KroneckerMap)
    if length(A.maps) == length(B.maps) && all(M -> size(M[1], 2) == size(M[2], 1), zip(A.maps, B.maps))
        return kron(map(prod, zip(A.maps, B.maps))...)
    else
        return CompositeMap{T}(tuple(B, A))
    end
end

Base.:(==)(A::KroneckerMap, B::KroneckerMap) = (eltype(A) == eltype(B) && A.maps == B.maps)

function LinearMaps.A_mul_B!(y::AbstractVector, L::KroneckerMap{T,<:NTuple{2,LinearMap}}, x::AbstractVector) where {T}
    # require_one_based_indexing(y)
    (length(y) == size(L, 1) && length(x) == size(L, 2)) || throw(DimensionMismatch("A_mul_B!"))
    A, B = L.maps
    X = LinearMap(reshape(x, (size(B, 2), size(A, 2))); issymmetric=false, ishermitian=false, isposdef=false)
    _kronmul!(y, B, X, transpose(A), T)
    return y
end
function LinearMaps.A_mul_B!(y::AbstractVector, L::KroneckerMap{T}, x::AbstractVector) where {T}
    # require_one_based_indexing(y)
    (length(y) == size(L, 1) && length(x) == size(L, 2)) || throw(DimensionMismatch("A_mul_B!"))
    A = first(L.maps)
    B = kron(Base.tail(L.maps)...)
    X = LinearMap(reshape(x, (size(B, 2), size(A, 2))); issymmetric=false, ishermitian=false, isposdef=false)
    _kronmul!(y, B, X, transpose(A), T)
    return y
end

function _kronmul!(y, B, X, At, T)
    na, ma = size(At)
    mb, nb = size(B)
    v = zeros(T, ma)
    Ty = eltype(y)
    temp1 = Array{Ty}(undef, na)
    temp2 = Array{Ty}(undef, nb)
    @views @inbounds for i in 1:ma
        v[i] = one(T)
        A_mul_B!(temp1, At, v)
        A_mul_B!(temp2, X, temp1)
        A_mul_B!(y[((i-1)*mb+1):i*mb], B, temp2)
        v[i] = zero(T)
    end
    return y
end
# function _kronmul!(y, B::MatrixMap, X, At::MatrixMap, T)
#     na, ma = size(At)
#     mb, nb = size(B)
#     if (nb + ma) * na < (ma + mb) * nb
#         mul!(reshape(y, (size(B, 1), size(At, 2))), B.lmap, X*At.lmap)
#     else
#         mul!(reshape(y, (size(B, 1), size(At, 2))), B.lmap*X, At.lmap)
#     end
#     return y
# end

LinearMaps.At_mul_B!(y::AbstractVector, A::KroneckerMap, x::AbstractVector) = A_mul_B!(y, transpose(A), x)

LinearMaps.Ac_mul_B!(y::AbstractVector, A::KroneckerMap, x::AbstractVector) = A_mul_B!(y, adjoint(A), x)

###############
# KroneckerSumMap
###############
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
    kronsum(A, B, Cs...)

Construct a `KroneckerSumMap <: LinearMap` object, a (lazy) representation of the
Kronecker sum `A⊕B = kron(A, Ib) + kron(Ia, B)` of two square `LinearMap`s. Here,
`Ia` and `Ib` are identity operators of the size of `A` and `B`, respectively.
Arguments of type `AbstractMatrix` are promoted to `LinearMap`s as long as there
is a `LinearMap` object among the first 8 arguments.

# Examples
```jldoctest; setup=(using LinearAlgebra, SparseArrays, LinearMaps)
julia> J = LinearMap(I, 2) # 2×2 identity map
LinearMaps.UniformScalingMap{Bool}(true, 2)

julia> E = spdiagm(-1 => trues(1)); D = LinearMap(E + E' - 2I);

julia> Δ₁ = kron(D, J) + kron(J, D); # discrete 2D-Laplace operator

julia> Δ₂ = LinearMaps.kronsum(D, D); # same operator as Kronecker sum

julia> Matrix(Δ₁) == Matrix(Δ₂)
true
```
"""
kronsum(A::LinearMap{TA}, B::LinearMap{TB}) where {TA,TB} = KroneckerSumMap{promote_type(TA,TB)}((A, B))
kronsum(A::LinearMap, B::LinearMap, C::LinearMap, Ds::LinearMap...) = kronsum(A, kronsum(B, C, Ds...))
# kronsum(A::LinearMap, B::AbstractArray) = kronsum(A, LinearMap(B))
# kronsum(A::AbstractArray, B::LinearMap) = kronsum(LinearMap(A), B)
# promote AbstractMatrix arguments to LinearMaps, then take LinearMap-Kronecker sum
for k in 1:8 # is 8 sufficient?
    Is = ntuple(n->:($(Symbol(:A,n))::AbstractMatrix), Val(k-1))
    # yields (:A1, :A2, :A3, ..., :A(k-1))
    L = :($(Symbol(:A,k))::LinearMap)
    # yields :Ak
    mapargs = ntuple(n -> :(LinearMap($(Symbol(:A,n)))), Val(k-1))
    # yields (:LinearMap(A1), :LinearMap(A2), ..., :LinearMap(A(k-1)))

    @eval kronsum($(Is...), $L, As::Union{LinearMap,AbstractMatrix}...) =
        kronsum($(mapargs...), $(Symbol(:A,k)), promote_to_lmaps(As...)...)
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
    mul!(Y, X, convert(AbstractMatrix, transpose(A)))
    mul!(Y, B, X, true, true)
    return y
end

LinearMaps.At_mul_B!(y::AbstractVector, A::KroneckerSumMap, x::AbstractVector) = A_mul_B!(y, transpose(A), x)

LinearMaps.Ac_mul_B!(y::AbstractVector, A::KroneckerSumMap, x::AbstractVector) = A_mul_B!(y, adjoint(A), x)

"""
    kron(A::LinearMap, k::Int)

Construct a lazy representation of the `k`-th Kronecker power `A ⊗ A ⊗ ... ⊗ A`
for `k≥2`. This function is currently not type-stable!
"""
function Base.kron(A::LinearMap, k::Int)
    k > 1 || throw(ArgumentError("the Kronecker power is only defined for exponents larger than 1, got $k"))
    return kron(Base.fill_to_length((), A, Val(k))...)
end
