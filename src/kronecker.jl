struct KroneckerMap{T, As<:LinearMapTupleOrVector} <: LinearMap{T}
    maps::As
    function KroneckerMap{T}(maps::LinearMapTupleOrVector) where {T}
        for TA in Base.Iterators.map(eltype, maps)
            promote_type(T, TA) == T ||
                error("eltype $TA cannot be promoted to $T in KroneckerMap constructor")
        end
        return new{T,typeof(maps)}(maps)
    end
end

"""
    kron(A::LinearMap, B::LinearMap)::KroneckerMap
    kron(A, B, Cs...)::KroneckerMap

Construct a (lazy) representation of the Kronecker product `A⊗B`. One of the two factors
can be an `AbstractMatrix`, which is then promoted to a `LinearMap` automatically.

To avoid fallback to the generic [`Base.kron`](@ref) in the multi-map case,
there must be a `LinearMap` object among the first 8 arguments in usage like
`kron(A, B, Cs...)`.

For convenience, one can also use `A ⊗ B` or `⊗(A, B, Cs...)` (typed as `\\otimes+TAB`) to construct the
`KroneckerMap`, even when all arguments are of `AbstractMatrix` type.

If `A`, `B`, `C` and `D` are linear maps of such size that one can form the matrix
products `A*C` and `B*D`, then the mixed-product property `(A⊗B)*(C⊗D) = (A*C)⊗(B*D)`
holds. Upon vector multiplication, this rule is checked for applicability.

# Examples
```jldoctest; setup=(using LinearAlgebra, SparseArrays, LinearMaps)
julia> J = LinearMap(I, 2) # 2×2 identity map
2×2 LinearMaps.UniformScalingMap{Bool} with scaling factor: true

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
Base.kron(A::LinearMap, B::LinearMap) =
    KroneckerMap{promote_type(eltype(A), eltype(B))}((A, B))
Base.kron(A::LinearMap, B::KroneckerMap) =
    KroneckerMap{promote_type(eltype(A), eltype(B))}(_combine(A, B.maps))
Base.kron(A::KroneckerMap, B::LinearMap) =
    KroneckerMap{promote_type(eltype(A), eltype(B))}(_combine(A.maps, B))
Base.kron(A::KroneckerMap, B::KroneckerMap) =
    KroneckerMap{promote_type(eltype(A), eltype(B))}(_combine(A.maps, B.maps))
# hoist out scalings
Base.kron(A::ScaledMap, B::LinearMap) = A.λ * kron(A.lmap, B)
Base.kron(A::LinearMap, B::ScaledMap) = kron(A, B.lmap) * B.λ
Base.kron(A::ScaledMap, B::ScaledMap) = (A.λ * B.λ) * kron(A.lmap, B.lmap)
# reduce UniformScalingMaps
Base.kron(A::UniformScalingMap, B::UniformScalingMap) = UniformScalingMap(A.λ * B.λ, A.M * B.M)
# disambiguation
Base.kron(A::ScaledMap, B::KroneckerMap) = A.λ * kron(A.lmap, B)
Base.kron(A::KroneckerMap, B::ScaledMap) = kron(A, B.lmap) * B.λ
# generic definitions
Base.kron(A::LinearMap, B::LinearMap, C::LinearMap, Ds::LinearMap...) =
    kron(kron(A, B), C, Ds...)
Base.kron(A::AbstractVecOrMat, B::LinearMap) = kron(LinearMap(A), B)
Base.kron(A::LinearMap, B::AbstractVecOrMat) = kron(A, LinearMap(B))
# promote AbstractMatrix arguments to LinearMaps, then take LinearMap-Kronecker product
for k in 3:8 # is 8 sufficient?
    Is = ntuple(n->:($(Symbol(:A, n))::AbstractVecOrMat), Val(k-1))
    # yields (:A1, :A2, :A3, ..., :A(k-1))
    L = :($(Symbol(:A, k))::LinearMap)
    # yields :Ak::LinearMap
    mapargs = ntuple(n -> :(LinearMap($(Symbol(:A, n)))), Val(k-1))
    # yields (:LinearMap(A1), :LinearMap(A2), ..., :LinearMap(A(k-1)))

    @eval Base.kron($(Is...), $L, As::MapOrVecOrMat...) =
        kron($(mapargs...), $(Symbol(:A, k)), convert_to_lmaps(As...)...)
end

@doc raw"""
    squarekron(A₁::MapOrMatrix, A₂::MapOrMatrix, A₃::MapOrMatrix, Aᵢ::MapOrMatrix...)::CompositeMap

Construct a (lazy) representation of the Kronecker product `⨂ᵢ₌₁ⁿ Aᵢ` of at least 3 _square_
Kronecker factors. In contrast to [`kron`](@ref), this function assumes that all Kronecker
factors are square, and makes use of the following identity[^1]:

```math
\bigotimes_{i=1}^n A_i = \prod_{i=1}^n I_1 \otimes \ldots \otimes I_{i-1} \otimes A_i \otimes I_{i+1} \otimes \ldots \otimes I_n
```

where ``I_k`` is an identity matrix of the size of ``A_k``. By associativity, the
Kronecker product of the identity operators may be combined to larger identity operators
``I_{1:i-1}`` and ``I_{i+1:n}``, which yields

```math
\bigotimes_{i=1}^n A_i = \prod_{i=1}^n I_{1:i-1} \otimes A_i \otimes I_{i+1:n}
```

i.e., a `CompositeMap` where each factor is a Kronecker product consisting of three maps:
outer `UniformScalingMap`s and the respective Kronecker factor. This representation is
expected to yield significantly faster multiplication (and reduce memory allocation)
compared to [`kron`](@ref), but benchmarking intended use cases is highly recommended.

[^1]: Fernandes, P. and Plateau, B. and Stewart, W. J. ["Efficient Descriptor-Vector Multiplications in Stochastic Automata Networks"](https://doi.org/10.1145/278298.278303), _Journal of the ACM_, 45(3), 381–414, 1998.
"""
function squarekron(A::MapOrMatrix, B::MapOrMatrix, C::MapOrMatrix, Ds::MapOrMatrix...)
    maps = (A, B, C, Ds...)
    T = promote_type(map(eltype, maps)...)
    all(_issquare, maps) || throw(ArgumentError("operators need to be square in Kronecker sums"))
    ns = map(a -> size(a, 1), maps)
    firstmap = first(maps) ⊗ UniformScalingMap(true, prod(ns[2:end]))
    lastmap  = UniformScalingMap(true, prod(ns[1:end-1])) ⊗ last(maps)
    middlemaps = prod(enumerate(maps[2:end-1])) do (i, map)
        UniformScalingMap(true, prod(ns[1:i])) ⊗ map ⊗ UniformScalingMap(true, prod(ns[i+2:end]))
    end
    return firstmap * middlemaps * lastmap
end

struct KronPower{p}
    function KronPower(p::Integer)
        p > 1 || throw(ArgumentError("the Kronecker power is only defined for exponents larger than 1, got $k"))
        return new{p}()
    end
end

"""
    ⊗(k::Integer)

Construct a lazy representation of the `k`-th Kronecker power
`A^⊗(k) = A ⊗ A ⊗ ... ⊗ A`, where `A` can be an `AbstractMatrix` or a `LinearMap`.
"""
⊗(k::Integer) = KronPower(k)

⊗(A, B, Cs...) = kron(convert_to_lmaps(A, B, Cs...)...)

Base.:(^)(A::MapOrMatrix, ::KronPower{p}) where {p} =
    kron(ntuple(n -> convert(LinearMap, A), Val(p))...)

Base.size(A::KroneckerMap) = map(*, size.(A.maps)...)

LinearAlgebra.issymmetric(A::KroneckerMap) = all(issymmetric, A.maps)
LinearAlgebra.ishermitian(A::KroneckerMap) = all(ishermitian, A.maps)

LinearAlgebra.adjoint(A::KroneckerMap) = KroneckerMap{eltype(A)}(map(adjoint, A.maps))
LinearAlgebra.transpose(A::KroneckerMap) = KroneckerMap{eltype(A)}(map(transpose, A.maps))

Base.:(==)(A::KroneckerMap, B::KroneckerMap) =
    (eltype(A) == eltype(B) && all(A.maps .== B.maps))

#################
# multiplication helper functions
#################

@inline function _kronmul!(Y, B, X, A, cache)
    # minimize intermediate memory allocation
    if size(B, 2) * size(A, 1) <= size(B, 1) * size(A, 2)
        _unsafe_mul!(cache, X, transpose(A))
        _unsafe_mul!(Y, B, cache)
    else
        _unsafe_mul!(cache, B, X)
        _unsafe_mul!(Y, cache, transpose(A))
    end
    return Y
end
@inline function _kronmul!(Y, B::UniformScalingMap, X, A, _)
    _unsafe_mul!(Y, X, transpose(A))
    !isone(B.λ) && lmul!(B.λ, Y)
    return Y
end
@inline function _kronmul!(Y, B, X, A::UniformScalingMap, _)
    _unsafe_mul!(Y, B, X)
    !isone(A.λ) && rmul!(Y, A.λ)
    return Y
end
# disambiguation (cannot occur)
@inline function _kronmul!(Y, B::UniformScalingMap, X, A::UniformScalingMap, _)
    mul!(parent(Y), A.λ * B.λ, parent(X))
    return Y
end
# @inline function _kronmul!(Y, B, X, A::VecOrMatMap)
#     At = transpose(A.lmap)
#     if size(B, 2) * size(A, 1) <= size(B, 1) * size(A, 2)
#         _unsafe_mul!(Y, B, X * At)
#     else
#         _unsafe_mul!(Y, Matrix(B * X), At)
#     end
#     return Y
# end
# @inline function _kronmul!(Y, B::UniformScalingMap, X, A::VecOrMatMap)
#     _unsafe_mul!(Y, X, transpose(A.lmap))
#     !isone(B.λ) && lmul!(B.λ, Y)
#     return Y
# end

const VectorMap{T} = WrappedMap{T,<:AbstractVector}
const AdjOrTransVectorMap{T} = WrappedMap{T,<:LinearAlgebra.AdjOrTransAbsVec}

#################
# multiplication with vectors
#################

const KroneckerMap2{T} = KroneckerMap{T, <:Tuple{LinearMap, LinearMap}}
const OuterProductMap{T} = KroneckerMap{T, <:Tuple{VectorMap, AdjOrTransVectorMap}}
function _unsafe_mul!(y, L::OuterProductMap, x::AbstractVector; cache=nothing)
    a, bt = L.maps
    mul!(y, a.lmap, bt.lmap * x)
end
function _unsafe_mul!(y, L::KroneckerMap2, x::AbstractVector; cache=create_cache(L, x))
    require_one_based_indexing(y)
    A, B = L.maps
    ma, na = size(A)
    mb, nb = size(B)
    X = reshape(x, (nb, na))
    Y = reshape(y, (mb, ma))
    _kronmul!(Y, B, X, A, cache)
    return y
end
function _unsafe_mul!(y, L::KroneckerMap, x::AbstractVector; cache=create_cache(L, x))
    require_one_based_indexing(y)
    maps = L.maps
    if length(maps) == 2 # reachable only for L.maps::Vector
        A, B = maps
        ma, na = size(A)
        mb, nb = size(B)
        X = reshape(x, (nb, na))
        Y = reshape(y, (mb, ma))
        _kronmul!(Y, B, X, A, cache)
    else
        A = first(maps)
        B = KroneckerMap{eltype(L)}(_tail(maps))
        ma, na = size(A)
        mb, nb = size(B)
        X = reshape(x, (nb, na))
        Y = reshape(y, (mb, ma))
        _kronmul!(Y, B, X, A, cache)
    end
    return y
end
# mixed-product rule, prefer the right if possible
# (A₁ ⊗ A₂ ⊗ ... ⊗ Aᵣ) * (B₁ ⊗ B₂ ⊗ ... ⊗ Bᵣ) = (A₁B₁) ⊗ (A₂B₂) ⊗ ... ⊗ (AᵣBᵣ)
function _unsafe_mul!(y,
                        L::CompositeMap{<:Any,<:Tuple{KroneckerMap,KroneckerMap}},
                        x::AbstractVector)
    require_one_based_indexing(y)
    B, A = L.maps
    if length(A.maps) == length(B.maps) && all(_iscompatible, zip(A.maps, B.maps))
        _unsafe_mul!(y, KroneckerMap{eltype(L)}(map(*, A.maps, B.maps)), x)
    else
        _unsafe_mul!(y, LinearMap(A)*B, x)
    end
    return y
end
# mixed-product rule, prefer the right if possible
# (A₁⊗B₁) * (A₂⊗B₂) * ... * (Aᵣ⊗Bᵣ) = (A₁*A₂*...*Aᵣ) ⊗ (B₁*B₂*...*Bᵣ)
function _unsafe_mul!(y,
                        L::CompositeMap{T, <:Union{Tuple{Vararg{KroneckerMap2}},AbstractVector{<:KroneckerMap2}}},
                        x::AbstractVector) where {T}
    require_one_based_indexing(y)
    As = map(AB -> AB.maps[1], L.maps)
    Bs = map(AB -> AB.maps[2], L.maps)
    As1, As2 = _front(As), _tail(As)
    Bs1, Bs2 = _front(Bs), _tail(Bs)
    apply = all(_iscompatible, zip(As1, As2)) && all(_iscompatible, zip(Bs1, Bs2))
    if apply
        _unsafe_mul!(y, kron(prod(As), prod(Bs)), x)
    else
        _unsafe_mul!(y, CompositeMap{T}(map(LinearMap, L.maps)), x)
    end
    return y
end

#################
# multiplication with matrices
#################

_create_cache(::UniformScalingMap, ::LinearMap, X) = nothing
_create_cache(::LinearMap, ::UniformScalingMap, X) = nothing
_create_cache(::VectorMap{T}, B::AdjOrTransVectorMap{S}, X) where {T,S} =
    LinearAlgebra.wrapperop(B.lmap)(similar(X, promote_type(T, S), size(X, 2)))
function _create_cache(A::LinearMap, B::LinearMap, X)
    if size(B, 2) * size(A, 1) <= size(B, 1) * size(A, 2)
        cache = similar(X, promote_type(eltype(A), eltype(B), eltype(X)), (size(B, 2), size(A, 1)))
    else
        cache = similar(X, promote_type(eltype(A), eltype(B), eltype(X)), (size(B, 1), size(A, 2)))
    end
    return cache
end
create_cache(K::KroneckerMap2, X) = _create_cache(first(K.maps), last(K.maps), X)
function create_cache(K::KroneckerMap, X)
    maps = K.maps
    if length(maps) == 2
        A, B = maps
        cache = _create_cache(A, B, X)
    else
        A = first(maps)
        B = KroneckerMap{eltype(K)}(_tail(maps))
        cache = _create_cache(A, B, X)
    end
    return cache
end

function _unsafe_mul!(Y, K::KroneckerMap, X::AbstractMatrix; cache=create_cache(K, X))
    Xcol = eachcol(X)
    z = similar(first(Xcol))
    for (Xi, Yi) in zip(Xcol, eachcol(Y))
        copyto!(z, Xi)
        _unsafe_mul!(Yi, K, z, cache=cache)
    end
    return Y
end
function _unsafe_mul!(y, K::OuterProductMap, X::AbstractMatrix; cache=create_cache(K, X))
    a, bt = K.maps
    mul!(cache, bt.lmap, X)
    mul!(y, a.lmap, cache)
end


###############
# KroneckerSumMap
###############
struct KroneckerSumMap{T, As<:Tuple{LinearMap, LinearMap}} <: LinearMap{T}
    maps::As
    function KroneckerSumMap{T}(maps::Tuple{LinearMap,LinearMap}) where {T}
        A1, A2 = maps
        (_issquare(A1) && _issquare(A2)) ||
            throw(ArgumentError("operators need to be square in Kronecker sums"))
        for TA in Base.Iterators.map(eltype, maps)
            promote_type(T, TA) == T ||
                error("eltype $TA cannot be promoted to $T in KroneckerSumMap constructor")
        end
        return new{T, typeof(maps)}(maps)
    end
end

"""
    kronsum(A, B)::KroneckerSumMap
    kronsum(A, B, Cs...)::KroneckerSumMap

Construct a (lazy) representation of the Kronecker sum `A⊕B = A ⊗ Ib + Ia ⊗ B`
of two square linear maps of type `LinearMap` or `AbstractMatrix`. Here, `Ia`
and `Ib` are identity operators of the size of `A` and `B`, respectively.
Arguments of type `AbstractMatrix` are automatically promoted to `LinearMap`.

For convenience, one can also use `A ⊕ B` or `⊕(A, B, Cs...)` (typed as
`\\oplus+TAB`) to construct the `KroneckerSumMap`.

# Examples
```jldoctest; setup=(using LinearAlgebra, SparseArrays, LinearMaps)
julia> J = LinearMap(I, 2) # 2×2 identity map
2×2 LinearMaps.UniformScalingMap{Bool} with scaling factor: true

julia> E = spdiagm(-1 => trues(1)); D = LinearMap(E + E' - 2I);

julia> Δ₁ = kron(D, J) + kron(J, D); # discrete 2D-Laplace operator, Kronecker sum

julia> Δ₂ = kronsum(D, D);

julia> Δ₃ = D^⊕(2);

julia> Matrix(Δ₁) == Matrix(Δ₂) == Matrix(Δ₃)
true
```
"""
kronsum(A::MapOrMatrix, B::MapOrMatrix) =
    KroneckerSumMap{promote_type(eltype(A), eltype(B))}(convert_to_lmaps(A, B))
kronsum(A::MapOrMatrix, B::MapOrMatrix, C::MapOrMatrix, Ds::MapOrMatrix...) =
    kronsum(A, kronsum(B, C, Ds...))

@doc raw"""
    sumkronsum(A, B)::LinearCombination
    sumkronsum(A, B, Cs...)::LinearCombination

Construct a (lazy) representation of the Kronecker sum `A⊕B` of two or more square
objects of type `LinearMap` or `AbstractMatrix`. This function makes use of the following
representation of Kronecker sums[^1]:

```math
\bigoplus_{i=1}^n A_i = \sum_{i=1}^n I_1 \otimes \ldots \otimes I_{i-1} \otimes A_i \otimes I_{i+1} \otimes \ldots \otimes I_n
```

where ``I_k`` is the identity operator of the size of ``A_k``. By associativity, the
Kronecker product of the identity operators may be combined to larger identity operators
``I_{1:i-1}`` and ``I_{i+1:n}``, which yields

```math
\bigoplus_{i=1}^n A_i = \sum_{i=1}^n I_{1:i-1} \otimes A_i \otimes I_{i+1:n},
```

i.e., a `LinearCombination` where each summand is a Kronecker product consisting of three
maps: outer `UniformScalingMap`s and the respective Kronecker factor. This representation is
expected to yield significantly faster multiplication (and reduce memory allocation)
compared to [`kronsum`](@ref), especially for 3 or more Kronecker summands, but
benchmarking intended use cases is highly recommended.

# Examples
```jldoctest; setup=(using LinearAlgebra, SparseArrays, LinearMaps)
julia> J = LinearMap(I, 2) # 2×2 identity map
2×2 LinearMaps.UniformScalingMap{Bool} with scaling factor: true

julia> E = spdiagm(-1 => trues(1)); D = LinearMap(E + E' - 2I);

julia> Δ₁ = kron(D, J) + kron(J, D); # discrete 2D-Laplace operator, Kronecker sum

julia> Δ₂ = sumkronsum(D, D);

julia> Δ₃ = D^⊕(2);

julia> Matrix(Δ₁) == Matrix(Δ₂) == Matrix(Δ₃)
true
```

[^1]: Fernandes, P. and Plateau, B. and Stewart, W. J. ["Efficient Descriptor-Vector Multiplications in Stochastic Automata Networks"](https://doi.org/10.1145/278298.278303), _Journal of the ACM_, 45(3), 381–414, 1998.
"""
function sumkronsum(A::MapOrMatrix, B::MapOrMatrix)
    LinearAlgebra.checksquare(A, B)
    A ⊗ UniformScalingMap(true, size(B,1)) + UniformScalingMap(true, size(A,1)) ⊗ B
end
function sumkronsum(A::MapOrMatrix, B::MapOrMatrix, C::MapOrMatrix, Ds::MapOrMatrix...)
    maps = (A, B, C, Ds...)
    all(_issquare, maps) || throw(ArgumentError("operators need to be square in Kronecker sums"))
    ns = map(a -> size(a, 1), maps)
    n = length(maps)
    firstmap = first(maps) ⊗ UniformScalingMap(true, prod(ns[2:end]))
    lastmap  = UniformScalingMap(true, prod(ns[1:end-1])) ⊗ last(maps)
    middlemaps = sum(enumerate(Base.front(Base.tail(maps)))) do (i, map)
        UniformScalingMap(true, prod(ns[1:i])) ⊗ map ⊗ UniformScalingMap(true, prod(ns[i+2:end]))
    end
    return firstmap + middlemaps + lastmap
end

struct KronSumPower{p}
    function KronSumPower(p::Integer)
        p > 1 || throw(ArgumentError("the Kronecker sum power is only defined for exponents larger than 1, got $k"))
        return new{p}()
    end
end

"""
    ⊕(k::Integer)

Construct a lazy representation of the `k`-th Kronecker sum power `A^⊕(k) = A ⊕ A ⊕ ... ⊕ A`,
where `A` can be a square `AbstractMatrix` or a `LinearMap`. This calls [`sumkronsum`](@ref)
on the `k`-tuple `(A, ..., A)` for `k ≥ 3`.

# Example
```jldoctest
julia> Matrix([1 0; 0 1]^⊕(2))
4×4 Matrix{Int64}:
 2  0  0  0
 0  2  0  0
 0  0  2  0
 0  0  0  2
"""
⊕(k::Integer) = KronSumPower(k)

⊕(a, b, c...) = kronsum(a, b, c...)

Base.:(^)(A::MapOrMatrix, ::KronSumPower{2}) =
    kronsum(convert(LinearMap, A), convert(LinearMap, A))
Base.:(^)(A::MapOrMatrix, ::KronSumPower{p}) where {p} =
    sumkronsum(ntuple(_ -> convert(LinearMap, A), Val(p))...)

Base.size(A::KroneckerSumMap, i) = prod(size.(A.maps, i))
Base.size(A::KroneckerSumMap) = (size(A, 1), size(A, 2))

LinearAlgebra.issymmetric(A::KroneckerSumMap) = all(issymmetric, A.maps)
LinearAlgebra.ishermitian(A::KroneckerSumMap) = all(ishermitian, A.maps)

LinearAlgebra.adjoint(A::KroneckerSumMap) =
    KroneckerSumMap{eltype(A)}(map(adjoint, A.maps))
LinearAlgebra.transpose(A::KroneckerSumMap) =
    KroneckerSumMap{eltype(A)}(map(transpose, A.maps))

Base.:(==)(A::KroneckerSumMap, B::KroneckerSumMap) =
    (eltype(A) == eltype(B) && all(A.maps .== B.maps))

function _unsafe_mul!(y, L::KroneckerSumMap, x::AbstractVector)
    A, B = L.maps
    a = size(A, 1)
    b = size(B, 1)
    X = reshape(x, (b, a))
    Y = reshape(y, (b, a))
    _unsafe_mul!(Y, X, transpose(A))
    _unsafe_mul!(Y, B, X, true, true)
    return y
end
