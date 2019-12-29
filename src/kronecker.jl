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
    kron(A, B, Cs...)

Construct a `KroneckerMap <: LinearMap` object, a (lazy) representation of the
Kronecker product of two `LinearMap`s. One of the two factors can be an `AbstractMatrix`,
which is then promoted to a `LinearMap` automatically.

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
Base.kron(A::LinearMap, B::LinearMap) = KroneckerMap{promote_type(eltype(A), eltype(B))}((A, B))
Base.kron(A::LinearMap, B::KroneckerMap) = KroneckerMap{promote_type(eltype(A), eltype(B))}(tuple(A, B.maps...))
Base.kron(A::KroneckerMap, B::LinearMap) = KroneckerMap{promote_type(eltype(A), eltype(B))}(tuple(A.maps..., B))
Base.kron(A::KroneckerMap, B::KroneckerMap) = KroneckerMap{promote_type(eltype(A), eltype(B))}(tuple(A.maps..., B.maps...))
Base.kron(A::LinearMap, B::LinearMap, Cs::LinearMap...) = KroneckerMap{promote_type(eltype(A), eltype(B), map(eltype, Cs)...)}(tuple(A, B, Cs...))
Base.kron(A::AbstractMatrix, B::LinearMap) = kron(LinearMap(A), B)
Base.kron(A::LinearMap, B::AbstractMatrix) = kron(A, LinearMap(B))
# promote AbstractMatrix arguments to LinearMaps, then take LinearMap-Kronecker product
for k in 3:8 # is 8 sufficient?
    Is = ntuple(n->:($(Symbol(:A,n))::AbstractMatrix), Val(k-1))
    # yields (:A1, :A2, :A3, ..., :A(k-1))
    L = :($(Symbol(:A,k))::LinearMap)
    # yields :Ak::LinearMap
    mapargs = ntuple(n -> :(LinearMap($(Symbol(:A,n)))), Val(k-1))
    # yields (:LinearMap(A1), :LinearMap(A2), ..., :LinearMap(A(k-1)))

    @eval Base.kron($(Is...), $L, As::MapOrMatrix...) =
        kron($(mapargs...), $(Symbol(:A,k)), convert_to_lmaps(As...)...)
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

⊗(A, B, Cs...) = KroneckerMap{promote_type(eltype(A), eltype(B), map(eltype, Cs)...)}(convert_to_lmaps(A, B, Cs...))

Base.:(^)(A::MapOrMatrix, ::KronPower{p}) where {p} =
    ⊗(ntuple(n -> convert_to_lmaps_(A), Val(p))...)

Base.size(A::KroneckerMap) = map(*, size.(A.maps)...)

LinearAlgebra.issymmetric(A::KroneckerMap) = all(issymmetric, A.maps)
LinearAlgebra.ishermitian(A::KroneckerMap{<:Real}) = issymmetric(A)
LinearAlgebra.ishermitian(A::KroneckerMap) = all(ishermitian, A.maps)

LinearAlgebra.adjoint(A::KroneckerMap{T}) where {T} = KroneckerMap{T}(map(adjoint, A.maps))
LinearAlgebra.transpose(A::KroneckerMap{T}) where {T} = KroneckerMap{T}(map(transpose, A.maps))

Base.:(==)(A::KroneckerMap, B::KroneckerMap) = (eltype(A) == eltype(B) && A.maps == B.maps)

#################
# multiplication helper functions
#################

@inline function _kronmul!(y, B, X, At, T)
    na, ma = size(At)
    mb, nb = size(B)
    v = zeros(T, ma)
    Ty = eltype(y)
    temp1 = similar(y, Ty, na)
    temp2 = similar(y, Ty, nb)
    @views @inbounds for i in 1:ma
        v[i] = one(T)
        A_mul_B!(temp1, At, v)
        A_mul_B!(temp2, X, temp1)
        A_mul_B!(y[((i-1)*mb+1):i*mb], B, temp2)
        v[i] = zero(T)
    end
    return y
end
@inline function _kronmul!(y, B::Union{MatrixMap,UniformScalingMap}, X, At::Union{MatrixMap,UniformScalingMap}, T)
    na, ma = size(At)
    mb, nb = size(B)
    if nb*ma < mb*na
        mul!(reshape(y, (mb, ma)), B, convert(Matrix, X*At))
    else
        mul!(reshape(y, (mb, ma)), convert(Matrix, B*X), At isa MatrixMap ? At.lmap : At.λ)
    end
    return y
end

#################
# multiplication with vectors
#################

Base.@propagate_inbounds function A_mul_B!(y::AbstractVector, L::KroneckerMap{T,<:NTuple{2,LinearMap}}, x::AbstractVector) where {T}
    require_one_based_indexing(y)
    @boundscheck check_dim_mul(y, L, x)
    A, B = L.maps
    X = LinearMap(reshape(x, (size(B, 2), size(A, 2))); issymmetric=false, ishermitian=false, isposdef=false)
    _kronmul!(y, B, X, transpose(A), T)
    return y
end
Base.@propagate_inbounds function A_mul_B!(y::AbstractVector, L::KroneckerMap{T}, x::AbstractVector) where {T}
    require_one_based_indexing(y)
    @boundscheck check_dim_mul(y, L, x)
    A = first(L.maps)
    B = kron(Base.tail(L.maps)...)
    X = LinearMap(reshape(x, (size(B, 2), size(A, 2))); issymmetric=false, ishermitian=false, isposdef=false)
    _kronmul!(y, B, X, transpose(A), T)
    return y
end
# mixed-product rule, prefer the right if possible
# (A₁ ⊗ A₂ ⊗ ... ⊗ Aᵣ) * (B₁ ⊗ B₂ ⊗ ... ⊗ Bᵣ) = (A₁B₁) ⊗ (A₂B₂) ⊗ ... ⊗ (AᵣBᵣ)
Base.@propagate_inbounds function A_mul_B!(y::AbstractVector, L::CompositeMap{<:Any,<:Tuple{KroneckerMap,KroneckerMap}}, x::AbstractVector)
    B, A = L.maps
    if length(A.maps) == length(B.maps) && all(M -> check_dim_mul(M[1], M[2]), zip(A.maps, B.maps))
        A_mul_B!(y, kron(map(*, A.maps, B.maps)...), x)
    else
        A_mul_B!(y, LinearMap(A)*B, x)
    end
end
# mixed-product rule, prefer the right if possible
# (A₁ ⊗ B₁)*(A₂⊗B₂)*...*(Aᵣ⊗Bᵣ) = (A₁*A₂*...*Aᵣ) ⊗ (B₁*B₂*...*Bᵣ)
Base.@propagate_inbounds function A_mul_B!(y::AbstractVector, L::CompositeMap{T,<:Tuple{Vararg{KroneckerMap{<:Any,<:Tuple{LinearMap,LinearMap}}}}}, x::AbstractVector) where {T}
    As = map(AB -> AB.maps[1], L.maps)
    Bs = map(AB -> AB.maps[2], L.maps)
    As1, As2 = Base.front(As), Base.tail(As)
    Bs1, Bs2 = Base.front(Bs), Base.tail(Bs)
    apply = all(A -> check_dim_mul(A...), zip(As1, As2)) && all(A -> check_dim_mul(A...), zip(Bs1, Bs2))
    if apply
        A_mul_B!(y, kron(prod(As), prod(Bs)), x)
    else
        A_mul_B!(y, CompositeMap{T}(map(LinearMap, L.maps)), x)
    end
end

Base.@propagate_inbounds At_mul_B!(y::AbstractVector, A::KroneckerMap, x::AbstractVector) =
    A_mul_B!(y, transpose(A), x)

Base.@propagate_inbounds Ac_mul_B!(y::AbstractVector, A::KroneckerMap, x::AbstractVector) =
    A_mul_B!(y, adjoint(A), x)

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
    kronsum(A, B)
    kronsum(A, B, Cs...)

Construct a `KroneckerSumMap <: LinearMap` object, a (lazy) representation of the
Kronecker sum `A⊕B = A ⊗ Ib + Ia ⊗ B` of two square linear maps of type
`LinearMap` or `AbstractMatrix`. Here, `Ia` and `Ib` are identity operators of
the size of `A` and `B`, respectively. Arguments of type `AbstractMatrix` are
automatically promoted to `LinearMap`s.

For convenience, one can also use `A ⊕ B` or `⊕(A, B, Cs...)` (typed as
`\\oplus+TAB`) to construct the `KroneckerSumMap`.

# Examples
```jldoctest; setup=(using LinearAlgebra, SparseArrays, LinearMaps)
julia> J = LinearMap(I, 2) # 2×2 identity map
LinearMaps.UniformScalingMap{Bool}(true, 2)

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

struct KronSumPower{p}
    function KronSumPower(p::Integer)
        p > 1 || throw(ArgumentError("the Kronecker sum power is only defined for exponents larger than 1, got $k"))
        return new{p}()
    end
end

"""
    ⊕(k::Integer)

Construct a lazy representation of the `k`-th Kronecker sum power `A^⊕(k) = A ⊕ A ⊕ ... ⊕ A`,
where `A` can be a square `AbstractMatrix` or a `LinearMap`.
"""
⊕(k::Integer) = KronSumPower(k)

⊕(a, b, c...) = kronsum(a, b, c...)

Base.:(^)(A::MapOrMatrix, ::KronSumPower{p}) where {p} = kronsum(ntuple(n -> convert_to_lmaps_(A), Val(p))...)

Base.size(A::KroneckerSumMap, i) = prod(size.(A.maps, i))
Base.size(A::KroneckerSumMap) = (size(A, 1), size(A, 2))

LinearAlgebra.issymmetric(A::KroneckerSumMap) = all(issymmetric, A.maps)
LinearAlgebra.ishermitian(A::KroneckerSumMap{<:Real}) = all(issymmetric, A.maps)
LinearAlgebra.ishermitian(A::KroneckerSumMap) = all(ishermitian, A.maps)

LinearAlgebra.adjoint(A::KroneckerSumMap{T}) where {T} = KroneckerSumMap{T}(map(adjoint, A.maps))
LinearAlgebra.transpose(A::KroneckerSumMap{T}) where {T} = KroneckerSumMap{T}(map(transpose, A.maps))

Base.:(==)(A::KroneckerSumMap, B::KroneckerSumMap) = (eltype(A) == eltype(B) && A.maps == B.maps)

Base.@propagate_inbounds function A_mul_B!(y::AbstractVector, L::KroneckerSumMap, x::AbstractVector)
    @boundscheck check_dim_mul(y, L, x)
    A, B = L.maps
    ma, na = size(A)
    mb, nb = size(B)
    X = reshape(x, (nb, na))
    Y = reshape(y, (nb, na))
    mul!(Y, X, convert(AbstractMatrix, transpose(A)))
    mul!(Y, B, X, true, true)
    return y
end

Base.@propagate_inbounds At_mul_B!(y::AbstractVector, A::KroneckerSumMap, x::AbstractVector) =
    A_mul_B!(y, transpose(A), x)

Base.@propagate_inbounds Ac_mul_B!(y::AbstractVector, A::KroneckerSumMap, x::AbstractVector) =
    A_mul_B!(y, adjoint(A), x)
