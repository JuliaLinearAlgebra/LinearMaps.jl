module LinearMapsSparseArraysExt

import SparseArrays: sparse, blockdiag, SparseMatrixCSC
using SparseArrays: AbstractSparseMatrix

using LinearMaps
import LinearMaps: _issymmetric, _ishermitian
using LinearMaps: WrappedMap, CompositeMap, LinearCombination, ScaledMap, UniformScalingMap,
    AdjointMap, TransposeMap, BlockMap, BlockDiagonalMap, KroneckerMap, KroneckerSumMap,
    VecOrMatMap, AbstractVecOrMatOrQ, MapOrVecOrMat, convert_to_lmaps, _tail, _unsafe_mul!

using LinearMaps.LinearAlgebra

_issymmetric(A::AbstractSparseMatrix) = issymmetric(A)
_ishermitian(A::AbstractSparseMatrix) = ishermitian(A)

# blockdiagonal concatenation via extension of blockdiag

"""
    blockdiag(As::Union{LinearMap,AbstractVecOrMat,AbstractQ}...)::BlockDiagonalMap

Construct a (lazy) representation of the diagonal concatenation of the arguments.
To avoid fallback to the generic `blockdiag`, there must be a `LinearMap`
object among the first 8 arguments.
"""
blockdiag

for k in 1:8 # is 8 sufficient?
    Is = ntuple(n->:($(Symbol(:A, n))::AbstractVecOrMatOrQ), Val(k-1))
    # yields (:A1, :A2, :A3, ..., :A(k-1))
    L = :($(Symbol(:A, k))::LinearMap)
    # yields :Ak
    mapargs = ntuple(n ->:($(Symbol(:A, n))), Val(k-1))
    # yields (:LinearMap(A1), :LinearMap(A2), ..., :LinearMap(A(k-1)))

    @eval function blockdiag($(Is...), $L, As::MapOrVecOrMat...)
        return BlockDiagonalMap(convert_to_lmaps($(mapargs...))...,
                                $(Symbol(:A, k)),
                                convert_to_lmaps(As...)...)
    end
end

# conversion to sparse arrays
# sparse: create sparse matrix representation of LinearMap
function sparse(A::LinearMap{T}) where {T}
    M, N = size(A)
    rowind = Int[]
    nzval = T[]
    colptr = Vector{Int}(undef, N+1)
    v = fill(zero(T), N)
    Av = Vector{T}(undef, M)

    @inbounds for i in eachindex(v)
        v[i] = one(T)
        _unsafe_mul!(Av, A, v)
        js = findall(!iszero, Av)
        colptr[i] = length(nzval) + 1
        if length(js) > 0
            append!(rowind, js)
            append!(nzval, Av[js])
        end
        v[i] = zero(T)
    end
    colptr[N+1] = length(nzval) + 1

    return SparseMatrixCSC(M, N, colptr, rowind, nzval)
end
Base.convert(::Type{SparseMatrixCSC}, A::LinearMap) = sparse(A)
SparseMatrixCSC(A::LinearMap) = sparse(A)

sparse(A::ScaledMap{<:Any, <:Any, <:VecOrMatMap}) =
    A.λ * sparse(A.lmap.lmap)
sparse(A::WrappedMap) = sparse(A.lmap)
Base.convert(::Type{SparseMatrixCSC}, A::WrappedMap) = convert(SparseMatrixCSC, A.lmap)
for (T, t) in ((:AdjointMap, adjoint), (:TransposeMap, transpose))
    @eval sparse(A::$T) = $t(convert(SparseMatrixCSC, A.lmap))
end
function sparse(ΣA::LinearCombination{<:Any, <:Tuple{Vararg{VecOrMatMap}}})
    mats = map(A->getfield(A, :lmap), ΣA.maps)
    return sum(sparse, mats)
end
function sparse(AB::CompositeMap{<:Any, <:Tuple{VecOrMatMap, VecOrMatMap}})
    B, A = AB.maps
    return sparse(A.lmap)*sparse(B.lmap)
end
function sparse(λA::CompositeMap{<:Any, <:Tuple{VecOrMatMap, UniformScalingMap}})
    A, J = λA.maps
    return J.λ*sparse(A.lmap)
end
function sparse(Aλ::CompositeMap{<:Any, <:Tuple{UniformScalingMap, VecOrMatMap}})
    J, A = Aλ.maps
    return sparse(A.lmap)*J.λ
end
function sparse(A::BlockMap)
    return hvcat(
        A.rows,
        convert(SparseMatrixCSC, first(A.maps)),
        convert.(AbstractArray, _tail(A.maps))...
    )
end
function sparse(A::BlockDiagonalMap)
    return blockdiag(convert.(SparseMatrixCSC, A.maps)...)
end
Base.convert(::Type{AbstractMatrix}, A::BlockDiagonalMap) = sparse(A)
function sparse(A::KroneckerMap)
    return kron(
        convert(SparseMatrixCSC, first(A.maps)),
        convert.(AbstractMatrix, _tail(A.maps))...
    )
end
function sparse(L::KroneckerSumMap)
    A, B = L.maps
    IA = sparse(Diagonal(ones(Bool, size(A, 1))))
    IB = sparse(Diagonal(ones(Bool, size(B, 1))))
    return kron(convert(AbstractMatrix, A), IB) + kron(IA, convert(AbstractMatrix, B))
end

end # module LinearMapsSparseArraysExt
