struct IndexMap{T, As <: LinearMap, Rs <: AbstractVector{Int},
                    Cs <: AbstractVector{Int}} <: LinearMap{T}
    lmap::As
    dims::Dims{2}
    rows::Rs # typically i1:i2 with 1 <= i1 <= i2 <= size(map,1)
    cols::Cs # typically j1:j2 with 1 <= j1 <= j2 <= size(map,2)

    function IndexMap{T}(map::As, dims::Dims{2}, rows::Rs, cols::Cs) where {T,
                    As <: LinearMap, Rs <: AbstractVector{Int}, Cs <: AbstractVector{Int}}
        check_index(rows, size(map, 1), dims[1])
        check_index(cols, size(map, 2), dims[2])
        return new{T,As,Rs,Cs}(map, dims, rows, cols)
    end
end

IndexMap(map::LinearMap{T}, dims::Dims{2}; offset::Dims{2}) where {T} =
    IndexMap{T}(map, dims, offset[1] .+ (1:size(map, 1)), offset[2] .+ (1:size(map, 2)))
IndexMap(map::LinearMap, dims::Dims{2}, rows::AbstractVector{Int}, cols::AbstractVector{Int}) =
    IndexMap{eltype(map)}(map, dims, rows, cols)

Base.reverse(A::LinearMap; dims=:) = _reverse(A, dims)
function _reverse(A, dims::Integer)
    if dims == 1
        return IndexMap(A, size(A), reverse(axes(A, 1)), axes(A, 2))
    elseif dims == 2
        return IndexMap(A, size(A), axes(A, 1), reverse(axes(A, 2)))
    else
        throw(ArgumentError("invalid dims argument to reverse, should be 1 or 2, got $dims"))
    end
end
_reverse(A, ::Colon) = IndexMap(A, size(A), map(reverse, axes(A))...)
_reverse(A, dims::NTuple{1,Integer}) = _reverse(A, first(dims))
function _reverse(A, dims::NTuple{M,Integer}) where {M}
    dimrev = ntuple(k -> k in dims, 2)
    if 2 < M || M != sum(dimrev)
        throw(ArgumentError("invalid dimensions $dims in reverse!"))
    end
    ax = ntuple(k -> dimrev[k] ? reverse(axes(A, k)) : axes(A, k), 2)
    return IndexMap(A, size(A), ax...)
end

function check_index(index::AbstractVector{Int}, dimA::Int, dimB::Int)
    length(index) != dimA && throw(ArgumentError("invalid length of index vector"))
    minimum(index) <= 0 && throw(ArgumentError("minimal index is below 1"))
    maximum(index) > dimB && throw(ArgumentError(
        "maximal index $(maximum(index)) exceeds dimension $dimB"
        ))
    # _isvalidstep(index) || throw(ArgumentError("non-monotone index set"))
    nothing
end

# _isvalidstep(index::AbstractRange) = step(index) > 0
# _isvalidstep(index::AbstractVector) = all(diff(index) .> 0)

Base.size(A::IndexMap) = A.dims

LinearAlgebra.issymmetric(A::IndexMap) = issymmetric(A.lmap) && (A.dims[1] == A.dims[2]) && (A.rows == A.cols)
LinearAlgebra.ishermitian(A::IndexMap) = ishermitian(A.lmap) && (A.dims[1] == A.dims[2]) && (A.rows == A.cols)

Base.:(==)(A::IndexMap, B::IndexMap) = (eltype(A) == eltype(B)) && (A.lmap == B.lmap) &&
    (A.dims == B.dims) && (A.rows == B.rows) && (A.cols == B.cols)

LinearAlgebra.adjoint(A::IndexMap) = IndexMap(adjoint(A.lmap), reverse(A.dims), A.cols, A.rows)
LinearAlgebra.transpose(A::IndexMap) = IndexMap(transpose(A.lmap), reverse(A.dims), A.cols, A.rows)

for (In, Out) in ((AbstractVector, AbstractVecOrMat), (AbstractMatrix, AbstractMatrix))
    @eval function _unsafe_mul!(y::$Out, A::IndexMap, x::$In)
        fill!(y, zero(eltype(y)))
        _unsafe_mul!(selectdim(y, 1, A.rows), A.lmap, selectdim(x, 1, A.cols))
        return y
    end
    @eval function _unsafe_mul!(y::$Out, A::IndexMap, x::$In, alpha::Number, beta::Number)
        LinearAlgebra._rmul_or_fill!(y, beta)
        _unsafe_mul!(selectdim(y, 1, A.rows), A.lmap, selectdim(x, 1, A.cols), alpha, !iszero(beta))
        return y
    end
end
