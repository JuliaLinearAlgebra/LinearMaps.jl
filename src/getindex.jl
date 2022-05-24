# module GetIndex

# using ..LinearMaps: LinearMap, AdjointMap, TransposeMap, FillMap, LinearCombination,
#     ScaledMap, UniformScalingMap, WrappedMap

const Indexer = AbstractVector{<:Integer}

Base.IndexStyle(::LinearMap) = IndexCartesian()
# required in Base.to_indices for [:]-indexing
Base.eachindex(::IndexLinear, A::LinearMap) = Base.OneTo(length(A))
Base.lastindex(A::LinearMap) = last(eachindex(IndexLinear(), A))
Base.firstindex(A::LinearMap) = first(eachindex(IndexLinear(), A))

function Base.checkbounds(A::LinearMap, i, j)
    Base.checkbounds_indices(Bool, axes(A), (i, j)) || throw(BoundsError(A, (i, j)))
    nothing
end
# Linear indexing is explicitly allowed when there is only one (non-cartesian) index
function Base.checkbounds(A::LinearMap, i)
    Base.checkindex(Bool, Base.OneTo(length(A)), i) || throw(BoundsError(A, i))
    nothing
end
# checkbounds in indexing via CartesianIndex
Base.checkbounds(A::LinearMap, i::Union{CartesianIndex{2}, AbstractVecOrMat{CartesianIndex{2}}}) =
    Base.checkbounds_indices(Bool, axes(A), (i,))
Base.checkbounds(A::LinearMap, I::AbstractArray{Bool,2}) = axes(A) == axes(I)

# main entry point
function Base.getindex(A::LinearMap, I...)
    # TODO: introduce some sort of switch?
    @boundscheck checkbounds(A, I...)
    _getindex(A, Base.to_indices(A, I)...)
end
# quick pass forward
Base.@propagate_inbounds Base.getindex(A::ScaledMap, I...) = A.λ .* A.lmap[I...]
Base.@propagate_inbounds Base.getindex(A::WrappedMap, I...) = A.lmap[I...]
Base.@propagate_inbounds Base.getindex(A::WrappedMap, i::Integer) = A.lmap[i]
Base.@propagate_inbounds Base.getindex(A::WrappedMap, i::Integer, j::Integer) = A.lmap[i,j]

########################
# linear indexing
########################
function _getindex(A::LinearMap, i::Integer)
    i1, i2 = Base._ind2sub(axes(A), i)
    return _getindex(A, i1, i2)
end
_getindex(A::LinearMap, I::Indexer) = [_getindex(A, i) for i in I]
_getindex(A::LinearMap, ::Base.Slice) = vec(Matrix(A))
_getindex(A::LinearMap, I::Vector{CartesianIndex{2}}) = [(@inbounds A[i]) for i in I]

########################
# Cartesian indexing
########################
_getindex(A::LinearMap, i::Union{Integer,Indexer}, j::Integer) = (@inbounds (A*basevec(A, 2, j))[i])
_getindex(A::LinearMap, ::Base.Slice, j::Integer) = A*basevec(A, 2, j)
function _getindex(A::LinearMap, i::Integer, J::Indexer)
    # try
        # requires adjoint action to be defined
        return @inbounds (basevec(A, 1, i)'A)[J]
    # catch
    #     return _fillbycols!(zeros(eltype(A), Base.index_shape(i, J)), A, i, J)
    # end
end
function _getindex(A::LinearMap, i::Integer, J::Base.Slice)
    # try
        # requires adjoint action to be defined
        return vec(basevec(A, 1, i)'A)
    # catch
    #     return _fillbycols!(zeros(eltype(A), Base.index_shape(i, J)), A, i, J)
    # end
end
function _getindex(A::LinearMap, I::Indexer, J::Indexer)
    dest = zeros(eltype(A), Base.index_shape(I, J))
    if length(I) <= length(J)
        # try
            # requires adjoint action to be defined
            _fillbyrows!(dest, A, I, J)
        # catch
        #     _fillbycols!(dest, A, I, J)
        # end
    else
        _fillbycols!(dest, A, I, J)
    end
    return dest
end
_getindex(A::LinearMap, ::Base.Slice, ::Base.Slice) = Matrix(A)

# specialized methods
_getindex(A::FillMap, ::Integer, ::Integer) = A.λ
_getindex(A::LinearCombination, i::Integer, j::Integer) =
    sum(a -> (@inbounds A.maps[a][i, j]), eachindex(A.maps))
_getindex(A::AdjointMap, i::Integer, j::Integer) = @inbounds adjoint(A.lmap[j, i])
_getindex(A::TransposeMap, i::Integer, j::Integer) = @inbounds transpose(A.lmap[j, i])
_getindex(A::UniformScalingMap, i::Integer, j::Integer) = ifelse(i == j, A.λ, zero(eltype(A)))

# helpers
function basevec(A, dim, i::Integer)
    x = zeros(eltype(A), size(A, dim))
    @inbounds x[i] = one(eltype(A))
    return x
end

function _fillbyrows!(dest, A, I, J)
    x = zeros(eltype(A), size(A, 1))
    temp = similar(x, eltype(A), size(A, 2))
    @views @inbounds for (di, i) in zip(eachcol(dest), I)
        x[i] = one(eltype(A))
        _unsafe_mul!(temp, A', x)
        di .= adjoint.(temp[J])
        x[i] = zero(eltype(A))
    end
    return dest
end
function _fillbycols!(dest, A, i, J)
    x = zeros(eltype(A), size(A, 2))
    temp = similar(x, eltype(A), size(A, 1))
    @views @inbounds for (ind, j) in enumerate(J)
        x[j] = one(eltype(A))
        _unsafe_mul!(temp, A, x)
        _copycol!(dest, ind, temp, i)
        x[j] = zero(eltype(A))
    end
    return dest
end
function _fillbycols!(dest, A, ::Base.Slice, J)
    x = zeros(eltype(A), size(A, 2))
    @views @inbounds for (ind, j) in enumerate(J)
        x[j] = one(eltype(A))
        _unsafe_mul!(selectdim(dest, 2, ind), A, x)
        x[j] = zero(eltype(A))
    end
    return dest
end

@inline _copycol!(dest, ind, temp, i::Integer) = (@inbounds dest[ind] = temp[i])
@inline _copycol!(dest, ind, temp, I::Indexer) =
    (@views @inbounds dest[:,ind] .= temp[I])

# diagonal indexing
function LinearAlgebra.diagind(A::LinearMap, k::Integer=0)
    require_one_based_indexing(A)
    diagind(size(A,1), size(A,2), k)
end

LinearAlgebra.diag(A::LinearMap, k::Integer=0) = A[diagind(A,k)]

# logical indexing
Base.getindex(A::LinearMap, mask::AbstractVecOrMat{Bool}) = A[findall(mask)]
Base.getindex(A::LinearMap, i, mask::AbstractVector{Bool}) = A[i, findall(mask)]
Base.getindex(A::LinearMap, mask::AbstractVector{Bool}, j) = A[findall(mask), j]
Base.getindex(A::LinearMap, im::AbstractVector{Bool}, jm::AbstractVector{Bool}) =
    A[findall(im), findall(jm)]
# disambiguation
for typ in (:WrappedMap, :ScaledMap)
    @eval begin
        Base.getindex(A::$typ, mask::AbstractVecOrMat{Bool}) = A[findall(mask)]
        Base.getindex(A::$typ, i, mask::AbstractVector{Bool}) = A[i, findall(mask)]
        Base.getindex(A::$typ, mask::AbstractVector{Bool}, j) = A[findall(mask), j]
        Base.getindex(A::$typ, im::AbstractVector{Bool}, jm::AbstractVector{Bool}) =
            A[findall(im), findall(jm)]       
    end
end

# nogetindex_error() = error("indexing not allowed for LinearMaps; consider setting `LinearMaps.allowgetindex = true`")

# end # module
