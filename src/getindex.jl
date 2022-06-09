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
function Base.checkbounds(A::LinearMap, i)
    Base.checkindex(Bool, Base.OneTo(length(A)), i) || throw(BoundsError(A, i))
    nothing
end
# checkbounds in indexing via CartesianIndex
Base.checkbounds(A::LinearMap, i::Union{CartesianIndex{2}, AbstractArray{CartesianIndex{2}}}) =
    Base.checkbounds_indices(Bool, axes(A), (i,)) || throw(BoundsError(A, i))
Base.checkbounds(A::LinearMap, I::AbstractMatrix{Bool}) =
    axes(A) == axes(I) || throw(BoundsError(A, I))

# main entry point
function Base.getindex(A::LinearMap, I...)
    @boundscheck checkbounds(A, I...)
    _getindex(A, Base.to_indices(A, I)...)
end
# quick pass forward
Base.@propagate_inbounds Base.getindex(A::ScaledMap, I...) = A.λ * A.lmap[I...]
Base.@propagate_inbounds Base.getindex(A::WrappedMap, I...) = A.lmap[I...]

########################
# linear indexing
########################
_getindex(A::LinearMap, _) = error("linear indexing of LinearMaps is not supported")
_getindex(A::LinearMap, ::Base.Slice) = vec(Matrix(A))

########################
# Cartesian indexing
########################
_getindex(A::LinearMap, i::Integer, j::Integer) =
    error("scalar indexing of LinearMaps is not supported, consider using A[:,j][i] instead")
_getindex(A::LinearMap, I::Indexer, j::Integer) =
    error("partial vertical slicing of LinearMaps is not supported, consider using A[:,j][I] instead")
_getindex(A::LinearMap, ::Base.Slice, j::Integer) = A*unitvec(A, 2, j)
_getindex(A::LinearMap, i::Integer, J::Indexer) =
    error("partial horizontal slicing of LinearMaps is not supported, consider using A[i,:][J] instead")
function _getindex(A::LinearMap, i::Integer, J::Base.Slice)
    try
        # requires adjoint action to be defined
        return vec(unitvec(A, 1, i)'A)
    catch
        error("horizontal slicing A[$i,:] requires the adjoint of $(typeof(A)) to be defined")
    end
end
_getindex(A::LinearMap, I::Indexer, J::Indexer) =
    error("partial two-dimensional slicing of LinearMaps is not supported, consider using A[:,J][I] or A[I,:][J] instead")
_getindex(A::LinearMap, ::Base.Slice, ::Base.Slice) =
    error("two-dimensional slicing of LinearMaps is not supported, consider using Matrix(A) or convert(Matrix, A)") 
_getindex(A::LinearMap, I::Base.Slice, J::Indexer) = __getindex(A, I, J)
_getindex(A::LinearMap, I::Indexer, J::Base.Slice) = __getindex(A, I, J)
function __getindex(A, I, J)
    dest = zeros(eltype(A), Base.index_shape(I, J))
    # choose whichever requires less map applications
    if length(I) <= length(J)
        try
            # requires adjoint action to be defined
            _fillbyrows!(dest, A, I, J)
        catch
            error("wide slicing A[I,J] with length(I) <= length(J) requires the adjoint of $(typeof(A)) to be defined")
        end
    else
        _fillbycols!(dest, A, I, J)
    end
    return dest
end

# helpers
function unitvec(A, dim, i)
    x = zeros(eltype(A), size(A, dim))
    @inbounds x[i] = one(eltype(A))
    return x
end

function _fillbyrows!(dest, A, I, J)
    x = zeros(eltype(A), size(A, 1))
    temp = similar(x, eltype(A), size(A, 2))
    @views @inbounds for (di, i) in zip(eachrow(dest), I)
        x[i] = one(eltype(A))
        _unsafe_mul!(temp, A', x)
        di .= adjoint.(temp[J])
        x[i] = zero(eltype(A))
    end
    return dest
end
function _fillbycols!(dest, A, I::Indexer, J)
    x = zeros(eltype(A), size(A, 2))
    temp = similar(x, eltype(A), size(A, 1))
    @inbounds for (ind, j) in enumerate(J)
        x[j] = one(eltype(A))
        _unsafe_mul!(temp, A, x)
        dest[:,ind] .= temp[I]
        x[j] = zero(eltype(A))
    end
    return dest
end
function _fillbycols!(dest, A, ::Base.Slice, J)
    x = zeros(eltype(A), size(A, 2))
    @inbounds for (ind, j) in enumerate(J)
        x[j] = one(eltype(A))
        _unsafe_mul!(selectdim(dest, 2, ind), A, x)
        x[j] = zero(eltype(A))
    end
    return dest
end
