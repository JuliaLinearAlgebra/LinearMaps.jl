# module GetIndex

# using ..LinearMaps: LinearMap, AdjointMap, TransposeMap, FillMap, LinearCombination,
#     ScaledMap, UniformScalingMap, WrappedMap

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
_getindex(A::LinearMap, I::AbstractVector{<:Integer}) = [_getindex(A, i) for i in I]
_getindex(A::LinearMap, ::Base.Slice) = vec(Matrix(A))

########################
# Cartesian indexing
########################
_getindex(A::LinearMap, i::Integer, j::Integer) =
    @inbounds _getindex(A, Base.Slice(axes(A)[1]), j)[i]
function _getindex(A::LinearMap, i::Integer, J::AbstractVector{<:Integer})
    try
        return @inbounds (basevec(A, i)'A)[J]
    catch
        x = zeros(eltype(A), size(A, 2))
        y = similar(x, eltype(A), size(A, 1))
        r = similar(x, eltype(A), length(J))
        @inbounds for (ind, j) in enumerate(J)
            x[j] = one(eltype(A))
            _unsafe_mul!(y, A, x)
            r[ind] = y[i]
            x[j] = zero(eltype(A))
        end
        return r
    end
end
function _getindex(A::LinearMap, i::Integer, J::Base.Slice)
    try
        return vec(basevec(A, i)'A)
    catch
        return vec(_getindex(A, i:i, J))
    end
end
_getindex(A::LinearMap, I::AbstractVector{<:Integer}, j::Integer) =
    @inbounds _getindex(A, Base.Slice(axes(A)[1]), j)[I] # = A[:,j][I] w/o bounds check
_getindex(A::LinearMap, ::Base.Slice, j::Integer) = A*basevec(A, j)
function _getindex(A::LinearMap, Is::Vararg{AbstractVector{<:Integer},2})
    shape = Base.index_shape(Is...)
    dest = zeros(eltype(A), shape)
    I, J = Is
    for (ind, ij) in zip(eachindex(dest), Iterators.product(I, J))
        i, j = ij
        dest[ind] = _getindex(A, i, j)
    end
    return dest
end
function _getindex(A::LinearMap, I::AbstractVector{<:Integer}, ::Base.Slice)
    x = zeros(eltype(A), size(A, 2))
    y = similar(x, eltype(A), size(A, 1))
    r = similar(x, eltype(A), (length(I), size(A, 2)))
    @inbounds @views for j in axes(A)[2]
        x[j] = one(eltype(A))
        _unsafe_mul!(y, A, x)
        r[:,j] .= y[I]
        x[j] = zero(eltype(A))
    end
    return r
end
function _getindex(A::LinearMap, ::Base.Slice, J::AbstractVector{<:Integer})
    x = zeros(eltype(A), size(A, 2))
    y = similar(x, eltype(A), (size(A, 1), length(J)))
    @inbounds for (i, j) in enumerate(J)
        x[j] = one(eltype(A))
        _unsafe_mul!(selectdim(y, 2, i), A, x)
        x[j] = zero(eltype(A))
    end
    return y
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
function basevec(A, i::Integer)
    x = zeros(eltype(A), size(A, 2))
    @inbounds x[i] = one(eltype(A))
    return x
end

# nogetindex_error() = error("indexing not allowed for LinearMaps; consider setting `LinearMaps.allowgetindex = true`")

# end # module
