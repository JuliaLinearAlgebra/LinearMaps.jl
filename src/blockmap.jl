##############
# BlockSizes #
##############

abstract type AbstractBlockSizes end

# Keeps track of the (cumulative) sizes of all the blocks in the `BlockMap`.
struct BlockSizes{VT<:Tuple{Vector{Int},Vector{Int}}} <: AbstractBlockSizes
    cumul_sizes::VT
    # Takes a tuple of sizes, accumulates them and create a `BlockSizes`
    BlockSizes{VT}() where {VT<:Tuple{Vector{Int},Vector{Int}}} = new{VT}()
    BlockSizes{VT}(cs::VT) where {VT<:Tuple{Vector{Int},Vector{Int}}} = new{VT}(cs)
end

BlockSizes(cs::VT) where {VT<:Tuple{Vector{Int},Vector{Int}}} = BlockSizes{typeof(cs)}(cs)
# BlockSizes(cs::Tuple{Vector{Int},Vector{Int}}) = BlockSizes(cs)

function BlockSizes(size1::AbstractVector{Int}, size2::AbstractVector{Int})
    cumul_sizes = (_cumul_vec(size1), _cumul_vec(size2))
    return BlockSizes(cumul_sizes)
end

Base.:(==)(a::BlockSizes, b::BlockSizes) = cumulsizes(a) == cumulsizes(b)

Base.dataids(b::BlockSizes) = _splatmap(dataids, b.cumul_sizes)
# _splatmap taken from Base:
_splatmap(f, ::Tuple{}) = ()
_splatmap(f, t::Tuple) = (f(t[1])..., _splatmap(f, tail(t))...)

function _cumul_vec(v::AbstractVector{T}) where {T}
    v_cumul = similar(v, length(v) + 1)
    z = one(T)
    v_cumul[1] = z
    for i in eachindex(v)
        z += v[i]
        v_cumul[i+1] = z
    end
    return v_cumul
end

Base.@propagate_inbounds cumulsizes(block_sizes::BlockSizes) = block_sizes.cumul_sizes
Base.@propagate_inbounds cumulsizes(block_sizes::AbstractBlockSizes, i) = cumulsizes(block_sizes)[i]
Base.@propagate_inbounds cumulsizes(block_sizes::AbstractBlockSizes, i, j) = cumulsizes(block_sizes,i)[j]

Base.@propagate_inbounds blocksize(block_sizes::AbstractBlockSizes, i, j) =
    cumulsizes(block_sizes, i, j+1) - cumulsizes(block_sizes, i, j)

function blocksize(block_sizes::AbstractBlockSizes, i::Tuple{Integer, Integer})
    return blocksize(block_sizes, 1, i[1]), blocksize(block_sizes, 1, i[1])
end

# Gives the total sizes
function Base.size(block_sizes::AbstractBlockSizes)
    @inbounds return cumulsizes(block_sizes, 1)[end] - 1, cumulsizes(block_sizes, 2)[end] - 1
end

function Base.show(io::IO, block_sizes::AbstractBlockSizes)
    print(io, " × ", diff(cumulsizes(block_sizes, 2)))
end

@inline function searchlinear(vec::AbstractVector, a)
    l = length(vec)
    @inbounds for i in 1:l
        vec[i] > a && return i - 1
    end
    return l
end

_find_block(bs::AbstractVector, i::Integer) = length(bs) > 10 ? last(searchsorted(bs, i)) : searchlinear(bs, i)

@inline function _find_block(block_sizes::AbstractBlockSizes, dim::Integer, i::Integer)
    bs = cumulsizes(block_sizes, dim)
    block = _find_block(bs, i)
    @inbounds cum_size = cumulsizes(block_sizes, dim, block) - 1
    return block, i - cum_size
end

nblocks(block_sizes::AbstractBlockSizes) = nblocks(block_sizes, 1), nblocks(block_sizes, 2)

# @inline nblocks(block_array::AbstractArray) = nblocks(blocksizes(block_array))

@inline Base.@propagate_inbounds nblocks(block_sizes::AbstractBlockSizes, i::Integer) =
    length(cumulsizes(block_sizes, i)) - 1

function nblocks(block_sizes::AbstractBlockSizes, i::Integer, j::Integer)
    b = nblocks(block_sizes)
    return b[i], b[j]
end

function Base.copy(block_sizes::BlockSizes)
    return BlockSizes(copy(cumulsizes(block_sizes, 1)), copy(cumulsizes(block_sizes, 2)))
end

@inline function globalrange(block_sizes::AbstractBlockSizes, block_index::NTuple{2, Integer})
    @inbounds v = (cumulsizes(block_sizes, 1, block_index[1]):cumulsizes(block_sizes, 1, block_index[1] + 1) - 1,
                   cumulsizes(block_sizes, 2, block_index[2]):cumulsizes(block_sizes, 2, block_index[2] + 1) - 1)
    return v
end

"""
    blocksize(A, inds)

Returns a tuple containing the size of the block at block index `inds`.
"""

@inline blocksize(block_array::AbstractArray, i::Integer...) =
    blocksize(blocksizes(block_array), i...)

@inline blocksize(block_array::AbstractMatrix{T}, i::Tuple{Integer, Integer}) where {T} =
    blocksize(blocksizes(block_array), i)

cumulsizes(A::AbstractArray) = cumulsizes(blocksizes(A))
@inline cumulsizes(A::AbstractArray, i) = cumulsizes(blocksizes(A), i)
@inline cumulsizes(A::AbstractArray, i, j) = cumulsizes(blocksizes(A), i, j)

##############
# BlockMaps #
##############

abstract type AbstractBlockMap{T,As,Bs} <: LinearMap{T} end

struct HBlockMap{T,As<:Tuple{Vararg{LinearMap}},BS<:AbstractBlockSizes} <: AbstractBlockMap{T,As,BS}
    maps::As
    block_sizes::BS
    global function _HBlockMap(maps::R, block_sizes::BS) where {T, R<:Tuple{Vararg{LinearMap{T}}}, BS<:AbstractBlockSizes}
        new{T, R, BS}(maps, block_sizes)
    end
end

struct VBlockMap{T,As<:Tuple{Vararg{LinearMap}},BS<:AbstractBlockSizes} <: AbstractBlockMap{T,As,BS}
    maps::As
    block_sizes::BS
    global function _VBlockMap(maps::R, block_sizes::BS) where {T, R<:Tuple{Vararg{LinearMap{T}}}, BS<:AbstractBlockSizes}
        new{T, R, BS}(maps, block_sizes)
    end
end

# BlockMap(maps::Tuple{Vararg{LinearMap}}, rows::Integer) = isone(rows) ? _HBlockMap(maps, sizes_from_maps(maps, rows))
HBlockMap(maps::Tuple{Vararg{LinearMap{T}}}) where {T} = _HBlockMap(maps, sizes_from_maps(maps, 1))
VBlockMap(maps::Tuple{Vararg{LinearMap{T}}}) where {T} = _VBlockMap(maps, sizes_from_maps(maps, length(maps)))

function sizes_from_maps(maps::Tuple{Vararg{LinearMap}}, rows::Integer)
	N = length(maps)
	if N == 0
        return zeros.(Int, size(maps))
    end
	cols, rem = divrem(N, rows)
    if rem != 0
        error("Cannot construct a BlockMap with $rows block rows from $N linear maps")
    end
    fullsizes::Array{Tuple{Int,Int}} = reshape([map(size, maps)...,], rows, cols)
    block_sizes = ntuple(2) do i
        [s[i] for s in view(fullsizes, ntuple(j -> j == i ? (:) : 1, 2)...)]
    end
    checksizes(fullsizes, block_sizes)
    return BlockSizes(block_sizes...)
end

getsizes(block_sizes, block_index) = getindex.(block_sizes, block_index)

function checksizes(fullsizes::Matrix{Tuple{Int,Int}}, block_sizes::Tuple{Vector{Int},Vector{Int}})
    Base.@nloops 2 i fullsizes begin
        block_index = Base.@ntuple 2 i
        if fullsizes[block_index...] != getsizes(block_sizes, block_index)
			error("size(blocks[", strip(repr(block_index), ['(', ')']),
                      "]) (= ", fullsizes[block_index...],
                      ") is incompatible with expected size: ",
                      getsizes(block_sizes, block_index))
        end
    end
    return fullsizes
end

@inline Base.size(A::AbstractBlockMap) = size(blocksizes(A))

"""
    blocksizes(A)

Returns a subtype of `AbstractBlockSizes` that contains information about the
block sizes of `A`.
"""
@inline blocksizes(A::AbstractBlockMap) = A.block_sizes

@inline function blockcheckbounds(A::AbstractBlockMap{T}, i::Integer, j::Integer) where {T}
    n = nblocks(A)
	if i <= 0 || i > n[1]
		return throw(BoundsError(A, (i, j)))
	end
	if j <= 0 || j > n[2]
		return throw(BoundsError(A, (i, j)))
	end
    return nothing
end

@inline nblocks(A::AbstractBlockMap) = nblocks(blocksizes(A))

nblocks(A::AbstractBlockMap, i::Integer) = nblocks(A)[i]

nblocks(A::AbstractBlockMap, i::Integer, j::Integer) = nblocks(blocksizes(A), i, j)

@inline function getblock(A::AbstractBlockMap, i::Integer, j::Integer)
    @boundscheck blockcheckbounds(A, i, j)
	m, n = nblocks(A)
    A.maps[(j-1)*m + i]
end

############
# hcat
############

function Base.hcat(As::Union{LinearMap,UniformScaling}...)
    nrows = 0
	T = promote_type(map(eltype, As)...)
	for A in As
		if !(A isa UniformScaling)
			eltype(A) == T || throw(ArgumentError("eltype type mismatch in hcat of linear maps"))
		end
	end

	# find first non-UniformScaling to detect number of rows
	for A in As
		if !(A isa UniformScaling)
			nrows = size(A, 1)
			break
		end
	end
	nrows == 0 && throw(ArgumentError("hcat of only UniformScaling-like objects cannot determine the linear map size"))

	maps = map(As) do A
		if A isa UniformScaling
			return UniformScalingMap(convert(T, A.λ), nrows)
		else
			size(A, 1) == nrows || throw(DimensionMismatch("hcat of LinearMaps"))
			return A
		end
	end
	return hcat(maps...)
end
function Base.hcat(A::LinearMap{T}, As::LinearMap{T}...) where {T}
	for Ai in As
		size(A, 1) == size(Ai, 1) || throw(DimensionMismatch("hcat of LinearMaps"))
	end
	hcat(A, hcat(As...))
end
function Base.hcat(A::LinearMap{T}, B::LinearMap{T}) where {T}
	size(A, 1) == size(B, 1) || throw(DimensionMismatch("hcat of LinearMaps"))
	HBlockMap(tuple(A, B))
end
Base.hcat(A::LinearMap{T}, B::HBlockMap{T}) where {T} = HBlockMap(tuple(A, B.maps...))

function Base.vcat(As::Union{LinearMap,UniformScaling}...)
    ncols = 0
	T = promote_type(map(eltype, As)...)
	for A in As
		if !(A isa UniformScaling)
			eltype(A) == T || throw(ArgumentError("eltype type mismatch in vcat of linear maps"))
		end
	end

	# find first non-UniformScaling to detect number of columns
	for A in As
		if !(A isa UniformScaling)
			ncols = size(A, 2)
			break
		end
	end
	ncols == 0 && throw(ArgumentError("hcat of only UniformScaling-like objects cannot determine the linear map size"))

	maps = map(As) do A
		if A isa UniformScaling
			return UniformScalingMap(convert(T, A.λ), ncols)
		else
			size(A, 2) == ncols || throw(DimensionMismatch("vcat of LinearMaps"))
			return A
		end
	end
	return vcat(maps...)
end
function Base.vcat(A::LinearMap{T}, As::LinearMap{T}...) where {T}
	for Ai in As
		size(A, 2) == size(Ai, 2) || throw(DimensionMismatch("vcat of LinearMaps"))
	end
	return vcat(A, vcat(As...))
end
function Base.vcat(A::LinearMap{T}, B::LinearMap{T}) where {T}
	size(A, 2) == size(B, 2) || throw(DimensionMismatch("vcat of LinearMaps"))
	return VBlockMap(tuple(A, B))
end
Base.vcat(A::LinearMap{T}, B::VBlockMap{T}) where {T} = VBlockMap(tuple(A, B.maps...))

function Base.hvcat(rows::Tuple{Vararg{Int}}, As::Union{LinearMap,UniformScaling}...)
	T = promote_type(map(eltype, As)...)
    nr = length(rows)
    sum(rows) == length(As) || throw(ArgumentError("mismatch between row sizes and number of arguments"))
    n = fill(-1, length(As))
    needcols = false # whether we also need to infer some sizes from the column count
    j = 0
    for i in 1:nr # infer UniformScaling sizes from row counts, if possible:
        ni = -1 # number of rows in this block-row, -1 indicates unknown
        for k in 1:rows[i]
            if !isa(As[j+k], UniformScaling)
                na = size(As[j+k], 1)
                ni >= 0 && ni != na &&
                    throw(DimensionMismatch("mismatch in number of rows"))
                ni = na
            end
        end
        if ni >= 0
            for k = 1:rows[i]
                n[j+k] = ni
            end
        else # row consisted only of UniformScaling objects
            needcols = true
        end
        j += rows[i]
    end
    if needcols # some sizes still unknown, try to infer from column count
        nc = -1
        j = 0
        for i in 1:nr
            nci = 0
            rows[i] > 0 && n[j+1] == -1 && (j += rows[i]; continue)
            for k = 1:rows[i]
                nci += isa(As[j+k], UniformScaling) ? n[j+k] : size(As[j+k], 2)
            end
            nc >= 0 && nc != nci && throw(DimensionMismatch("mismatch in number of columns"))
            nc = nci
            j += rows[i]
        end
        nc == -1 && throw(ArgumentError("sizes of UniformScalings could not be inferred"))
        j = 0
        for i in 1:nr
            if rows[i] > 0 && n[j+1] == -1 # this row consists entirely of UniformScalings
                nci = nc ÷ rows[i]
                nci * rows[i] != nc && throw(DimensionMismatch("indivisible UniformScaling sizes"))
                for k = 1:rows[i]
                    n[j+k] = nci
                end
            end
            j += rows[i]
        end
    end
	maps = ntuple(length(As)) do i
		if As[i] isa UniformScaling
			return UniformScalingMap(convert(T, As[i].λ), n[i])
		else
			# size(A, 2) == n[i] || throw(DimensionMismatch("vcat of LinearMaps"))
			return As[i] # size check is going to be performed in h/v/cat below
		end
	end

	return hvcat(rows, maps)
end

function Base.hvcat(rows::Tuple{Vararg{Int}}, As::Tuple{Vararg{LinearMap{T}}}) where {T}
	nr = length(rows)
	j = 0
	A::NTuple{nr,HBlockMap{T}} = ntuple(nr) do i
		hmaps = As[j+1:j+rows[i]]
		j += rows[i]
		return hcat(hmaps...)::HBlockMap{T,typeof(hmaps)}
    end

    return vcat(A...)::VBlockMap{T,typeof(A)}
end

############
# basic methods
############

# function LinearAlgebra.issymmetric(A::BlockMap)
#     m, n = nblocks(A)
#     m == n || return false
#     for i in 1:m, j in i:m
#         if (i == j && !issymmetric(getblock(A, i, i)))
#             return false
#         elseif getblock(A, i, j) != transpose(getblock(A, j, i))
#             return false
#         end
#     end
#     return true
# end
#
# LinearAlgebra.ishermitian(A::BlockMap{<:Real}) = issymmetric(A)
# function LinearAlgebra.ishermitian(A::BlockMap)
#     m, n = nblocks(A)
#     m == n || return false
#     for i in 1:m, j in i:m
#         if (i == j && !ishermitian(getblock(A, i, i)))
#             return false
#         elseif getblock(A, i, j) != adjoint(getblock(A, j, i))
#             return false
#         end
#     end
#     return true
# end
# TODO, currently falls back on the generic `false`
# LinearAlgebra.isposdef(A::BlockMap)

############
# comparison of BlockMap objects, sufficient but not necessary
############

Base.:(==)(A::AbstractBlockMap, B::AbstractBlockMap) = (eltype(A) == eltype(B) && A.maps == B.maps && A.block_indices == B.block_indices)

# special transposition behavior

LinearAlgebra.transpose(A::VBlockMap) = HBlockMap(map(transpose, A.maps))
LinearAlgebra.transpose(A::HBlockMap) = VBlockMap(map(transpose, A.maps))
LinearAlgebra.adjoint(A::VBlockMap)   = HBlockMap(map(adjoint, A.maps))
LinearAlgebra.adjoint(A::HBlockMap)   = VBlockMap(map(adjoint, A.maps))

############
# multiplication with vectors
############

function A_mul_B!(y::AbstractVector, A::AbstractBlockMap, x::AbstractVector)
    M, N = size(A)
    (length(x) == N && length(y) == M) || throw(DimensionMismatch())
    yinds, xinds = cumulsizes(A.block_sizes)
    # fill y with results of first block column
    @inbounds for i in 1:nblocks(A, 1)
        @views A_mul_B!(y[yinds[i]:yinds[i+1]-1], getblock(A, i, 1), x[xinds[1]:xinds[2]-1])
    end
    # add to y results of the following block columns
    @inbounds for j in 2:nblocks(A, 2), i in 1:nblocks(A, 1)
        @views mul!(y[yinds[i]:yinds[i+1]-1], getblock(A, i, j), x[xinds[j]:xinds[j+1]-1], 1, 1)
    end
    return y
end
function At_mul_B!(y::AbstractVector, A::AbstractBlockMap, x::AbstractVector)
    M, N = size(A)
    (length(x) == N && length(y) == M) || throw(DimensionMismatch())
    xinds, yinds = cumulsizes(A.block_sizes)
    # fill y with results of first block column
    @inbounds for i in 1:nblocks(A, 2)
        @views A_mul_B!(y[yinds[i]:yinds[i+1]-1], transpose(getblock(A, 1, i)), x[xinds[1]:xinds[2]-1])
    end
    # add to y results of the following block columns
    @inbounds for j in 2:nblocks(A, 1), i in 1:nblocks(A, 2)
        @views mul!(y[yinds[i]:yinds[i+1]-1], transpose(getblock(A, j, i)), x[xinds[j]:xinds[j+1]-1], 1, 1)
    end
    return y
end
function Ac_mul_B!(y::AbstractVector, A::AbstractBlockMap, x::AbstractVector)
	M, N = size(A)
    (length(x) == N && length(y) == M) || throw(DimensionMismatch())
    xinds, yinds = cumulsizes(A.block_sizes)
    # fill y with results of first block column
    @inbounds for i in 1:nblocks(A, 2)
        @views A_mul_B!(y[yinds[i]:yinds[i+1]-1], adjoint(getblock(A, 1, i)), x[xinds[1]:xinds[2]-1])
    end
    # add to y results of the following block columns
    @inbounds for j in 2:nblocks(A, 1), i in 1:nblocks(A, 2)
        @views mul!(y[yinds[i]:yinds[i+1]-1], adjoint(getblock(A, j, i)), x[xinds[j]:xinds[j+1]-1], 1, 1)
    end
    return y
end

############
# show methods
############

block2string(b, s) = string(join(map(string, b), '×'), "-blocked ", Base.dims2string(s))
Base.summary(a::AbstractBlockMap) = string(block2string(nblocks(a), size(a)), " ", typeof(a))
# _show_typeof(io, a) = show(io, typeof(a))
function Base.summary(io::IO, a::AbstractBlockMap)
    print(io, block2string(nblocks(a), size(a)))
    print(io, ' ')
    _show_typeof(io, a)
end
function _show_typeof(io::IO, a::AbstractBlockMap{T}) where {T}
    Base.show_type_name(io, typeof(a).name)
    print(io, '{')
    show(io, T)
    print(io, '}')
end
