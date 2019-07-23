##############
# BlockMaps #
##############

abstract type AbstractBlockMap{T,As} <: LinearMap{T} end

struct HBlockMap{T,As<:Tuple{Vararg{LinearMap}}} <: AbstractBlockMap{T,As}
    maps::As
    block_sizes::Tuple{Vector{Int},Vector{Int}}
    function HBlockMap(maps::R, block_sizes::Tuple{Vector{Int},Vector{Int}}) where {T, R<:Tuple{Vararg{LinearMap{T}}}}
        new{T,R}(maps, block_sizes)
    end
end

struct VBlockMap{T,As<:Tuple{Vararg{LinearMap}}} <: AbstractBlockMap{T,As}
    maps::As
    block_sizes::Tuple{Vector{Int},Vector{Int}}
    function VBlockMap(maps::R, block_sizes::Tuple{Vector{Int},Vector{Int}}) where {T, R<:Tuple{Vararg{LinearMap{T}}}}
        new{T,R}(maps, block_sizes)
    end
end

HBlockMap(maps::Tuple{Vararg{LinearMap{T}}}) where {T} = HBlockMap(maps, sizes_from_maps_h(maps))
VBlockMap(maps::Tuple{Vararg{LinearMap{T}}}) where {T} = VBlockMap(maps, sizes_from_maps_v(maps))

function sizes_from_maps_h(maps::Tuple{Vararg{LinearMap}})::Tuple{Vector{Int},Vector{Int}} where {T}
	m = size(maps[1], 1)
	for map in maps
		m == size(map, 1) || throw(DimensionMismatch())
	end
	return [1, m+1], cumsum!(zeros(Int, length(maps)+1), [1, map(m -> size(m, 2), maps)...,])
end
function sizes_from_maps_v(maps::Tuple{Vararg{LinearMap}})::Tuple{Vector{Int},Vector{Int}} where {T}
	n = size(maps[1], 2)
	for map in maps
		n == size(map, 2) || throw(DimensionMismatch())
	end
	return cumsum!(zeros(Int, length(maps)+1), [1, map(m -> size(m, 1), maps)...,]), [1, n+1]
end

@inline Base.size(A::AbstractBlockMap) = (A.block_sizes[1][end]-1, A.block_sizes[2][end]-1)

############
# hcat
############

function Base.hcat(As::Union{LinearMap,UniformScaling}...)
	T = promote_type(map(eltype, As)...)
	for A in As
		if !(A isa UniformScaling)
			eltype(A) == T || throw(ArgumentError("eltype mismatch in hcat of linear maps"))
		end
	end

	nrows = 0
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
			# size(A, 1) == nrows || throw(DimensionMismatch("hcat of LinearMaps"))
			return A
		end
	end
	return hcat(maps...)
end
Base.hcat(A::LinearMap{T}, As::LinearMap{T}...) where {T} = hcat(A, hcat(As...))
Base.hcat(A::LinearMap{T}, B::LinearMap{T}) where {T} = HBlockMap(tuple(A, B))
Base.hcat(A::LinearMap{T}, B::HBlockMap{T}) where {T} = HBlockMap(tuple(A, B.maps...))
Base.hcat(A::HBlockMap{T}, B::HBlockMap{T}) where {T} = HBlockMap(tuple(A.maps..., B.maps...))

function Base.vcat(As::Union{LinearMap,UniformScaling}...)
	T = promote_type(map(eltype, As)...)
	for A in As
		if !(A isa UniformScaling)
			eltype(A) == T || throw(ArgumentError("eltype type mismatch in vcat of linear maps"))
		end
	end

	ncols = 0
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
			# size(A, 2) == ncols || throw(DimensionMismatch("vcat of LinearMaps"))
			return A
		end
	end
	return vcat(maps...)
end
Base.vcat(A::LinearMap{T}, As::LinearMap{T}...) where {T} = vcat(A, vcat(As...))
Base.vcat(A::LinearMap{T}, B::LinearMap{T}) where {T} = VBlockMap(tuple(A, B))
Base.vcat(A::LinearMap{T}, B::VBlockMap{T}) where {T} = VBlockMap(tuple(A, B.maps...))
Base.vcat(A::VBlockMap{T}, B::VBlockMap{T}) where {T} = VBlockMap(tuple(A.maps..., B.maps...))

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
		return hcat(hmaps...)
    end

    return vcat(A...)
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

function A_mul_B!(y::AbstractVector, A::HBlockMap, x::AbstractVector)
    M, N = size(A)
    (length(x) == N && length(y) == M) || throw(DimensionMismatch())
    _, xinds = A.block_sizes
    # fill y with results of first block
	A_mul_B!(y, A.maps[1], @views x[xinds[1]:xinds[2]-1])
	# add to y results of the following block columns
    @inbounds @views for j in 2:length(A.maps)
        mul!(y, A.maps[j], x[xinds[j]:xinds[j+1]-1], 1, 1)
    end
    return y
end
function A_mul_B!(y::AbstractVector, A::VBlockMap, x::AbstractVector)
	M, N = size(A)
    (length(x) == N && length(y) == M) || throw(DimensionMismatch())
    yinds, _ = A.block_sizes
    # fill parts of y according to block sizes
	@inbounds @views for j in 1:length(A.maps)
        A_mul_B!(y[yinds[j]:yinds[j+1]-1], A.maps[j], x)
    end
    return y
end

At_mul_B!(y::AbstractVector, A::AbstractBlockMap, x::AbstractVector) = A_mul_B!(y, transpose(A), x)

Ac_mul_B!(y::AbstractVector, A::AbstractBlockMap, x::AbstractVector) = A_mul_B!(y, adjoint(A), x)

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
