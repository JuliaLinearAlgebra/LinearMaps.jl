struct BlockMap{T,As<:Tuple{Vararg{LinearMap}},Rs<:Tuple{Vararg{Int}}} <: LinearMap{T}
    maps::As
    rows::Rs
    rowranges::Vector{UnitRange{Int}}
    colranges::Vector{UnitRange{Int}}
    function BlockMap{T,R,S}(maps::R, rows::S) where {T, R<:Tuple{Vararg{LinearMap}}, S<:Tuple{Vararg{Int}}}
        for A in maps
            promote_type(T, eltype(A)) == T || throw(InexactError())
        end
        rowranges, colranges = rowcolranges(maps, rows)
        return new{T,R,S}(maps, rows, rowranges, colranges)
    end
end

BlockMap{T}(maps::As, rows::S) where {T,As<:Tuple{Vararg{LinearMap}},S} = BlockMap{T,As,S}(maps, rows)

function check_dim(A::LinearMap, dim, n)
    n == size(A, dim) || throw(DimensionMismatch("Expected $n, got $(size(A, dim))"))
    return nothing
end

"""
    rowcolranges(maps, rows)

Determines the range of rows for each block row and the range of columns for each
map in `maps`, according to its position in a virtual matrix representation of the
block linear map obtained from `hvcat(rows, maps...)`.
"""
function rowcolranges(maps, rows)::Tuple{Vector{UnitRange{Int}},Vector{UnitRange{Int}}}
    rowranges = Vector{UnitRange{Int}}(undef, length(rows))
    colranges = Vector{UnitRange{Int}}(undef, length(maps))
    mapind = 0
    rowstart = 1
    for rowind in 1:length(rows)
        xinds = vcat(1, map(a -> size(a, 2), maps[mapind+1:mapind+rows[rowind]])...)
        cumsum!(xinds, xinds)
        mapind += 1
        rowend = rowstart + size(maps[mapind], 1) - 1
        rowranges[rowind] = rowstart:rowend
        colranges[mapind] = xinds[1]:xinds[2]-1
        for colind in 2:rows[rowind]
            mapind +=1
            colranges[mapind] = xinds[colind]:xinds[colind+1]-1
        end
        rowstart = rowend + 1
    end
    return rowranges, colranges
end

function Base.size(A::BlockMap)
    return A.rowranges[end][end], A.colranges[end][end]
end

############
# hcat
############

for k in 1:8 # is 8 sufficient?
    Is = ntuple(n->:($(Symbol(:A,n))::UniformScaling), Val(k-1))
    L = :($(Symbol(:A,k))::LinearMap)
    args = ntuple(n->Symbol(:A,n), Val(k))

    @eval Base.hcat($(Is...), $L, As::Union{LinearMap,UniformScaling}...) = _hcat($(args...), As...)
    @eval Base.vcat($(Is...), $L, As::Union{LinearMap,UniformScaling}...) = _vcat($(args...), As...)
    @eval Base.hvcat(rows::Tuple{Vararg{Int}}, $(Is...), $L, As::Union{LinearMap,UniformScaling}...) = _hvcat(rows, $(args...), As...)
end

function _hcat(As::Union{LinearMap,UniformScaling}...)
    T = promote_type(map(eltype, As)...)
    nbc = length(As)

    nrows = -1
    # find first non-UniformScaling to detect number of rows
    for A in As
        if !(A isa UniformScaling)
            nrows = size(A, 1)
            break
        end
    end
    nrows == -1 && throw(ArgumentError("hcat of only UniformScaling objects cannot determine the matrix size"))
    return BlockMap{T}(promote_to_lmaps(fill(nrows, nbc), 1, 1, As...), (nbc,))
end

############
# vcat
############

function _vcat(As::Union{LinearMap,UniformScaling}...)
    T = promote_type(map(eltype, As)...)
    nbr = length(As)

    ncols = -1
    # find first non-UniformScaling to detect number of columns
    for A in As
        if !(A isa UniformScaling)
            ncols = size(A, 2)
            break
        end
    end
    ncols == -1 && throw(ArgumentError("vcat of only UniformScaling objects cannot determine the matrix size"))

    return BlockMap{T}(promote_to_lmaps(fill(ncols, nbr), 1, 2, As...), ntuple(i->1, nbr))
end

############
# hvcat
############

function _hvcat(rows::Tuple{Vararg{Int}}, As::Union{LinearMap,UniformScaling}...)
    nr = length(rows)
    T = promote_type(map(eltype, As)...)
    sum(rows) == length(As) || throw(ArgumentError("mismatch between row sizes and number of arguments"))
    n = fill(-1, length(As))
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
        end
        j += rows[i]
    end
    # check for consistent total column number
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
            nci, r = divrem(nc, rows[i])
            r != 0 && throw(DimensionMismatch("indivisible UniformScaling sizes"))
            for k = 1:rows[i]
                n[j+k] = nci
            end
        end
        j += rows[i]
    end

    return BlockMap{T}(promote_to_lmaps(n, 1, 1, As...), rows)
end

promote_to_lmaps_(n::Int, dim, J::UniformScaling) = UniformScalingMap(J.λ, n)
promote_to_lmaps_(n::Int, dim, A::LinearMap) = (check_dim(A, dim, n); A)
promote_to_lmaps(n, k, dim) = ()
promote_to_lmaps(n, k, dim, A) = (promote_to_lmaps_(n[k], dim, A),)
@inline promote_to_lmaps(n, k, dim, A, B, Cs...) =
    (promote_to_lmaps_(n[k], dim, A), promote_to_lmaps_(n[k+1], dim, B), promote_to_lmaps(n, k+2, dim, Cs...)...)

############
# basic methods
############

function isblocksquare(A::BlockMap)
    rows = A.rows
    N = length(rows)
    return all(r -> r == N, rows)
end

# the following rules are sufficient but not necessary
function LinearAlgebra.issymmetric(A::BlockMap)
    isblocksquare(A) || return false
    N = length(A.rows)
    maps = A.maps
    symindex = vec(permutedims(reshape(collect(1:N*N), N, N)))
    for i in 1:N*N
        if (i == symindex[i] && !issymmetric(maps[i]))
            return false
        elseif (maps[i] != transpose(maps[symindex[i]]))
            return false
        end
    end
    return true
end

LinearAlgebra.ishermitian(A::BlockMap{<:Real}) = issymmetric(A)
function LinearAlgebra.ishermitian(A::BlockMap)
    isblocksquare(A) || return false
    N = length(A.rows)
    maps = A.maps
    symindex = vec(permutedims(reshape(collect(1:N*N), N, N)))
    for i in 1:N*N
        if (i == symindex[i] && !ishermitian(maps[i]))
            return false
        elseif (maps[i] != adjoint(maps[symindex[i]]))
            return false
        end
    end
    return true
end

############
# comparison of BlockMap objects, sufficient but not necessary
############

Base.:(==)(A::BlockMap, B::BlockMap) = (eltype(A) == eltype(B) && A.maps == B.maps && A.rows == B.rows)

# special transposition behavior

LinearAlgebra.transpose(A::BlockMap) = TransposeMap(A)
LinearAlgebra.adjoint(A::BlockMap)  = AdjointMap(A)

############
# multiplication with vectors
############

function A_mul_B!(y::AbstractVector, A::BlockMap, x::AbstractVector)
    m, n = size(A)
    @boundscheck (m == length(y) && n == length(x)) || throw(DimensionMismatch("A_mul_B!"))
    maps, rows, yinds, xinds = A.maps, A.rows, A.rowranges, A.colranges
    mapind = 0
    @views @inbounds for rowind in 1:length(rows)
        yrow = y[yinds[rowind]]
        mapind += 1
        A_mul_B!(yrow, maps[mapind], x[xinds[mapind]])
        for colind in 2:rows[rowind]
            mapind +=1
            mul!(yrow, maps[mapind], x[xinds[mapind]], true, true)
        end
    end
    return y
end

function At_mul_B!(y::AbstractVector, A::BlockMap, x::AbstractVector)
    m, n = size(A)
    @boundscheck (n == length(y) && m == length(x)) || throw(DimensionMismatch("At_mul_B!"))
    maps, rows, xinds, yinds = A.maps, A.rows, A.rowranges, A.colranges
    mapind = 0
    # first block row (rowind = 1), fill all of y
    @views @inbounds begin
        xcol = x[xinds[1]]
        for colind in 1:rows[1]
            mapind +=1
            A_mul_B!(y[yinds[colind]], transpose(maps[mapind]), xcol)
        end
        # subsequent block rows, add results to corresponding parts of y
        for rowind in 2:length(rows)
            xcol = x[xinds[rowind]]
            for colind in 1:rows[rowind]
                mapind +=1
                mul!(y[yinds[colind]], transpose(maps[mapind]), xcol, true, true)
            end
        end
    end
    return y
end

function Ac_mul_B!(y::AbstractVector, A::BlockMap, x::AbstractVector)
    m, n = size(A)
    @boundscheck (n == length(y) && m == length(x)) || throw(DimensionMismatch("At_mul_B!"))
    maps, rows, xinds, yinds = A.maps, A.rows, A.rowranges, A.colranges
    mapind = 0
    # first block row (rowind = 1), fill all of y
    @views @inbounds begin
        xcol = x[xinds[1]]
        for colind in 1:rows[1]
            mapind +=1
            A_mul_B!(y[yinds[colind]], adjoint(maps[mapind]), xcol)
        end
        # subsequent block rows, add results to corresponding parts of y
        for rowind in 2:length(rows)
            xcol = x[xinds[rowind]]
            for colind in 1:rows[rowind]
                mapind +=1
                mul!(y[yinds[colind]], adjoint(maps[mapind]), xcol, true, true)
            end
        end
    end
    return y
end

############
# show methods
############

# block2string(b, s) = string(join(map(string, b), '×'), "-blocked ", Base.dims2string(s))
# Base.summary(a::BlockMap) = string(block2string(nblocks(a), size(a)), " ", typeof(a))
# # _show_typeof(io, a) = show(io, typeof(a))
# function Base.summary(io::IO, a::AbstractBlockMap)
#     print(io, block2string(nblocks(a), size(a)))
#     print(io, ' ')
#     _show_typeof(io, a)
# end
# function _show_typeof(io::IO, a::AbstractBlockMap{T}) where {T}
#     Base.show_type_name(io, typeof(a).name)
#     print(io, '{')
#     show(io, T)
#     print(io, '}')
# end
