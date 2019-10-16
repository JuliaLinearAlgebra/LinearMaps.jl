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

MulStyle(A::BlockMap) = MulStyle(A.maps...)

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

Base.size(A::BlockMap) = (last(A.rowranges[end]), last(A.colranges[end]))

############
# concatenation
############

for k in 1:8 # is 8 sufficient?
    Is = ntuple(n->:($(Symbol(:A,n))::UniformScaling), Val(k-1))
    L = :($(Symbol(:A,k))::LinearMap)
    args = ntuple(n->Symbol(:A,n), Val(k))

    @eval Base.hcat($(Is...), $L, As::Union{LinearMap,UniformScaling}...) = _hcat($(args...), As...)
    @eval Base.vcat($(Is...), $L, As::Union{LinearMap,UniformScaling}...) = _vcat($(args...), As...)
    @eval Base.hvcat(rows::Tuple{Vararg{Int}}, $(Is...), $L, As::Union{LinearMap,UniformScaling}...) = _hvcat(rows, $(args...), As...)
end

############
# hcat
############
"""
    hcat(As::Union{LinearMap,UniformScaling}...)

Construct a `BlockMap <: LinearMap` object, a (lazy) representation of the
horizontal concatenation of the arguments. `UniformScaling` objects are promoted
to `LinearMap` automatically. To avoid fallback to the generic [`Base.hcat`](@ref),
there must be a `LinearMap` object among the first 8 arguments.

# Examples
```jldoctest; setup=(using LinearMaps)
julia> CS = LinearMap{Int}(cumsum, 3)::LinearMaps.FunctionMap;

julia> L = [CS LinearMap(ones(Int, 3, 3))]::LinearMaps.BlockMap;

julia> L * ones(Int, 6)
3-element Array{Int64,1}:
 4
 5
 6
```
"""
Base.hcat

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
    nrows == -1 && throw(ArgumentError("hcat of only UniformScaling objects cannot determine the linear map size"))
    return BlockMap{T}(promote_to_lmaps(fill(nrows, nbc), 1, 1, As...), (nbc,))
end

############
# vcat
############
"""
    vcat(As::Union{LinearMap,UniformScaling}...)

Construct a `BlockMap <: LinearMap` object, a (lazy) representation of the
vertical concatenation of the arguments. `UniformScaling` objects are promoted
to `LinearMap` automatically. To avoid fallback to the generic [`Base.vcat`](@ref),
there must be a `LinearMap` object among the first 8 arguments.

# Examples
```jldoctest; setup=(using LinearMaps)
julia> CS = LinearMap{Int}(cumsum, 3)::LinearMaps.FunctionMap;

julia> L = [CS; LinearMap(ones(Int, 3, 3))]::LinearMaps.BlockMap;

julia> L * ones(Int, 3)
6-element Array{Int64,1}:
 1
 2
 3
 3
 3
 3
```
"""
Base.vcat

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
    ncols == -1 && throw(ArgumentError("vcat of only UniformScaling objects cannot determine the linear map size"))

    return BlockMap{T}(promote_to_lmaps(fill(ncols, nbr), 1, 2, As...), ntuple(i->1, nbr))
end

############
# hvcat
############
"""
    hvcat(rows::Tuple{Vararg{Int}}, As::Union{LinearMap,UniformScaling}...)

Construct a `BlockMap <: LinearMap` object, a (lazy) representation of the
horizontal-vertical concatenation of the arguments. The first argument specifies
the number of arguments to concatenate in each block row. `UniformScaling` objects
are promoted to `LinearMap` automatically. To avoid fallback to the generic
[`Base.hvcat`](@ref), there must be a `LinearMap` object among the first 8 arguments.

# Examples
```jldoctest; setup=(using LinearMaps)
julia> CS = LinearMap{Int}(cumsum, 3)::LinearMaps.FunctionMap;

julia> L = [CS CS; CS CS]::LinearMaps.BlockMap;

julia> L.rows
(2, 2)

julia> L * ones(Int, 6)
6-element Array{Int64,1}:
 2
 4
 6
 2
 4
 6
```
"""
Base.hvcat

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
    return all(==(N), rows)
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

Base.@propagate_inbounds function A_mul_B!(y::AbstractVector, A::BlockMap, x::AbstractVector)
    require_one_based_indexing(y, x)
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

Base.@propagate_inbounds function At_mul_B!(y::AbstractVector, A::BlockMap, x::AbstractVector)
    require_one_based_indexing(y, x)
    m, n = size(A)
    @boundscheck (n == length(y) && m == length(x)) || throw(DimensionMismatch("At_mul_B!"))
    maps, rows, xinds, yinds = A.maps, A.rows, A.rowranges, A.colranges
    mapind = 0
    # first block row (rowind = 1) of A, meaning first block column of A', fill all of y
    @views @inbounds begin
        xcol = x[xinds[1]]
        for colind in 1:rows[1]
            mapind +=1
            A_mul_B!(y[yinds[mapind]], transpose(maps[mapind]), xcol)
        end
        # subsequent block rows of A, add results to corresponding parts of y
        for rowind in 2:length(rows)
            xcol = x[xinds[rowind]]
            for colind in 1:rows[rowind]
                mapind +=1
                mul!(y[yinds[mapind]], transpose(maps[mapind]), xcol, true, true)
            end
        end
    end
    return y
end

Base.@propagate_inbounds function Ac_mul_B!(y::AbstractVector, A::BlockMap, x::AbstractVector)
    require_one_based_indexing(y, x)
    m, n = size(A)
    @boundscheck (n == length(y) && m == length(x)) || throw(DimensionMismatch("At_mul_B!"))
    maps, rows, xinds, yinds = A.maps, A.rows, A.rowranges, A.colranges
    mapind = 0
    # first block row (rowind = 1) of A, fill all of y
    @views @inbounds begin
        xcol = x[xinds[1]]
        for colind in 1:rows[1]
            mapind +=1
            A_mul_B!(y[yinds[mapind]], adjoint(maps[mapind]), xcol)
        end
        # subsequent block rows of A, add results to corresponding parts of y
        for rowind in 2:length(rows)
            xcol = x[xinds[rowind]]
            for colind in 1:rows[rowind]
                mapind +=1
                mul!(y[yinds[mapind]], adjoint(maps[mapind]), xcol, true, true)
            end
        end
    end
    return y
end

Base.@propagate_inbounds function LinearAlgebra.mul!(y::AbstractVector, A::BlockMap, x::AbstractVector, α::Number=true, β::Number=false)
    require_one_based_indexing(y, x)
    m, n = size(A)
    @boundscheck (m == length(y) && n == length(x)) || throw(DimensionMismatch("A_mul_B!"))
    iszero(β) && (A_mul_B!(y, A, x); rmul!(y, α); return y)
    !isone(β) && rmul!(y, β)
    maps, rows, yinds, xinds = A.maps, A.rows, A.rowranges, A.colranges
    mapind = 0
    @views @inbounds for rowind in 1:length(rows)
        yrow = y[yinds[rowind]]
        for colind in 1:rows[rowind]
            mapind +=1
            mul!(yrow, maps[mapind], x[xinds[mapind]], α, true)
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

struct BlockDiagonalMap{T,As<:Tuple{Vararg{LinearMap}}} <: LinearMap{T}
    maps::As
    rowranges::Vector{UnitRange{Int}}
    colranges::Vector{UnitRange{Int}}
    function BlockDiagonalMap{T,As}(maps::As) where {T, As<:Tuple{Vararg{LinearMap}}}
        for A in maps
            promote_type(T, eltype(A)) == T || throw(InexactError())
        end
        # row ranges
        inds = vcat(1, size.(maps, 1)...)
        cumsum!(inds, inds)
        rowranges = map(i -> inds[i]:inds[i+1]-1, 1:length(maps))
        # column ranges
        inds[2:end] .= size.(maps, 2)
        cumsum!(inds, inds)
        colranges = map(i -> inds[i]:inds[i+1]-1, 1:length(maps))
        return new{T,As}(maps, rowranges, colranges)
    end
end

BlockDiagonalMap{T}(maps::As) where {T,As<:Tuple{Vararg{LinearMap}}} =
    BlockDiagonalMap{T,As}(maps)
BlockDiagonalMap(maps::LinearMap...) =
    BlockDiagonalMap{promote_type(map(eltype, maps)...)}(maps)

################ needs to be removed after kron is merged ################
convert_to_lmaps_(A::AbstractMatrix) = LinearMap(A)
convert_to_lmaps_(A::LinearMap) = A
convert_to_lmaps() = ()
convert_to_lmaps(A) = (convert_to_lmaps_(A),)
@inline convert_to_lmaps(A, B, Cs...) =
    (convert_to_lmaps_(A), convert_to_lmaps_(B), convert_to_lmaps(Cs...)...)
################# end of what needs to be removed ################

for k in 1:8 # is 8 sufficient?
    Is = ntuple(n->:($(Symbol(:A,n))::AbstractMatrix), Val(k-1))
    # yields (:A1, :A2, :A3, ..., :A(k-1))
    L = :($(Symbol(:A,k))::LinearMap)
    # yields :Ak
    mapargs = ntuple(n -> :(LinearMap($(Symbol(:A,n)))), Val(k-1))
    # yields (:LinearMap(A1), :LinearMap(A2), ..., :LinearMap(A(k-1)))

    @eval begin
        SparseArrays.blockdiag($(Is...), $L, As::Union{LinearMap,AbstractMatrix}...) =
            BlockDiagonalMap($(mapargs...), $(Symbol(:A,k)), convert_to_lmaps(As...)...)
        function Base.cat($(Is...), $L, As::Union{LinearMap,AbstractMatrix}...; dims::Dims{2})
            if dims == (1,2)
                return BlockDiagonalMap($(mapargs...), $(Symbol(:A,k)), convert_to_lmaps(As...)...)
            else
                throw(ArgumentError("dims keyword in cat of LinearMaps must be (1,2)"))
            end
        end
    end
end

Base.size(A::BlockDiagonalMap) = (last(A.rowranges[end]), last(A.colranges[end]))

LinearAlgebra.issymmetric(A::BlockDiagonalMap) = all(issymmetric, A.maps)
LinearAlgebra.ishermitian(A::BlockDiagonalMap{<:Real}) = all(issymmetric, A.maps)
LinearAlgebra.ishermitian(A::BlockDiagonalMap) = all(ishermitian, A.maps)

LinearAlgebra.adjoint(A::BlockDiagonalMap{T}) where {T} = BlockDiagonalMap{T}(map(adjoint, A.maps))
LinearAlgebra.transpose(A::BlockDiagonalMap{T}) where {T} = BlockDiagonalMap{T}(map(transpose, A.maps))

Base.:(==)(A::BlockDiagonalMap, B::BlockDiagonalMap) = (eltype(A) == eltype(B) && A.maps == B.maps)

Base.@propagate_inbounds function A_mul_B!(y::AbstractVector, A::BlockDiagonalMap, x::AbstractVector)
    require_one_based_indexing(y, x)
    m, n = size(A)
    @boundscheck (m == length(y) && n == length(x)) || throw(DimensionMismatch("A_mul_B!"))
    maps, yinds, xinds = A.maps, A.rowranges, A.colranges
    @views @inbounds for i in length(maps)
        A_mul_B!(y[yinds[i]], maps[i], x[xinds[i]])
    end
    return y
end

Base.@propagate_inbounds At_mul_B!(y::AbstractVector, A::BlockDiagonalMap, x::AbstractVector) = A_mul_B!(y, transpose(A), x)

Base.@propagate_inbounds Ac_mul_B!(y::AbstractVector, A::BlockDiagonalMap, x::AbstractVector) = A_mul_B!(y, adjoint(A), x)

Base.@propagate_inbounds function LinearAlgebra.mul!(y::AbstractVector, A::BlockDiagonalMap, x::AbstractVector, α::Number=true, β::Number=false)
    require_one_based_indexing(y, x)
    m, n = size(A)
    @boundscheck (m == length(y) && n == length(x)) || throw(DimensionMismatch("A_mul_B!"))
    maps, yinds, xinds = A.maps, A.rowranges, A.colranges
    @views @inbounds for i in 1:length(maps)
        mul!(y[yinds[i]], maps[i], x[xinds[i]], α, β)
    end
    return y
end
