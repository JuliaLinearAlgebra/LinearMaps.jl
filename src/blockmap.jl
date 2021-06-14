struct BlockMap{T,
                As<:LinearMapTuple,
                Rs<:Tuple{Vararg{Int}},
                Rranges<:Tuple{Vararg{UnitRange{Int}}},
                Cranges<:Tuple{Vararg{UnitRange{Int}}}} <: LinearMap{T}
    maps::As
    rows::Rs
    rowranges::Rranges
    colranges::Cranges
    function BlockMap{T,As,Rs}(maps::As, rows::Rs) where
                {T, As<:LinearMapTuple, Rs<:Tuple{Vararg{Int}}}
        for TA in Base.Generator(eltype, maps)
            promote_type(T, TA) == T ||
                error("eltype $TA cannot be promoted to $T in BlockMap constructor")
        end
        rowranges, colranges = rowcolranges(maps, rows)
        Rranges, Cranges = typeof(rowranges), typeof(colranges)
        return new{T, As, Rs, Rranges, Cranges}(maps, rows, rowranges, colranges)
    end
end

BlockMap{T}(maps::As, rows::Rs) where {T, As<:LinearMapTuple, Rs} =
    BlockMap{T, As, Rs}(maps, rows)

MulStyle(A::BlockMap) = MulStyle(A.maps...)

"""
    rowcolranges(maps, rows)

Determines the range of rows for each block row and the range of columns for each
map in `maps`, according to its position in a virtual matrix representation of the
block linear map obtained from `hvcat(rows, maps...)`.
"""
function rowcolranges(maps, rows)
    rowranges = ntuple(n->1:0, Val(length(rows)))
    colranges = ntuple(n->1:0, Val(length(maps)))
    mapind = 0
    rowstart = 1
    for (i, row) in enumerate(rows)
        mapind += 1
        rowend = rowstart + Int(size(maps[mapind], 1))::Int - 1
        rowranges = Base.setindex(rowranges, rowstart:rowend, i)
        colstart = 1
        colend = Int(size(maps[mapind], 2))::Int
        colranges = Base.setindex(colranges, colstart:colend, mapind)
        for colind in 2:row
            mapind += 1
            colstart = colend + 1
            colend += Int(size(maps[mapind], 2))::Int
            colranges = Base.setindex(colranges, colstart:colend, mapind)
        end
        rowstart = rowend + 1
    end
    return rowranges, colranges
end

Base.size(A::BlockMap) = (last(last(A.rowranges)), last(last(A.colranges)))

############
# hcat
############
"""
    hcat(As::Union{LinearMap,UniformScaling,AbstractVecOrMat}...)::BlockMap

Construct a (lazy) representation of the horizontal concatenation of the arguments.
All arguments are promoted to `LinearMap`s automatically.

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
function Base.hcat(As::Union{LinearMap, UniformScaling, AbstractVecOrMat}...)
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
    @assert nrows != -1
    # this should not happen, function should only be called with at least one LinearMap
    return BlockMap{T}(promote_to_lmaps(ntuple(i->nrows, nbc), 1, 1, As...), (nbc,))
end

############
# vcat
############
"""
    vcat(As::Union{LinearMap,UniformScaling,AbstractVecOrMat}...)::BlockMap

Construct a (lazy) representation of the vertical concatenation of the arguments.
All arguments are promoted to `LinearMap`s automatically.

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
function Base.vcat(As::Union{LinearMap,UniformScaling,AbstractVecOrMat}...)
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
    @assert ncols != -1
    # this should not happen, function should only be called with at least one LinearMap
    rows = ntuple(i->1, nbr)
    return BlockMap{T}(promote_to_lmaps(ntuple(i->ncols, nbr), 1, 2, As...), rows)
end

############
# hvcat
############
"""
    hvcat(rows::Tuple{Vararg{Int}}, As::Union{LinearMap,UniformScaling,AbstractVecOrMat}...)::BlockMap

Construct a (lazy) representation of the horizontal-vertical concatenation of the arguments.
The first argument specifies the number of arguments to concatenate in each block row.
All arguments are promoted to `LinearMap`s automatically.

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

function Base.hvcat(rows::Tuple{Vararg{Int}},
                    As::Union{LinearMap, UniformScaling, AbstractVecOrMat}...)
    nr = length(rows)
    T = promote_type(map(eltype, As)...)
    sum(rows) == length(As) ||
        throw(ArgumentError("mismatch between row sizes and number of arguments"))
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
            n[j .+ (1:rows[i])] .= ni
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

function check_dim(A, dim, n)
    n == size(A, dim) || throw(DimensionMismatch("Expected $n, got $(size(A, dim))"))
    return nothing
end

promote_to_lmaps_(n::Int, dim, A::AbstractVecOrMat) = (check_dim(A, dim, n); LinearMap(A))
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

symindex(i, N) = ((k, l) = divrem(i-1, N); return k + l * N + 1)

# the following rules are sufficient but not necessary
function LinearAlgebra.issymmetric(A::BlockMap)
    isblocksquare(A) || return false
    N = length(A.rows)
    maps = A.maps
    for i in 1:N*N
        isym = symindex(i, N)
        if (i == isym && !issymmetric(maps[i]))
            return false
        elseif (maps[i] != transpose(maps[isym]))
            return false
        end
    end
    return true
end

function LinearAlgebra.ishermitian(A::BlockMap)
    isblocksquare(A) || return false
    N = length(A.rows)
    maps = A.maps
    for i in 1:N*N
        isym = symindex(i, N)
        if (i == isym && !ishermitian(maps[i]))
            return false
        elseif (maps[i] != adjoint(maps[isym]))
            return false
        end
    end
    return true
end

############
# comparison of BlockMap objects, sufficient but not necessary
############

Base.:(==)(A::BlockMap, B::BlockMap) =
    (eltype(A) == eltype(B) && A.maps == B.maps && A.rows == B.rows)

############
# multiplication helper functions
############

function _blockmul!(y, A::BlockMap, x, α, β)
    if iszero(α)
        iszero(β) && return fill!(y, zero(eltype(y)))
        isone(β) && return y
        return rmul!(y, β)
    end
    return __blockmul!(MulStyle(A), y, A, x, α, β)
end

# provide one global intermediate storage vector if necessary
__blockmul!(::FiveArg, y, A, x, α, β)  = ___blockmul!(y, A, x, α, β, nothing)
__blockmul!(::ThreeArg, y, A, x, α, β) = ___blockmul!(y, A, x, α, β, similar(y))

function ___blockmul!(y, A, x, α, β, ::Nothing)
    maps, rows, yinds, xinds = A.maps, A.rows, A.rowranges, A.colranges
    mapind = 0
    for (row, yi) in zip(rows, yinds)
        yrow = selectdim(y, 1, yi)
        mapind += 1
        _unsafe_mul!(yrow, maps[mapind], selectdim(x, 1, xinds[mapind]), α, β)
        for _ in 2:row
            mapind += 1
            _unsafe_mul!(yrow, maps[mapind], selectdim(x, 1, xinds[mapind]), α, true)
        end
    end
    return y
end
function ___blockmul!(y, A, x, α, β, z)
    maps, rows, yinds, xinds = A.maps, A.rows, A.rowranges, A.colranges
    mapind = 0
    for (row, yi) in zip(rows, yinds)
        yrow = selectdim(y, 1, yi)
        zrow = selectdim(z, 1, yi)
        mapind += 1
        xrow = selectdim(x, 1, xinds[mapind])
        if MulStyle(maps[mapind]) === ThreeArg() && !iszero(β)
            !isone(β) && rmul!(yrow, β)
            muladd!(ThreeArg(), yrow, maps[mapind], xrow, α, zrow)
        else
            _unsafe_mul!(yrow, maps[mapind], xrow, α, β)
        end
        for _ in 2:row
            mapind +=1
            xrow = selectdim(x, 1, xinds[mapind])
            muladd!(MulStyle(maps[mapind]), yrow, maps[mapind], xrow, α, zrow)
        end
    end
    return y
end

function _transblockmul!(y, A::BlockMap, x, α, β, transform)
    maps, rows, xinds, yinds = A.maps, A.rows, A.rowranges, A.colranges
    if iszero(α)
        iszero(β) && return fill!(y, zero(eltype(y)))
        isone(β) && return y
        return rmul!(y, β)
    else
        # first block row (rowind = 1) of A, meaning first block column of A', fill all of y
        xrow = selectdim(x, 1, first(xinds))
        for rowind in 1:first(rows)
            yrow = selectdim(y, 1, yinds[rowind])
            _unsafe_mul!(yrow, transform(maps[rowind]), xrow, α, β)
        end
        mapind = first(rows)
        # subsequent block rows of A (block columns of A'),
        # add results to corresponding parts of y
        # TODO: think about multithreading
        for (row, xi) in zip(Base.tail(rows), Base.tail(xinds))
            xrow = selectdim(x, 1, xi)
            for _ in 1:row
                mapind +=1
                yrow = selectdim(y, 1, yinds[mapind])
                _unsafe_mul!(yrow, transform(maps[mapind]), xrow, α, true)
            end
        end
    end
    return y
end

############
# multiplication with vectors & matrices
############

for (In, Out) in ((AbstractVector, AbstractVecOrMat), (AbstractMatrix, AbstractMatrix))
    @eval begin
        function _unsafe_mul!(y::$Out, A::BlockMap, x::$In)
            require_one_based_indexing(y, x)
            return _blockmul!(y, A, x, true, false)
        end
        function _unsafe_mul!(y::$Out, A::BlockMap, x::$In, α::Number, β::Number)
            require_one_based_indexing(y, x)
            return _blockmul!(y, A, x, α, β)
        end
    end

    for (MT, transform) in ((:TransposeMap, :transpose), (:AdjointMap, :adjoint))
        @eval begin
            MapType = $MT{<:Any, <:BlockMap}
            function _unsafe_mul!(y::$Out, wrapA::MapType, x::$In)
                require_one_based_indexing(y, x)
                return _transblockmul!(y, wrapA.lmap, x, true, false, $transform)
            end
            function _unsafe_mul!(y::$Out, wrapA::MapType, x::$In, α::Number, β::Number)
                require_one_based_indexing(y, x)
                return _transblockmul!(y, wrapA.lmap, x, α, β, $transform)
            end
        end
    end
end

############
# BlockDiagonalMap
############
struct BlockDiagonalMap{T,
                        As<:LinearMapTuple,
                        Ranges<:Tuple{Vararg{UnitRange{Int}}}} <: LinearMap{T}
    maps::As
    rowranges::Ranges
    colranges::Ranges
    function BlockDiagonalMap{T, As}(maps::As) where {T, As<:LinearMapTuple}
        for TA in Base.Generator(eltype, maps)
            promote_type(T, TA) == T ||
                error("eltype $TA cannot be promoted to $T in BlockDiagonalMap constructor")
        end
        # row ranges
        inds = vcat(1, size.(maps, 1)...)
        cumsum!(inds, inds)
        rowranges = ntuple(i -> inds[i]:inds[i+1]-1, Val(length(maps)))
        # column ranges
        inds[2:end] .= size.(maps, 2)
        cumsum!(inds, inds)
        colranges = ntuple(i -> inds[i]:inds[i+1]-1, Val(length(maps)))
        return new{T, As, typeof(rowranges)}(maps, rowranges, colranges)
    end
end

BlockDiagonalMap{T}(maps::As) where {T, As<:LinearMapTuple} =
    BlockDiagonalMap{T,As}(maps)
BlockDiagonalMap(maps::LinearMap...) =
    BlockDiagonalMap{promote_type(map(eltype, maps)...)}(maps)

# since the below methods are more specific than the Base method,
# they would redefine Base/SparseArrays behavior
for k in 1:8 # is 8 sufficient?
    Is = ntuple(n->:($(Symbol(:A, n))::AbstractVecOrMat), Val(k-1))
    # yields (:A1, :A2, :A3, ..., :A(k-1))
    L = :($(Symbol(:A, k))::LinearMap)
    # yields :Ak
    mapargs = ntuple(n ->:($(Symbol(:A, n))), Val(k-1))
    # yields (:LinearMap(A1), :LinearMap(A2), ..., :LinearMap(A(k-1)))

    @eval begin
        function SparseArrays.blockdiag($(Is...), $L, As::MapOrVecOrMat...)
            return BlockDiagonalMap(convert_to_lmaps($(mapargs...))...,
                                    $(Symbol(:A, k)),
                                    convert_to_lmaps(As...)...)
        end

        function Base.cat($(Is...), $L, As::MapOrVecOrMat...; dims::Dims{2})
            if dims == (1,2)
                return BlockDiagonalMap(convert_to_lmaps($(mapargs...))...,
                                        $(Symbol(:A, k)),
                                        convert_to_lmaps(As...)...)
            else
                throw(ArgumentError("dims keyword in cat of LinearMaps must be (1,2)"))
            end
        end
    end
end

"""
    blockdiag(As::Union{LinearMap,AbstractVecOrMat}...)::BlockDiagonalMap

Construct a (lazy) representation of the diagonal concatenation of the arguments.
To avoid fallback to the generic `SparseArrays.blockdiag`, there must be a `LinearMap`
object among the first 8 arguments.
"""
SparseArrays.blockdiag

"""
    cat(As::Union{LinearMap,AbstractVecOrMat}...; dims=(1,2))::BlockDiagonalMap

Construct a (lazy) representation of the diagonal concatenation of the arguments.
To avoid fallback to the generic `Base.cat`, there must be a `LinearMap`
object among the first 8 arguments.
"""
Base.cat

Base.size(A::BlockDiagonalMap) = (last(A.rowranges[end]), last(A.colranges[end]))

MulStyle(A::BlockDiagonalMap) = MulStyle(A.maps...)

LinearAlgebra.issymmetric(A::BlockDiagonalMap) = all(issymmetric, A.maps)
LinearAlgebra.ishermitian(A::BlockDiagonalMap) = all(ishermitian, A.maps)

LinearAlgebra.adjoint(A::BlockDiagonalMap{T}) where {T} =
    BlockDiagonalMap{T}(map(adjoint, A.maps))
LinearAlgebra.transpose(A::BlockDiagonalMap{T}) where {T} =
    BlockDiagonalMap{T}(map(transpose, A.maps))

Base.:(==)(A::BlockDiagonalMap, B::BlockDiagonalMap) =
    (eltype(A) == eltype(B) && A.maps == B.maps)

for (In, Out) in ((AbstractVector, AbstractVecOrMat), (AbstractMatrix, AbstractMatrix))
    @eval begin
        function _unsafe_mul!(y::$Out, A::BlockDiagonalMap, x::$In)
            require_one_based_indexing(y, x)
            return _blockscaling!(y, A, x, true, false)
        end
        function _unsafe_mul!(y::$Out, A::BlockDiagonalMap, x::$In, α::Number, β::Number)
            require_one_based_indexing(y, x)
            return _blockscaling!(y, A, x, α, β)
        end
    end
end

function _blockscaling!(y, A::BlockDiagonalMap, x, α, β)
    maps, yinds, xinds = A.maps, A.rowranges, A.colranges
    # TODO: think about multi-threading here
    @views for i in eachindex(yinds, maps, xinds)
        _unsafe_mul!(selectdim(y, 1, yinds[i]), maps[i], selectdim(x, 1, xinds[i]), α, β)
    end
    return y
end
