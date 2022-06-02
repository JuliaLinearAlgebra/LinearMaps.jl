struct BlockMap{T,
                As<:LinearMapTupleOrVector,
                Rs<:Tuple{Vararg{Int}}} <: LinearMap{T}
    maps::As
    rows::Rs
    rowranges::Vector{UnitRange{Int}}
    colranges::Vector{UnitRange{Int}}
    function BlockMap{T,As,Rs}(maps::As, rows::Rs) where
                {T, As<:LinearMapTupleOrVector, Rs<:Tuple{Vararg{Int}}}
        for TA in Base.Generator(eltype, maps)
            promote_type(T, TA) == T ||
                error("eltype $TA cannot be promoted to $T in BlockMap constructor")
        end
        rowranges, colranges = rowcolranges(maps, rows)
        return new{T, As, Rs}(maps, rows, rowranges, colranges)
    end
end

BlockMap{T}(maps::As, rows::Rs) where {T, As<:LinearMapTupleOrVector, Rs} =
    BlockMap{T, As, Rs}(maps, rows)
BlockMap(maps::As, rows::Rs) where {As<:LinearMapTupleOrVector, Rs} =
    BlockMap{promote_type(map(eltype, maps)...), As, Rs}(maps, rows)

MulStyle(A::BlockMap) = MulStyle(A.maps...)

function _getranges(maps, dim, inds=1:length(maps))
    ends = map(i -> size(maps[i], dim)::Int, inds)
    cumsum!(ends, ends)
    starts = vcat(1, 1 .+ @views ends[1:end-1])
    return UnitRange.(starts, ends)
end

"""
    rowcolranges(maps, rows)

Determines the range of rows for each block row and the range of columns for each
map in `maps`, according to its position in a virtual matrix representation of the
block linear map obtained from `hvcat(rows, maps...)`.
"""
function rowcolranges(maps, rows)
    # find indices of the row-wise first maps
    firstmapinds = vcat(1, Base.front(rows)...)
    cumsum!(firstmapinds, firstmapinds)
    # compute rowranges from first dimension of the row-wise first maps
    rowranges = _getranges(maps, 1, firstmapinds)

    # compute ranges from second dimension as if all in one row
    temp = _getranges(maps, 2)
    # introduce "line breaks"
    colranges = map(1:length(maps)) do i
        # for each map find the index of the respective row-wise first map
        # something-trick just to assure the compiler that the index is an Int
        @inbounds firstmapind = firstmapinds[something(findlast(<=(i), firstmapinds), 1)]
        # shift ranges by the first col-index of the row-wise first map
        return @inbounds temp[i] .- first(temp[firstmapind]) .+ 1
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

    # find first non-UniformScaling to detect number of rows
    j = findfirst(A -> !isa(A, UniformScaling), As)
    # this should not happen, function should only be called with at least one LinearMap
    @assert !isnothing(j)
    @inbounds nrows = size(As[j], 1)::Int
    
    return BlockMap{T}(promote_to_lmaps(ntuple(_ -> nrows, Val(nbc)), 1, 1, As...), (nbc,))
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

    # find first non-UniformScaling to detect number of rows
    j = findfirst(A -> !isa(A, UniformScaling), As)
    # this should not happen, function should only be called with at least one LinearMap
    @assert !isnothing(j)
    @inbounds ncols = size(As[j], 2)::Int

    rows = ntuple(_ -> 1, Val(nbr))
    return BlockMap{T}(promote_to_lmaps(ntuple(_ -> ncols, Val(nbr)), 1, 2, As...), rows)
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
                na = size(As[j+k], 1)::Int
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
            nci += isa(As[j+k], UniformScaling) ? n[j+k] : size(As[j+k], 2)::Int
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
    (eltype(A) == eltype(B) && all(A.maps .== B.maps) && all(A.rows .== B.rows))

############
# multiplication helper functions
############

function _blockmul!(y, A, x, α, β)
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

function ___blockmul!(y, A, x::Number, α, β, _)
    maps, rows, yinds, xinds = A.maps, A.rows, A.rowranges, A.colranges
    mapind = 0
    if iszero(α)
        iszero(β) && return fill!(y, zero(eltype(y)))
        isone(β) && return y
        return rmul!(y, β)
    elseif iszero(β)
        s = x*α
        for (row, yi) in zip(rows, yinds)
            mapind += 1
            _unsafe_mul!(view(y, yi, xinds[mapind]), maps[mapind], s)
            for _ in 2:row
                mapind += 1
                _unsafe_mul!(view(y, yi, xinds[mapind]), maps[mapind], s)
            end
        end
    else
        for (row, yi) in zip(rows, yinds)
            mapind += 1
            _unsafe_mul!(view(y, yi, xinds[mapind]), maps[mapind], x, α, β)
            for _ in 2:row
                mapind += 1
                _unsafe_mul!(view(y, yi, xinds[mapind]), maps[mapind], x, α, β)
            end
        end
    end
    return y
end
function ___blockmul!(y, A, x::AbstractVecOrMat, α, β, ::Nothing)
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
function ___blockmul!(y, A, x::AbstractVecOrMat, α, β, z)
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

function _transblockmul!(y, A, x::Number, α, β, transform)
    maps, rows, xinds, yinds = A.maps, A.rows, A.rowranges, A.colranges
    if iszero(α)
        iszero(β) && return fill!(y, zero(eltype(y)))
        isone(β) && return y
        return rmul!(y, β)
    elseif iszero(β)
        s = x*α
        # first block row (rowind = 1) of A, meaning first block column of A', fill all of y
        for rowind in 1:first(rows)
            _unsafe_mul!(view(y, yinds[rowind], first(xinds)), transform(maps[rowind]), s)
        end
        mapind = first(rows)
        # subsequent block rows of A (block columns of A')
        @inbounds for i in 2:length(rows), _ in 1:rows[i]
            mapind +=1
            _unsafe_mul!(view(y, yinds[mapind], xinds[i]), transform(maps[mapind]), s)
        end
    else
        # first block row (rowind = 1) of A, meaning first block column of A', fill all of y
        for rowind in 1:first(rows)
            ytile = view(y, yinds[rowind], first(xinds))
            _unsafe_mul!(ytile, transform(maps[rowind]), x, α, β)
        end
        mapind = first(rows)
        # subsequent block rows of A (block columns of A'),
        # add results to corresponding parts of y
        # TODO: think about multithreading
        @inbounds for i in 2:length(rows), _ in 1:rows[i]
            mapind +=1
            ytile = view(y, yinds[mapind], xinds[i])
            _unsafe_mul!(ytile, transform(maps[mapind]), x, α, true)
        end
    end
    return y
end
function _transblockmul!(y, A, x, α, β, transform)
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
        @inbounds for i in 2:length(rows)
            xrow = selectdim(x, 1, xinds[i])
            for _ in 1:rows[i]
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
# multiplication with a scalar
############

function _unsafe_mul!(Y::AbstractMatrix, A::BlockMap, s::Number, α::Number=true, β::Number=false)
    require_one_based_indexing(Y, s)
    return _blockmul!(Y, A, s, α, β)
end
for (MT, transform) in ((:TransposeMap, :transpose), (:AdjointMap, :adjoint))
    @eval begin
        function _unsafe_mul!(Y::AbstractMatrix, wrapA::$MT{<:Any, <:BlockMap}, s::Number, 
                    α::Number=true, β::Number=false)
            require_one_based_indexing(Y)
            return _transblockmul!(Y, wrapA.lmap, s, α, β, $transform)
        end
    end
end

############
# BlockDiagonalMap
############
struct BlockDiagonalMap{T, As<:LinearMapTupleOrVector} <: LinearMap{T}
    maps::As
    rowranges::Vector{UnitRange{Int}}
    colranges::Vector{UnitRange{Int}}
    function BlockDiagonalMap{T, As}(maps::As) where {T, As<:LinearMapTupleOrVector}
        for TA in Base.Generator(eltype, maps)
            promote_type(T, TA) == T ||
                error("eltype $TA cannot be promoted to $T in BlockDiagonalMap constructor")
        end
        rowranges = _getranges(maps, 1)
        colranges = _getranges(maps, 2)
        return new{T, As}(maps, rowranges, colranges)
    end
end

BlockDiagonalMap{T}(maps::As) where {T, As<:LinearMapTupleOrVector} =
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

Base.size(A::BlockDiagonalMap) = (last(last(A.rowranges)), last(last(A.colranges)))

MulStyle(A::BlockDiagonalMap) = MulStyle(A.maps...)

LinearAlgebra.issymmetric(A::BlockDiagonalMap) = all(issymmetric, A.maps)
LinearAlgebra.ishermitian(A::BlockDiagonalMap) = all(ishermitian, A.maps)

LinearAlgebra.adjoint(A::BlockDiagonalMap{T}) where {T} =
    BlockDiagonalMap{T}(map(adjoint, A.maps))
LinearAlgebra.transpose(A::BlockDiagonalMap{T}) where {T} =
    BlockDiagonalMap{T}(map(transpose, A.maps))

Base.:(==)(A::BlockDiagonalMap, B::BlockDiagonalMap) =
    (eltype(A) == eltype(B) && all(A.maps .== B.maps))

for (In, Out) in ((AbstractVector, AbstractVecOrMat), (AbstractMatrix, AbstractMatrix), (Number, AbstractMatrix))
    @eval begin
        function _unsafe_mul!(y::$Out, A::BlockDiagonalMap, x::$In)
            require_one_based_indexing(y, x)
            return _blockscaling!(y, A, x)
        end
        function _unsafe_mul!(y::$Out, A::BlockDiagonalMap, x::$In, α::Number, β::Number)
            require_one_based_indexing(y, x)
            return _blockscaling!(y, A, x, α, β)
        end
    end
end

function _blockscaling!(y, A, x::Number)
    maps, yinds, xinds = A.maps, A.rowranges, A.colranges
    fill!(y, zero(eltype(y)))
    # TODO: think about multi-threading here
    @inbounds for (yind, map, xind) in zip(yinds, maps, xinds)
        _unsafe_mul!(view(y, yind, xind), map, x)
    end
    return y
end
function _blockscaling!(y, A, x::Number, α, β)
    maps, yinds, xinds = A.maps, A.rowranges, A.colranges
    LinearAlgebra._rmul_or_fill!(y, β)
    # TODO: think about multi-threading here
    @inbounds for (yind, map, xind) in zip(yinds, maps, xinds)
        _unsafe_mul!(view(y, yind, xind), map, x, α, true)
    end
    return y
end

function _blockscaling!(y, A, x)
    maps, yinds, xinds = A.maps, A.rowranges, A.colranges
    # TODO: think about multi-threading here
    @inbounds for (yind, map, xind) in zip(yinds, maps, xinds)
        _unsafe_mul!(selectdim(y, 1, yind), map, selectdim(x, 1, xind))
    end
    return y
end
function _blockscaling!(y, A, x, α, β)
    maps, yinds, xinds = A.maps, A.rowranges, A.colranges
    # TODO: think about multi-threading here
    @inbounds for (yind, map, xind) in zip(yinds, maps, xinds)
        _unsafe_mul!(selectdim(y, 1, yind), map, selectdim(x, 1, xind), α, β)
    end
    return y
end
