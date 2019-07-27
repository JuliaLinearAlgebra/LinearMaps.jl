struct BlockMap{T,As<:Tuple{Vararg{LinearMap}},Rs<:Tuple{Vararg{Int}}} <: LinearMap{T}
    maps::As
    rows::Rs
    function BlockMap{T,R,S}(As::R, rows::S) where {T, R<:Tuple{Vararg{LinearMap}}, S<:Tuple{Vararg{Int}}}
        for A in As
            promote_type(T, eltype(A)) == T || throw(InexactError())
        end
        new{T,R,S}(As, rows)
    end
end

BlockMap{T}(maps::As, rows::S) where {T,As<:Tuple{Vararg{LinearMap}},S} = BlockMap{T,As,S}(maps, rows)

function check_dim(A::LinearMap, dim, n)
    n == size(A, dim) || throw(DimensionMismatch("Expected $n, got $(size(A, dim))"))
    return nothing
end

sumshift(a::Tuple{Vararg{Int}}) = sumshift((1,), a)
sumshift(a::Tuple{Vararg{Int}}, as::Tuple{Vararg{Int}}) = sumshift(tuple(a..., last(a) + first(as)), Base.tail(as))
sumshift(a::Tuple{Vararg{Int}}, b::Tuple{Int}) = tuple(a..., last(a) + first(b))

function firstrowindices(A::BlockMap)
    as, rows = A.maps, A.rows

    firstrows = sumshift(A.rows)
    Afirstcol = ntuple(length(firstrows)-1) do i
        A.maps[firstrows[i]]
    end
    return sumshift(map(a -> size(a, 1), Afirstcol))
end

@inline firstcolindices(As::Tuple{Vararg{LinearMap}}) = sumshift(map(a -> size(a, 2), As))

function Base.size(A::BlockMap)
    as, rows = A.maps, A.rows

    nbr = length(rows)  # number of block rows
    nc = 0
    for i in 1:rows[1]
        nc += size(as[i],2)
    end

    nr = 0
    a = 1
    for i in 1:nbr
        nr += size(as[a],1)
        a += rows[i]
    end
    return nr, nc
end

############
# hcat
############

function Base.hcat(As::Union{LinearMap,UniformScaling}...)
    T = promote_type(map(eltype, As)...)
    nbc = length(As)

    nrows = 0
    # find first non-UniformScaling to detect number of rows
    for A in As
        if !(A isa UniformScaling)
            nrows = size(A, 1)
            break
        end
    end
    nrows == 0 && throw(ArgumentError("hcat of only UniformScaling-like objects cannot determine the linear map size"))

    return BlockMap{T}(promote_to_lmaps(ntuple(i->nrows, nbc), 1, 1, As...), (nbc,))
end

############
# vcat
############

function Base.vcat(As::Union{LinearMap,UniformScaling}...)
    T = promote_type(map(eltype, As)...)
    nbr = length(As)

    ncols = 0
    # find first non-UniformScaling to detect number of columns
    for A in As
        if !(A isa UniformScaling)
            ncols = size(A, 2)
            break
        end
    end
    ncols == 0 && throw(ArgumentError("hcat of only UniformScaling-like objects cannot determine the linear map size"))

    return BlockMap{T}(promote_to_lmaps(ntuple(i->ncols, nbr), 1, 2, As...), ntuple(i->1, nbr))
end

############
# hvcat
############

function Base.hvcat(rows::NTuple{nr,Int}, As::Union{LinearMap,UniformScaling}...) where nr
    T = promote_type(map(eltype, As)...)
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

    return BlockMap{T}(promote_to_lmaps(n, 1, 1, As...), rows)
end

promote_to_lmaps_(n::Int, dim, J::UniformScaling) = UniformScalingMap(J.λ, n)
promote_to_lmaps_(n::Int, dim, A::LinearMap) = (check_dim(A, dim, n); A)
promote_to_lmaps(n, k) = ()
promote_to_lmaps(n, k, dim, A) = (promote_to_lmaps_(n[k], dim, A),)
promote_to_lmaps(n, k, dim, A, B) = (promote_to_lmaps_(n[k], dim, A), promote_to_lmaps_(n[k+1], dim, B))
promote_to_lmaps(n, k, dim, A, B, C) =
    (promote_to_lmaps_(n[k], dim, A), promote_to_lmaps_(n[k+1], dim, B), promote_to_lmaps_(n[k+2], dim, C))
promote_to_lmaps(n, k, dim, A, B, Cs...) =
    (promote_to_lmaps_(n[k], dim, A), promote_to_lmaps_(n[k+1], dim, B), promote_to_lmaps(n, k+2, dim, Cs...)...)

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

Base.:(==)(A::BlockMap, B::BlockMap) = (eltype(A) == eltype(B) && A.maps == B.maps && A.rows == B.rows)

# special transposition behavior

LinearAlgebra.transpose(A::BlockMap) = TransposeMap(A)
LinearAlgebra.adjoint(A::BlockMap)  = AdjointMap(A)

############
# multiplication with vectors
############

function A_mul_B!(y::AbstractVector, A::BlockMap, x::AbstractVector)
    maps, rows = A.maps, A.rows
    mapind = 0
    yinds = firstrowindices(A)
    @views for rowind in 1:length(rows)
        xinds = firstcolindices(maps[mapind+1:mapind+rows[rowind]])
        yrow = y[yinds[rowind]:(yinds[rowind+1]-1)]
        mapind += 1
        A_mul_B!(yrow, maps[mapind], x[xinds[1]:xinds[2]-1])
        for colind in 2:rows[rowind]
            mapind +=1
            mul!(yrow, maps[mapind], x[xinds[colind]:xinds[colind+1]-1], 1, 1)
        end
    end
    return y
end

function At_mul_B!(y::AbstractVector, A::BlockMap, x::AbstractVector)
    maps, rows = A.maps, A.rows
    fill!(y, 0)
    mapind = 0
    xinds = firstrowindices(A)
    # first block row (rowind = 1), fill all of y
    yinds = firstcolindices(maps[mapind+1:mapind+rows[1]])
    xcol = @views x[xinds[1]:(xinds[2]-1)]
    @views for colind in 1:rows[1]
        mapind +=1
        A_mul_B!(y[yinds[colind]:yinds[colind+1]-1], transpose(maps[mapind]), xcol)
    end
    # subsequent block rows, add results to corresponding parts of y
    @views for rowind in 2:length(rows)
        yinds = firstcolindices(maps[mapind+1:mapind+rows[rowind]])
        xcol = x[xinds[rowind]:(xinds[rowind+1]-1)]
        for colind in 1:rows[rowind]
            mapind +=1
            mul!(y[yinds[colind]:yinds[colind+1]-1], transpose(maps[mapind]), xcol, 1, 1)
        end
    end
    return y
end

function Ac_mul_B!(y::AbstractVector, A::BlockMap, x::AbstractVector)
    maps, rows = A.maps, A.rows
    fill!(y, 0)
    mapind = 0
    xinds = firstrowindices(A)
    # first block row (rowind = 1), fill all of y
    yinds = firstcolindices(maps[mapind+1:mapind+rows[1]])
    xcol = @views x[xinds[1]:(xinds[2]-1)]
    @views for colind in 1:rows[1]
        mapind +=1
        A_mul_B!(y[yinds[colind]:yinds[colind+1]-1], adjoint(maps[mapind]), xcol)
    end
    # subsequent block rows, add results to corresponding parts of y
    @views for rowind in 2:length(rows)
        yinds = firstcolindices(maps[mapind+1:mapind+rows[rowind]])
        xcol = x[xinds[rowind]:(xinds[rowind+1]-1)]
        for colind in 1:rows[rowind]
            mapind +=1
            mul!(y[yinds[colind]:yinds[colind+1]-1], adjoint(maps[mapind]), xcol, 1, 1)
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
