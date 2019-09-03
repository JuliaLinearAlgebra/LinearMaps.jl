#=
indexmap.jl
2019-08-29 Jeff Fessler, University of Michigan
=#

export IndexMap, hcat_new, block_diag

"""
`check_index(row::AbstractVector{Int}, dims::Dims{2})`

Do not allow repeated index values because that would complicate the adjoint.
And insist on monotone increasing too because it is the easiest way to
ensure no repeats and probably better for cache too.
"""
function check_index(index::AbstractVector{Int}, dimA::Int, dimB::Int)
	length(index) != dimA && throw("invalid index length")
	minimum(index) < 0 && throw("negative index")
	maximum(index) > dimB &&
		throw("index $(maximum(index)) > dimB=$dimB")
	!(index isa UnitRange{Int64} ||
		((index isa StepRange{Int64,Int64}) && (index.step > 0)) ||
		all(diff(index) .> 0)) && throw("non-monotone index")
	nothing
end


"""
`IndexMap`

The original motivation was to support construction of linear operator
of the form `B = [0 0 0; 0 A 0; 0 0 0]`
from a given linear operator `A`
where `size(B) == dims`

This version is more general than the form above
because it allows the rows and colums of `A`
to be interspersed among the rows and columns of `B` arbitrarily.
Essentially like the above form but with some permutation-like matrices
before and after.
"""
struct IndexMap{T, As <: LinearMap, Rs <: AbstractVector{Int},
		Cs <: AbstractVector{Int}} <: LinearMap{T}
	map::As
	dims::Dims{2}
	set0::Bool # set to zero the output array?  default true
	rows::Rs # typically i1:i2 with 1 <= i1 <= i2 <= size(map,1)
	cols::Cs # typically j1:j2
	function IndexMap{T,As,Rs,Cs}(map::As, dims::Dims{2}, set0::Bool,
			rows::Rs, cols::Cs) where
			{T, As <: LinearMap,
			Rs <: AbstractVector{Int}, Cs <: AbstractVector{Int}}

		check_index(rows, size(map,1), dims[1])
		check_index(cols, size(map,2), dims[2])

		return new{T,As,Rs,Cs}(map, dims, set0, rows, cols)
	end
end

IndexMap{T}(map::As, dims::Dims{2} ; set0::Bool = true, offset::Dims{2}) where
		{T, As <: LinearMap} =
	IndexMap{T, As, UnitRange{Int64}, UnitRange{Int64}}(map, dims, set0,
		offset[1] .+ (1:size(map,1)),
		offset[2] .+ (1:size(map,2)))

IndexMap(map::LinearMap, dims::Dims{2} ; set0::Bool = true, offset::Dims{2}) =
	IndexMap{eltype(map)}(map, dims ; offset=offset, set0=set0)

IndexMap(map::LinearMap, dims::Dims{2},
		rows::AbstractVector{Int}, cols::AbstractVector{Int},
		; set0::Bool = true) =
	IndexMap{eltype(map), typeof(map), typeof(rows), typeof(cols)}(
		map, dims, set0, rows, cols)

Base.size(A::IndexMap) = A.dims


############
# basic methods
############

# the following rules are sufficient but perhaps not necessary
LinearAlgebra.issymmetric(A::IndexMap) = issymmetric(A.map) &&
	(A.dims[1] == A.dims[2]) && (A.rows == A.cols)

LinearAlgebra.ishermitian(A::IndexMap) = ishermitian(A.map) &&
	(A.dims[1] == A.dims[2]) && (A.rows == A.cols)

LinearAlgebra.ishermitian(A::IndexMap{<:Real}) = issymmetric(A)


# comparison of IndexMap objects
Base.:(==)(A::IndexMap, B::IndexMap) = (eltype(A) == eltype(B)) &&
	(A.map == B.map) && (A.dims == B.dims) &&
	(A.rows == B.rows) && (A.cols == B.cols) &&
	(A.set0 == B.set0)

# special transposition behavior
LinearAlgebra.transpose(A::IndexMap) =
	IndexMap(transpose(A.map), reverse(A.dims), A.cols, A.rows ; set0=A.set0)
LinearAlgebra.adjoint(A::IndexMap) =
	IndexMap(adjoint(A.map), reverse(A.dims), A.cols, A.rows ; set0=A.set0)


############
# multiplication with vectors
############

function A_mul_B!(y::AbstractVector, A::IndexMap, x::AbstractVector)
	m, n = size(A)
	@boundscheck (m == length(y) && n == length(x)) || throw(DimensionMismatch("A_mul_B!"))
#	@show A.set0, A.cols # todo: debug
	if A.set0
		fill!(y, zero(eltype(y)))
	end
	@views @inbounds mul!(y[A.rows], A.map, x[A.cols], true, true)
	return y
end


"(possibly) simplified version of hcat - initial draft"
function hcat_new(As::LinearMap...)
	rows = collect(map(A -> size(A,1), As))
	any(rows .!= rows[1]) && throw("row mismatch")
	cols = collect(map(A -> size(A,2), As))

	nmap = length(As)
	col_offsets = [0; cumsum(cols)[1:nmap-1]]
	dims = (rows[1], sum(cols))
	return +([IndexMap(As[ii], dims ; offset=(0, col_offsets[ii]),
		set0=true) # todo: for now always set to 0 to see timing penalty
# todo		set0=(ii==1))
# this was foiled by reversed order in
# https://github.com/Jutho/LinearMaps.jl/blob/master/src/linearcombination.jl#L33
		for ii in 1:nmap]...)
end


"""
first attempt at block diagonal.
it works, but has the overhead of zeroing out every block's output vector
"""
function block_diag(As::LinearMap...)
	rows = collect(map(A -> size(A,1), As))
	cols = collect(map(A -> size(A,2), As))
	dims = (sum(rows), sum(cols))
	nmap = length(As)
	col_offsets = [0; cumsum(cols)[1:nmap-1]]
	row_offsets = [0; cumsum(rows)[1:nmap-1]]
	return +([IndexMap(As[ii], dims ; offset=(row_offsets[ii], col_offsets[ii]),
		set0=true) # todo: for now always set to 0 to see timing penalty
# todo		set0=(ii==1))
		for ii in 1:nmap]...)
end

# todo: instead of relying on LinearCombination, make a special sum type?
