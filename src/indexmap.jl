#=
indexmap.jl
2019-08-29 Jeff Fessler, University of Michigan
=#

export IndexMap, hcat_new

#using LinearMaps
#import LinearMaps: A_mul_B!
#import LinearAlgebra


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
	rows::Rs # typically i1:i2 with 1 <= i1 <= i2 <= size(map,1)
	cols::Cs # typically j1:j2
	function IndexMap{T,As,Rs,Cs}(map::As, dims::Dims{2}, rows::Rs, cols::Cs) where
		{T, As <: LinearMap, Rs <: AbstractVector{Int}, Cs <: AbstractVector{Int}}

		check_index(rows, size(map,1), dims[1])
		check_index(cols, size(map,2), dims[2])

		return new{T,As,Rs,Cs}(map, dims, rows, cols)
	end
end

IndexMap{T}(map::As, dims::Dims{2} ; offset::Dims{2}) where {T, As <: LinearMap} =
	IndexMap{T, As, UnitRange{Int64}, UnitRange{Int64}}(map, dims,
		offset[1] .+ (1:size(map,1)),
		offset[2] .+ (1:size(map,2)))

IndexMap(map::LinearMap, dims::Dims{2} ; offset::Dims{2}) =
	IndexMap{eltype(map)}(map, dims ; offset=offset)

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
	(A.rows == B.rows) && (A.cols == B.cols)

# special transposition behavior
LinearAlgebra.transpose(A::IndexMap) = TransposeMap(A)
LinearAlgebra.adjoint(A::IndexMap) = AdjointMap(A)


############
# multiplication with vectors
############

function A_mul_B!(y::AbstractVector, A::IndexMap, x::AbstractVector)
	m, n = size(A)
	@boundscheck (m == length(y) && n == length(x)) || throw(DimensionMismatch("A_mul_B!"))
	y[:] .= 0
	@views @inbounds mul!(y[A.rows], A.map, x[A.cols], true, true)
	return y
end

function At_mul_B!(y::AbstractVector, A::IndexMap, x::AbstractVector)
	m, n = size(A)
	@boundscheck (n == length(y) && m == length(x)) || throw(DimensionMismatch("At_mul_B!"))
	y[:] .= 0
	@views @inbounds mul!(y[A.cols], transpose(A.map), x[A.rows], true, true)
	return y
end

function Ac_mul_B!(y::AbstractVector, A::IndexMap, x::AbstractVector)
	m, n = size(A)
	y[:] .= 0
	@boundscheck (n == length(y) && m == length(x)) || throw(DimensionMismatch("Ac_mul_B!"))
	@views @inbounds mul!(y[A.cols], adjoint(A.map), x[A.rows], true, true)
	return y
end

#=
=#
"simplified version of hcat"
function hcat_new(As::LinearMap...)
	rows = collect(map(A -> size(A,1), As))
	any(rows .!= rows[1]) && throw("row mismatch")
	cols = collect(map(A -> size(A,2), As))

	nmap = length(As)
	col_offsets = [0; cumsum(cols)[1:nmap-1]]
	dims = (rows[1], sum(cols))
	return +([IndexMap(As[ii], dims ; offset=(0, col_offsets[ii]))
		for ii in 1:nmap]...)
end
