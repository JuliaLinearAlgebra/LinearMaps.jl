# getindex.jl
#
# Provides getindex() capabilities like A[:,j] for LinearMap objects
# Currently this provides only a partial set of the possible ways
# one can use indexing for a matrix, because currently
# LinearMap is not a subtype of an AbstractMatrix.
# If LinearMap were a subtype of an AbstractMatrix, then all possible
# indexing will be supported by Base.getindex, albeit very likely
# by quite inefficient iterators.
#
# These capabilities are provided as a user convenience.
# The user must recognize that LinearMap objects are not designed
# for efficient element-wise indexing.
#
# 2018-01-19, Jeff Fessler, University of Michigan

# A[i,j]
function Base.getindex(A::LinearMap, i::Int, j::Int)
	e = zeros(size(A,2)); e[j] = 1
	tmp = A * e
	return tmp[i]
end

# A[k]
function Base.getindex(A::LinearMap, k::Int)
	c = CartesianIndices(size(A))[k] # is there a more elegant way?
	return A[c[1], c[2]]
#	return A[c[:]] # fails
#	return A[c...] # fails
#	return A[Tuple(c)...] # works
#	return getindex(A, c[1], c[2]) # works
end

# A[:,j]
# it is crucial to provide this function rather than to inherit from
# Base.getindex(A::AbstractArray, ::Colon, ::Int)
# because Base.getindex does this by iterating (I think).
function Base.getindex(A::LinearMap, ::Colon, j::Int)
	e = zeros(size(A,2)); e[j] = 1
	return A * e
end

# A[i,:]
function Base.getindex(A::LinearMap, i::Int, ::Colon)
	# in Julia: A[i,:] = A'[:,i] for real matrix A else need conjugate
	return isreal(A) ? A'[:,i] : conj.(A'[:,i])
end

# A[:,j:k]
# this one is also important for efficiency
function Base.getindex(A::LinearMap, ::Colon, ur::UnitRange)
	return hcat([A[:,j] for j in ur]...)
end

# A[i:k,:]
Base.getindex(A::LinearMap, ur::UnitRange, ::Colon) = A'[:,ur]'

# A[:,:] = Matrix(A)
Base.getindex(A::LinearMap, ::Colon, ::Colon) = Matrix(A)

# A[i:k,j:l]
# this one is inefficient so could be inherited from Base?
Base.getindex(A::LinearMap, r1::UnitRange, r2::UnitRange) = A[:,r2][r1,:]

# A[???]
# informative error message in case we have overlooked any types
if false
function Base.getindex(A::LinearMap, kw...)
	@show kw
	for arg in kw
		@show typeof(arg)
	end
	error("unsupported indexing type")
end
end
