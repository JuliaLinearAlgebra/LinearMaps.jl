# simple tests for getindex() capability
# something like this eventually should be merged into the main test routine

function LinearMap_test_getindex(A::LinearMap)
    B = Matrix(A)
    @assert all(size(A) .>= (4,4)) # so tests work
    @assert B[1] == A[1]
    @assert B[7] == A[7]
    @assert B[:,5] == A[:,5]
    @assert B[3,:] == A[3,:]
    @assert B[1,3] == A[1,3]
    @assert B[:,1:3] == A[:,1:3]
    @assert B[1:3,:] == A[1:3,:]
    @assert B[1:3,2:4] == A[1:3,2:4]
    @assert B == A[:,:]
    @assert B'[3] == A'[3]
    @assert B'[:,4] == A'[:,4]
    @assert B'[2,:] == A'[2,:]

    # The following do not work because currently LinearMap is not a
    # subtype of AbstractMatrix.  If it were such a subtype, then LinearMap
    # would inherit general Base.getindex abilities
    if false # todo later, after LinearMap is a subtype of AbstractMatrix
        @assert B[[1, 3, 4]] == A[[1, 3, 4]]
        @assert B[:, [1, 3, 4]] == A[:, [1, 3, 4]]
        @assert B[[1, 3, 4], :] == A[[1, 3, 4], :]
        @assert B[4:7] == A[4:7]
    end
end


# tests for cumsum()
# note: the adjoint of cumsum is reverse(cumsum(reverse(y)))

N = 5
A = LinearMap(cumsum, y -> reverse(cumsum(reverse(y))), N)
LinearMap_test_getindex(A)
display(Matrix(A))
display(Matrix(A'))
display(A[:,4])
