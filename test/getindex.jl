using LinearAlgebra, LinearMaps, Test
using LinearMaps: VecOrMatMap, ScaledMap
# using BenchmarkTools

function test_getindex(A::LinearMap, M::AbstractMatrix)
    @assert size(A) == size(M)
    mask = rand(Bool, size(A))
    imask = rand(Bool, size(A, 1))
    jmask = rand(Bool, size(A, 2))
    @test A[1,:] == M[1,:]
    @test A[:,1] == M[:,1]
    @test A[1:lastindex(A,1)-2,:] == M[1:lastindex(A,1)-2,:]
    @test A[:,1:4] == M[:,1:4]
    @test A[[2,1],:] == M[[2,1],:]
    @test A[:,[2,1]] == M[:,[2,1]]
    @test A[:,:] == M
    @test (lastindex(A, 1), lastindex(A, 2)) == size(A)
    if A isa VecOrMatMap || A isa ScaledMap{<:Any,<:Any,<:VecOrMatMap}
        @test A[:] == M[:]
        @test A[1,1] == M[1,1]
        @test A[1,1:3] == M[1,1:3]
        @test A[1:3,1] == M[1:3,1]
        @test A[2:end,1] == M[2:end,1]
        @test A[1:2,1:3] == M[1:2,1:3]
        @test A[[2,1],1:3] == M[[2,1],1:3]
        @test A[7] == M[7]
        @test A[3:7] == M[3:7]
        @test A[mask] == M[mask]
        @test A[findall(mask)] == M[findall(mask)]
        @test A[CartesianIndex(1,1)] == M[CartesianIndex(1,1)]
        @test A[imask, 1] == M[imask, 1]
        @test A[1, jmask] == M[1, jmask]
        @test A[imask, jmask] == M[imask, jmask]
    else
        @test_throws ErrorException A[:]
        @test_throws ErrorException A[1,1]
        @test_throws ErrorException A[1,1:3]
        @test_throws ErrorException A[1:3,1]
        @test_throws ErrorException A[2:end,1]
        @test_throws ErrorException A[1:2,1:3]
        @test_throws ErrorException A[[2,1],1:3]
        @test_throws ErrorException A[7]
        @test_throws ErrorException A[3:7]
        @test_throws ErrorException A[mask]
        @test_throws ErrorException A[findall(mask)]
        @test_throws ErrorException A[CartesianIndex(1,1)]
        @test_throws ErrorException A[imask, 1]
        @test_throws ErrorException A[1, jmask]
        @test_throws ErrorException A[imask, jmask]
    end
    @test_throws BoundsError A[lastindex(A,1)+1,1]
    @test_throws BoundsError A[1,lastindex(A,2)+1]
    @test_throws BoundsError A[2,1:lastindex(A,2)+1]
    @test_throws BoundsError A[1:lastindex(A,1)+1,2]
    @test_throws BoundsError A[ones(Bool, 2, 2)]
    @test_throws BoundsError A[[true, true], 1]
    @test_throws BoundsError A[1, [true, true]]
    return nothing
end

@testset "getindex" begin
    M = rand(4,6)
    A = LinearMap(M)
    test_getindex(A, M)
    test_getindex(2A, 2M)
    # @btime getindex($M, i) setup=(i = rand(1:24));
    # @btime getindex($A, i) setup=(i = rand(1:24));
    # @btime (getindex($M, i, j)) setup=(i = rand(1:4); j = rand(1:6));
    # @btime (getindex($A, i, j)) setup=(i = rand(1:4); j = rand(1:6));

    struct TwoMap <: LinearMaps.LinearMap{Float64} end
    Base.size(::TwoMap) = (5,5)
    LinearMaps._unsafe_mul!(y::AbstractVector, ::TwoMap, x::AbstractVector) = fill!(y, 2.0*sum(x))
    T = TwoMap()
    @test_throws ErrorException T[1,:]

    Base.transpose(A::TwoMap) = A
    test_getindex(TwoMap(), fill(2.0, size(T)))

    MA = rand(ComplexF64, 5, 5)
    FA = LinearMap{ComplexF64}((y, x) -> mul!(y, MA, x), (y, x) -> mul!(y, MA', x), 5, 5)
    F = LinearMap{ComplexF64}(x -> MA*x, y -> MA'y, 5, 5)
    test_getindex(FA, MA)
    test_getindex([FA FA], [MA MA])
    test_getindex([FA; FA], [MA; MA])
    test_getindex(F, MA)
    test_getindex(3FA, 3MA)
    test_getindex(FA + FA, 2MA)
    test_getindex(transpose(FA), transpose(MA))
    test_getindex(transpose(3FA), transpose(3MA))
    test_getindex(3transpose(FA), transpose(3MA))
    test_getindex(adjoint(FA), adjoint(MA))
    test_getindex(adjoint(3FA), adjoint(3MA))
    test_getindex(3adjoint(FA), adjoint(3MA))

    test_getindex(FillMap(0.5, (5, 5)), fill(0.5, (5, 5)))
    test_getindex(LinearMap(0.5I, 5), Matrix(0.5I, 5, 5))
end
