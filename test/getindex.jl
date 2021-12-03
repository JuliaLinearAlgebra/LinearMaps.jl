using BenchmarkTools, LinearAlgebra, LinearMaps, Test
# using LinearMaps.GetIndex

function test_getindex(A::LinearMap, M::AbstractMatrix)
    @assert size(A) == size(M)
    @test all((A[i,j] == M[i,j] for i in axes(A, 1), j in axes(A, 2)))
    @test all((A[i] == M[i] for i in 1:length(A)))
    @test A[1,1] == M[1,1]
    @test A[:] == M[:]
    @test A[1,:] == M[1,:]
    @test A[:,1] == M[:,1]
    @test A[1:4,:] == M[1:4,:]
    @test A[:,1:4] == M[:,1:4]
    @test A[1,1:3] == M[1,1:3]
    @test A[1:3,1] == M[1:3,1]
    @test A[2:end,1] == M[2:end,1]
    @test A[1:2,1:3] == M[1:2,1:3]
    @test A[[2,1],1:3] == M[[2,1],1:3]
    @test A[:,:] == M
    @test A[7] == M[7]
    @test A[3:7] == M[3:7]
    @test_throws BoundsError A[firstindex(A)-1]
    @test_throws BoundsError A[lastindex(A)+1]
    @test_throws BoundsError A[6,1]
    @test_throws BoundsError A[1,6]
    @test_throws BoundsError A[2,1:6]
    @test_throws BoundsError A[1:6,2]
    return true
end

@testset "getindex" begin
    A = rand(5,5)
    L = LinearMap(A)
    @test test_getindex(L, A)
    # @btime getindex($A, i) setup=(i = rand(1:9));
    # @btime getindex($L, i) setup=(i = rand(1:9));
    # @btime (getindex($A, i, j)) setup=(i = rand(1:3); j = rand(1:3));
    # @btime (getindex($L, i, j)) setup=(i = rand(1:3); j = rand(1:3));

    struct TwoMap <: LinearMaps.LinearMap{Float64} end
    Base.size(::TwoMap) = (5,5)
    LinearMaps._getindex(::TwoMap, i::Integer, j::Integer) = 2.0
    LinearMaps._unsafe_mul!(y::AbstractVector, ::TwoMap, x::AbstractVector) = fill!(y, 2.0*sum(x))

    @test test_getindex(TwoMap(), fill(2.0, 5, 5))
    Base.adjoint(A::TwoMap) = A
    @test test_getindex(TwoMap(), fill(2.0, 5, 5))

    MA = rand(ComplexF64, 5, 5)
    for FA in (
        LinearMap{ComplexF64}((y, x) -> mul!(y, MA, x), (y, x) -> mul!(y, MA', x), 5, 5),
        LinearMap{ComplexF64}((y, x) -> mul!(y, MA, x), 5, 5),
    )
        @test test_getindex(FA, MA)
        @test test_getindex(3FA, 3MA)
        @test test_getindex(FA + FA, 2MA)
        if !isnothing(FA.fc)
            @test test_getindex(transpose(FA), transpose(MA))
            @test test_getindex(transpose(3FA), transpose(3MA))
            @test test_getindex(3transpose(FA), transpose(3MA))
            @test test_getindex(adjoint(FA), adjoint(MA))
            @test test_getindex(adjoint(3FA), adjoint(3MA))
            @test test_getindex(3adjoint(FA), adjoint(3MA))
        end
    end

    @test test_getindex(FillMap(0.5, (5, 5)), fill(0.5, (5, 5)))
    @test test_getindex(LinearMap(0.5I, 5), Matrix(0.5I, 5, 5))
end
