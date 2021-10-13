using Test, LinearMaps, LinearAlgebra, LiftedMaps, BlockArrays

@testset "Non-traditional axes" begin

    ax1 = blockedrange([2,3])
    ax2 = blockedrange([3,4])

    A = rand(ComplexF64,2,4)
    L = LiftedMap(A,Block(1), Block(2), ax1, ax2)
    B = similar(PseudoBlockMatrix{ComplexF64}, ax1, ax2)
    fill!(B,0)
    B[Block(1),Block(2)] .= A

    M = @inferred LinearMap(L)
    N = @inferred LinearMap(B)
    @test axes(M) == axes(N)

    @test eltype(M) == eltype(L) == eltype(A)

    u = similar(Array{ComplexF64}, ax2)
    v = similar(Array{ComplexF64}, blockedrange([3,5]))
    w = similar(Array{ComplexF64}, blockedrange([4,3]))

    for i in eachindex(u) u[i] = rand(ComplexF64) end
    for i in eachindex(v) v[i] = rand(ComplexF64) end
    for i in eachindex(w) w[i] = rand(ComplexF64) end

    @test L*u == N*u
    @test_throws DimensionMismatch L*v
    @test_throws DimensionMismatch N*v

    Lu = L*u
    Nu = N*u

    @test axes(Lu)[1] == axes(L)[1] == axes(M)[1]
    @test axes(Nu)[1] == axes(N)[1] == axes(B)[1]

    @test blocksizes(axes(Lu)[1]) == blocksizes(axes(L)[1]) == ([2,3],)
    @test blocksizes(axes(Nu)[1]) == blocksizes(axes(N)[1]) == blocksizes(axes(B)[1]) == ([2,3],)

    C = L + 2N
    @test axes(C) === axes(L) === axes(N)
    @test C*u ≈ Lu + 2*Nu

    Cu = C*u
    @test axes(C)[1] == ax1
    @test blocksizes(axes(C)[1]) == blocksizes(ax1)

end
