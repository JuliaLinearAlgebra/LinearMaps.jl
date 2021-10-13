using Test, LinearMaps, LinearAlgebra, BlockArrays

@testset "Non-traditional axes" begin

    A = rand(ComplexF64,2,4)
    B = PseudoBlockMatrix{ComplexF64}(undef, [2,3], [3,4])

    ax1 = axes(B)[1]
    ax2 = axes(B)[2]
    fill!(B,0)
    B[Block(1),Block(2)] .= A

    N = @inferred LinearMap(B)
    @test axes(N) == (ax1,ax2)

    @test eltype(N) == eltype(B)

    u = similar(Array{ComplexF64}, ax2)
    v = PseudoBlockVector{ComplexF64}(undef, [3,5])
    w = PseudoBlockVector{ComplexF64}(undef, [4,3])
    # v = similar(Array{ComplexF64}, blockedrange([3,5]))
    # w = similar(Array{ComplexF64}, blockedrange([4,3]))

    for i in eachindex(u) u[i] = rand(ComplexF64) end
    for i in eachindex(v) v[i] = rand(ComplexF64) end
    for i in eachindex(w) w[i] = rand(ComplexF64) end

    @test B*u == N*u
    @test_throws DimensionMismatch N*v

    # Lu = L*u
    Nu = N*u

    @test axes(Nu)[1] == axes(N)[1] == axes(B)[1]
    @test blocklengths(axes(Nu)[1]) == blocklengths(axes(N)[1]) == blocklengths(axes(B)[1]) == [2,3]

    C = B + 2N
    @test axes(C) === axes(B) === axes(N)
    @test C*u ≈ 3*Nu

    Cu = C*u
    @test axes(C)[1] == ax1
    @test blocklengths(axes(C)[1]) == blocklengths(ax1)
end
