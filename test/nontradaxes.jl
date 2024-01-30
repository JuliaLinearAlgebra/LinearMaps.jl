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
    @test axes(N, 1) == ax1
    @test axes(N, 2) == ax2
    @test_throws ErrorException axes(N, 3)

    @test eltype(N) == eltype(B)

    u = similar(Array{ComplexF64}, ax2)
    v = PseudoBlockVector{ComplexF64}(undef, [3,5])
    w = similar(Array{ComplexF64}, ax1)

    for i in eachindex(u) u[i] = rand(ComplexF64) end
    for i in eachindex(v) v[i] = rand(ComplexF64) end
    for i in eachindex(w) w[i] = rand(ComplexF64) end

    @test B*u == N*u
    @test_throws DimensionMismatch N*v

    # Lu = L*u
    Nu = N*u

    @test axes(Nu)[1] == axes(N)[1] == axes(B)[1]
    @test blocklengths(axes(Nu)[1]) == blocklengths(axes(N)[1]) == blocklengths(axes(B)[1]) == [2,3]

    for trans in (adjoint, transpose)
        Nt = trans(LinearMap(N))
        Bt = trans(B)
        Ntw = Nt*w
        Btw = Bt*w
        @test Ntw ≈ Btw
        @test axes(Ntw)[1] == axes(Nt)[1] == axes(Bt)[1]
        @test blocklengths(axes(Ntw)[1]) == blocklengths(axes(Nt)[1]) == blocklengths(axes(Bt)[1]) == [3,4]
    end

    C = B + 2N
    @test axes(C) === axes(B) === axes(N)
    @test C*u ≈ 3*Nu

    Cu = C*u
    @test axes(C)[1] == ax1
    @test blocklengths(axes(C)[1]) == blocklengths(ax1)

    A = rand(ComplexF64,2,2)
    B = PseudoBlockMatrix{ComplexF64}(undef, [2,2], [2,2])
    ax1 = axes(B)[1]
    ax2 = axes(B)[2]
    fill!(B,0)
    B[Block(1),Block(2)] .= A
    L = LinearMap(B)
    L2 = L*L
    B2 = B*B
    @test axes(L2) == axes(B2)
    B2 = B*Matrix(B)
    L2 = L*LinearMap(Matrix(B))
    @test axes(L2) == axes(B2)
    u = similar(Array{ComplexF64}, ax2)
    B2u = B2*u; L2u = L2*u
    @test axes(B2u)[1] == axes(L2u)[1] == axes(B2)[1] == axes(L2)[1]
    @test blocklengths(axes(B2u)[1]) == blocklengths(axes(L2u)[1]) == [2,2]

    D1 = rand(4,5)
    D2 = rand(5,3)
    D3 = rand(3,6)
    D4 = rand(6,6)
    A1 = PseudoBlockMatrix(D1, [1,3], [2,3])
    A2 = PseudoBlockMatrix(D2, [2,3], [2,1])
    A3 = PseudoBlockMatrix(D3, [2,1], [3,2,1])
    A4 = PseudoBlockMatrix(D4, [3,2,1], [3,2,1])
    u = rand(6)
    x = PseudoBlockVector(u, [3,2,1])
    L = LinearMap(A1) * LinearMap(A2) * LinearMap(A3) * LinearMap(A4)
    y = L * x
    v = Vector(y)
    @test v ≈ D1*(D2*(D3*(D4*u)))
end
