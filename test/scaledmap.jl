using Test, LinearMaps, LinearAlgebra, BenchmarkTools

@testset "scaledmap" begin
    N = 7
    A = LinearMap(cumsum, reverse ∘ cumsum ∘ reverse, N)
    AM = Matrix(A)

    # real case
    α = float(π)
    B = α * A
    x = rand(N)

    @test @inferred size(B) == size(A)
    @test @inferred isreal(B)
    @test @inferred issymmetric(B) == issymmetric(A)
    @test @inferred ishermitian(B) == ishermitian(A)
    @test @inferred isposdef(B) == isposdef(A)
    @test @inferred transpose(B) == α * transpose(A)

    @test B == A * α
    @test B * x == α * (A * x)
    @test @inferred Matrix(B) == α * Matrix(A)
    @test @inferred Matrix(B') == Matrix(B)'

    @test -A == (-1) * A

    @test A * (α * I) == B
    @test (α * I) * A == B

    # complex case
    β = π + 2π * im
    C = β * A
    T = ComplexF64
    xc = rand(T, N)

    @test @inferred !isreal(C)
    @test @inferred !issymmetric(C)
    @test @inferred ishermitian(C) == ishermitian(A)
    @test @inferred !isposdef(C)
    @test @inferred transpose(C) == β * transpose(A)
    @test @inferred adjoint(C) == conj(β) * adjoint(A)
    for transform in (identity, transpose, adjoint)
        @test transform(C) * xc ≈ transform(β * AM) * xc
    end

    @test C == A * β
    @test convert(Array{T}, C * xc) ≈ β * (A * xc)
    @test @inferred Matrix(C) == β * Matrix(A)
    @test @inferred Matrix(adjoint(C)) == conj(β) * Matrix(adjoint(A))
    @test @inferred Matrix(C') == Matrix(C)'

    # composition
    BA = B*A
    @test BA isa LinearMaps.ScaledMap
    @test BA.λ == B.λ
    AB = A*B
    @test AB isa LinearMaps.ScaledMap
    @test AB.λ == B.λ
    BC = B * C
    @test BC isa LinearMaps.ScaledMap
    @test BC.λ == α * β
    @test Matrix(BC) ≈ α * β * Matrix(A) * Matrix(A)

    @test Matrix(α * BC) ≈ α * Matrix(BC)
    @test Matrix(BC * β) ≈ β * Matrix(BC)

    # in-place
    y1 = β * (A * xc)
    y2 = similar(y1)
    @test mul!(y2, C, xc) == y1

    x1 = conj(β) * (A' * y1)
    x2 = similar(x1)
    @test mul!(x2, C', y1) == x1

    # check scale*conj(scale)
    A = LinearMap{Float32}(rand(N, 2)) # rank=2 w.p.1
    B = β * A
    C = B' * B
    @test @inferred isposdef(C)

    N = 2^8
    A0 = LinearMap{T}(cumsum, reverse ∘ cumsum ∘ reverse, N) # out-of-place
    forw! = cumsum!
    back! = (x, y) -> reverse!(cumsum!(x, reverse!(copyto!(x, y))))
    A1 = LinearMap{T}(forw!, back!, N) # in-place
    λ = 4.2
    B0 = λ * A0
    B1 = λ * A1
    for (A, alloc) in ((A0, 1), (A1, 0), (B0, 1), (B1, 0), (A0', 3), (A1', 0), (B0', 3), (B1', 0))
        x = rand(N)
        y = similar(x)
        b = @benchmarkable mul!($y, $A, $x)
        @test run(b, samples = 3).allocs <= alloc
    end
end
