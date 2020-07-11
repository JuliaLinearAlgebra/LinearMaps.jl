using Test, LinearMaps, LinearAlgebra

@testset "scaledmap" begin
    N = 7
    A = LinearMap(cumsum, reverse ∘ cumsum ∘ reverse, N)

    # real case
    α = float(π)
    B = α*A
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

    @test A * (α*I) == B
    @test (α*I) * A == B

    # complex case
    β = π + 2π*im
    C = β * A
    T = ComplexF32
    xc = rand(T, N)

    @test @inferred !isreal(C)
    @test @inferred !issymmetric(C)
    @test @inferred ishermitian(C) == ishermitian(A)
    @test @inferred !isposdef(C)
    @test @inferred transpose(C) == β * transpose(A)
    @test @inferred adjoint(C) == conj(β) * adjoint(A)

    @test C == A * β
    @test convert(Array{T}, C * xc) ≈ β * (A * xc)
    @test @inferred Matrix(C) == β * Matrix(A)
    @test @inferred Matrix(adjoint(C)) == conj(β) * Matrix(adjoint(A))
    @test @inferred Matrix(C') == Matrix(C)'

    # composition
    BC = B * C
    @test Matrix(BC) ≈ α*β*Matrix(A)*Matrix(A)

    @test Matrix(α * BC) ≈ α * Matrix(BC)
    @test Matrix(BC * β) ≈ β * Matrix(BC)

    # in-place
    y1 = β * (A * xc)
    y2 = similar(y1)
    mul!(y2, C, xc)
    @test y2 == y1

    y1 = β * (A * xc)
    y2 = similar(y1)
    mul!(y2, C, xc)
    @test y2 == y1

    x1 = conj(β) * A'*y1
    x2 = similar(x1)
    mul!(x2, C', y1)
    @test x2 == x1

    # check scale*conj(scale)
    A = LinearMap{Float32}(rand(N,2)) # rank=2 w.p.1
    B = β * A
    C = B' * B
    @test @inferred isposdef(C)
end
