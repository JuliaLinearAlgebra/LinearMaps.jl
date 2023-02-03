using Test, LinearMaps, LinearAlgebra

@testset "scaledmap" begin
    N = 7
    A = LinearMap(cumsum, reverse ∘ cumsum ∘ reverse, N)
    AM = Matrix(A)

    # real case
    α = float(π)
    B = @inferred α * A
    @test occursin("7×7 LinearMaps.ScaledMap{Float64} with scale: $α", sprint((t, s) -> show(t, "text/plain", s), B))
    x = rand(N)

    @test @inferred size(B) == size(A)
    @test @inferred isreal(B)
    @test @inferred issymmetric(B) == issymmetric(A)
    @test @inferred ishermitian(B) == ishermitian(A)
    @test @inferred isposdef(B) == isposdef(A)
    @test @inferred transpose(B) == α * transpose(A)
    A! = LinearMap((y, x) -> cumsum!(y, x), N)
    L = 0.5A! + 0.5A! + 1.0A!
    @test LinearMaps.MulStyle(L) === LinearMaps.ThreeArg()
    L*x; y = similar(x)
    @test mul!(y, L, x) ≈ 2cumsum(x)
    LM = 0.5LinearMap(AM) + 0.5LinearMap(AM) + 1.0LinearMap(AM)
    @test LM*x ≈ 2cumsum(x)
    @test LinearMaps.MulStyle(LM) === LinearMaps.FiveArg()
    mul!(y, LM, x)
    @test (@allocated mul!(y, LM, x)) == 0

    @test B == A * α
    @test B * x == α * (A * x)
    @test @inferred Matrix(B) == α * Matrix(A)
    @test @inferred Matrix(B') == Matrix(B)'

    @test -A == (-1) * A

    @test A * (α * I) == B
    @test (α * I) * A == B

    # complex case
    β = π + 2π * im
    C = @inferred β * A
    @test_throws ErrorException LinearMaps.ScaledMap{Float64}(β, A)
    @inferred conj(β) * A' # needed in left-mul
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

    @test C == @inferred A * β
    @test convert(Array{T}, C * xc) ≈ β * (A * xc)
    @test @inferred Matrix(C) == β * Matrix(A)
    @test @inferred Matrix(adjoint(C)) == conj(β) * Matrix(adjoint(A))
    @test @inferred Matrix(C') == Matrix(C)'

    # composition
    BA = @inferred B*A
    @test BA isa LinearMaps.ScaledMap
    @test BA.λ == B.λ
    AB = @inferred A*B
    @test AB isa LinearMaps.ScaledMap
    @test AB.λ == B.λ
    BC = @inferred B * C
    @test BC isa LinearMaps.ScaledMap
    @test BC.λ == α * β
    @test Matrix(BC) ≈ α * β * Matrix(A) * Matrix(A)

    @test Matrix(@inferred α * BC) ≈ α * Matrix(BC)
    @test Matrix(@inferred BC * β) ≈ β * Matrix(BC)

    # in-place
    y1 = β * (A * xc)
    y2 = similar(y1)
    @test mul!(y2, C, xc) == y1

    x1 = conj(β) * (A' * y1)
    x2 = similar(x1)
    @test mul!(x2, C', y1) == x1

    # check scale*conj(scale)
    A = LinearMap{Float32}(rand(1, 2)) # rank=1 w.p.1
    B = @inferred β * A
    C = @inferred B' * B
    @test !isposdef(C)
    @test @inferred isposdef(C.λ)

    N = 2^8
    A0 = LinearMap{T}(cumsum, reverse ∘ cumsum ∘ reverse, N) # out-of-place
    forw! = cumsum!
    back! = (x, y) -> reverse!(cumsum!(x, reverse!(copyto!(x, y))))
    A1 = @inferred FunctionMap{T,true}(forw!, back!, N) # in-place
    λ = 4.2
    B0 = @inferred λ * A0
    B1 = @inferred λ * A1
    for (A, alloc) in ((A0, 1), (A1, 0), (B0, 1), (B1, 0), (A0', 3), (A1', 0), (B0', 3), (B1', 0))
        x = rand(N)
        y = similar(x)
        allocsize = @allocated similar(y)
        mul!(y, A, x)
        @test (@allocated mul!(y, A, x)) == alloc*allocsize
    end
end
