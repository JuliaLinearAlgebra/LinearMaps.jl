using Test, LinearMaps, LinearAlgebra, BenchmarkTools

@testset "linear combinations" begin
    CS! = LinearMap{ComplexF64}(cumsum!,
                                (y, x) -> (copyto!(y, x); reverse!(y); cumsum!(y, y)), 10;
                                ismutating=true)
    v = rand(10)
    u = similar(v)
    b = @benchmarkable mul!($u, $CS!, $v)
    @test run(b, samples=3).allocs == 0
    n = 10
    L = sum(fill(CS!, n))
    @test mul!(u, L, v) ≈ n * cumsum(v)
    b = @benchmarkable mul!($u, $L, $v)
    @test run(b, samples=5).allocs <= 1

    A = 2 * rand(ComplexF64, (10, 10)) .- 1
    B = rand(ComplexF64, size(A)...)
    M = @inferred LinearMap(A)
    N = @inferred LinearMap(B)
    LC = @inferred M + N
    v = rand(ComplexF64, 10)
    w = similar(v)
    b = @benchmarkable mul!($w, $M, $v)
    @test run(b, samples=3).allocs == 0
    if VERSION >= v"1.3.0-alpha.115"
        b = @benchmarkable mul!($w, $LC, $v)
        @test run(b, samples=3).allocs == 0
        for α in (false, true, rand(ComplexF64)), β in (false, true, rand(ComplexF64))
            b = @benchmarkable mul!($w, $LC, $v, $α, $β)
            @test run(b, samples=3).allocs == 0
            b = @benchmarkable mul!($w, $(LC + I), $v, $α, $β)
            @test run(b, samples=3).allocs == 0
            y = rand(ComplexF64, size(v))
            @test mul!(copy(y), LC, v, α, β) ≈ Matrix(LC)*v*α + y*β
            @test mul!(copy(y), LC+I, v, α, β) ≈ Matrix(LC + I)*v*α + y*β
        end
    end
    # @test_throws ErrorException LinearMaps.LinearCombination{ComplexF64}((M, N), (1, 2, 3))
    @test @inferred size(3M + 2.0N) == size(A)
    # addition
    @test @inferred convert(Matrix, LC) == A + B
    @test @inferred convert(Matrix, LC + LC) ≈ 2A + 2B
    @test @inferred convert(Matrix, M + LC) ≈ 2A + B
    @test @inferred convert(Matrix, M + M) ≈ 2A
    # subtraction
    @test (@inferred Matrix(LC - LC)) ≈ zeros(eltype(LC), size(LC)) atol=10eps()
    @test (@inferred Matrix(LC - M)) ≈ B
    @test (@inferred Matrix(N - LC)) ≈ -A
    @test (@inferred Matrix(M - M)) ≈ zeros(size(M)) atol=10eps()
    # scalar multiplication
    @test @inferred Matrix(-M) == -A
    @test @inferred Matrix(-LC) == -A - B
    @test @inferred Matrix(3 * M) == 3 * A
    @test @inferred Matrix(M * 3) == 3 * A
    @test Matrix(3.0 * LC) ≈ Matrix(LC * 3) ≈ 3A + 3B
    @test @inferred Matrix(3 \ M) ≈ A/3
    @test @inferred Matrix(M / 3) ≈ A/3
    @test @inferred Matrix(3 \ LC) ≈ (A + B) / 3
    @test @inferred Matrix(LC / 3) ≈ (A + B) / 3
    @test @inferred Array(3 * M' - CS!) == 3 * A' - Array(CS!)
    @test (3M - 1im*CS!)' == (3M + ((-1im)*CS!))' == M'*3 + CS!'*1im
    @test @inferred Array(M + A) == 2 * A
    @test @inferred Array(A + M) == 2 * A
    @test @inferred Array(M - A) == 0 * A
    @test Array(A - M) == 0 * A
end
