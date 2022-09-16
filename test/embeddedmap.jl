using Test, LinearMaps, LinearAlgebra, SparseArrays

@testset "embeddedmap" begin
    m = 6; n = 5
    M = 10(1:m) .+ (1:n)'; L = LinearMap(M)
    offset = (3,4)

    BM = [zeros(offset...) zeros(offset[1], size(M,2));
        zeros(size(M,1), offset[2]) M]
    BL = @inferred LinearMap(L, size(BM); offset=offset)
    s1, s2 = size(BM)
    @test (@inferred Matrix(BL)) == BM == mul!(zero(BM), BL, 1)
    @test mul!(copy(BM), BL, 2, true, true) == 3BM
    @test (@inferred Matrix(BL')) == BM'
    @test (@inferred Matrix(transpose(BL))) == transpose(BM)

    @test_throws UndefKeywordError LinearMap(M, (10, 10))
    @test_throws ArgumentError LinearMap(M, (m, n), (0:m, 1:n))
    @test_throws ArgumentError LinearMap(M, (m, n), (0:m-1, 1:n))
    @test_throws ArgumentError LinearMap(M, (m, n), (1:m, 1:n+1))
    @test_throws ArgumentError LinearMap(M, (m, n), (1:m, 2:n+1))
    @test_throws ArgumentError LinearMap(M, (m, n), offset=(3,3))
    # @test_throws ArgumentError LinearMap(M, (m, n), (m:-1:1, 1:n))
    # @test_throws ArgumentError LinearMap(M, (m, n), (collect(m:-1:1), 1:n))
    @test size(@inferred LinearMap(M, (2m, 2n), (1:2:2m, 1:2:2n))) == (2m, 2n)
    @test @inferred !ishermitian(BL)
    @test @inferred !issymmetric(BL)
    @test @inferred LinearMap(L, size(BM), (offset[1] .+ (1:m), offset[2] .+ (1:n))) == BL
    Wc = @inferred LinearMap([2 im; -im 0]; ishermitian=true)
    Bc = @inferred LinearMap(Wc, (4,4); offset=(2,2))
    @test (@inferred ishermitian(Bc))

    x = randn(s2); X = rand(s2, 3)
    y = BM * x; Y = zeros(s1, 3)

    @test @inferred BL * x ≈ BM * x
    @test @inferred BL' * y ≈ BM' * y

    for α in (true, false, rand()),
            β in (true, false, rand()),
            t in (identity, adjoint, transpose)

        @test t(BL) * x ≈ mul!(copy(y), t(BL), x) ≈ t(BM) * x
        @test Matrix(t(BL) * X) ≈ mul!(copy(Y), t(BL), X) ≈ t(BM) * X
        y = randn(s1); Y = randn(s1, 3)
        @test (@inferred mul!(copy(y), t(BL), x, α, β)) ≈ mul!(copy(y), t(BM), x, α, β)
        @test (@inferred mul!(copy(Y), t(BL), X, α, β)) ≈ mul!(copy(Y), t(BM), X, α, β)
    end

    if VERSION >= v"1.8"
        M = rand(3,4)
        L = LinearMap(M)
        @test Matrix(reverse(L)) == reverse(M)
        for dims in (1, 2, (1,), (2,), (1, 2), (2, 1), :)
            @test Matrix(reverse(L, dims=dims)) == reverse(M, dims=dims)
        end
    end
end
