using Test, LinearMaps, LinearAlgebra, BenchmarkTools

@testset "identity/scaling map" begin
    @test_throws ArgumentError LinearMaps.UniformScalingMap(true, 0)
    A = 2 * rand(ComplexF64, (10, 10)) .- 1
    B = rand(size(A)...)
    M = @inferred 1 * LinearMap(A)
    N = @inferred LinearMap(B)
    LC = @inferred M + N
    v = rand(ComplexF64, 10)
    w = similar(v)
    Id = @inferred LinearMaps.UniformScalingMap(1, 10)
    @test_throws ErrorException LinearMaps.UniformScalingMap(1, 10, 20)
    @test_throws ErrorException LinearMaps.UniformScalingMap(1, (10, 20))
    @test size(Id) == (10, 10)
    @test @inferred isreal(Id)
    @test @inferred issymmetric(Id)
    @test @inferred ishermitian(Id)
    @test @inferred isposdef(Id)
    @test Id * v == v
    @test (2 * M' + 3 * I) * v == 2 * A'v + 3v
    @test (3 * I + 2 * M') * v == 2 * A'v + 3v
    @test (2 * M' - 3 * I) * v == 2 * A'v - 3v
    @test (3 * I - 2 * M') * v == -2 * A'v + 3v
    @test transpose(LinearMap(2 * M' + 3 * I)) * v ≈ transpose(2 * A' + 3 * I) * v
    @test LinearMap(2 * M' + 3 * I)' * v ≈ (2 * A' + 3 * I)' * v
    for λ in (0, 1, rand()), α in (0, 1, rand()), β in (0, 1, rand())
        Λ = @inferred LinearMaps.UniformScalingMap(λ, 10)
        x = rand(10)
        y = rand(10)
        b = @benchmarkable mul!($y, $Λ, $x, $α, $β)
        @test run(b, samples=3).allocs == 0
        y = deepcopy(x)
        @inferred mul!(y, Λ, x, α, β)
        @test y ≈ λ * x * α + β * x
    end
    for elty in (Float64, ComplexF64), transform in (transpose, adjoint)
        λ = rand(elty)
        x = rand(10)
        J = @inferred LinearMap(LinearMaps.UniformScalingMap(λ, 10))
        @test transform(J) * x == transform(λ) * x
    end
end
