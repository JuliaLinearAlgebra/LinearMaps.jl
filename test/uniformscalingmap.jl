using Test, LinearMaps, LinearAlgebra

@testset "identity/scaling map" begin
    @test_throws ArgumentError LinearMaps.UniformScalingMap(true, -1)
    m = 5
    A = 2 * rand(ComplexF64, (m, m)) .- 1
    B = rand(size(A)...)
    M = @inferred 1 * LinearMap(A)
    N = @inferred LinearMap(B)
    LC = @inferred M + N
    v = rand(ComplexF64, m)
    w = similar(v)
    Id = @inferred LinearMap(I, m)
    @test occursin("$m×$m LinearMaps.UniformScalingMap{Bool}", sprint((t, s) -> show(t, "text/plain", s), Id))
    @test_throws ErrorException LinearMaps.UniformScalingMap(1, m, 2m)
    @test_throws ErrorException LinearMaps.UniformScalingMap(1, (m, 2m))
    @test size(Id) == (m, m)
    @test @inferred isreal(Id)
    @test @inferred issymmetric(Id)
    @test @inferred ishermitian(Id)
    @test @inferred isposdef(Id)
    @test Id * v == v
    @test (2 * M' + 3 * I) * v == 2 * A'v + 3v
    @test (3 * I + 2 * M') * v ≈ 3v + 2 * A'v rtol=2eps()
    @test (2 * M' - 3 * I) * v == 2 * A'v - 3v
    @test (3 * I - 2 * M') * v ≈ 3v - 2 * A'v rtol=2eps()
    @test transpose(LinearMap(2 * M' + 3 * I)) * v ≈ transpose(2 * A' + 3 * I) * v
    @test LinearMap(2 * M' + 0I)' * v ≈ (2 * A')' * v
    for λ in (0, 1, rand()), α in (0, 1, rand()), β in (0, 1, rand()), sz in (m, (m,5))
        Λ = @inferred LinearMap(λ*I, m)
        x = rand(Float64, sz)
        y = ones(Float64, sz)
        mul!(y, Λ, x, α, β)
        @test (@allocated mul!(y, Λ, x, α, β)) == 0
        y = copy(x)
        @inferred mul!(y, Λ, x, α, β)
        @test y ≈ λ * x * α + β * x
    end
    for elty in (Float64, ComplexF64), transform in (identity, transpose, adjoint)
        λ = rand(elty)
        x = rand(m)
        J = @inferred LinearMap(LinearMaps.UniformScalingMap(λ, m))
        @test transform(J) * x == transform(λ) * x
        J = @inferred LinearMap(λ*I, m)
        @test (λ * J) * x == (J * λ) * x == (λ * λ) * x
    end
    X = rand(m, m); Y = similar(X)
    @test mul!(Y, Id, X) == X
    @test Id*X isa LinearMap
    @test X*Id isa LinearMap
    @test Matrix(Id*X) == X
    @test Matrix(X*Id) == X
    @test Matrix(Id + X) == I + X
    @test Matrix(X + Id) == X + I
end
