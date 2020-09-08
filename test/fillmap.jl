using LinearMaps, LinearAlgebra, Test

@testset "filled maps" begin
    M, N = 2, 3
    α = rand()
    β = rand()
    μ = rand()
    for λ in (true, false, 3, μ, μ + 2im)
        L = LinearMap(λ, (M, N))
        A = fill(λ, (M, N))
        x = rand(typeof(λ) <: Real ? Float64 : ComplexF64, 3)
        X = rand(typeof(λ) <: Real ? Float64 : ComplexF64, 3, 3)
        w = similar(x, 2)
        W = similar(X, 2, 3)
        @test size(L) == (M, N)
        @test adjoint(L) == LinearMap(adjoint(λ), (3,2))
        @test transpose(L) == LinearMap(λ, (3,2))
        @test Matrix(L) == A
        @test L * x ≈ A * x
        @test mul!(w, L, x) ≈ A * x
        @test mul!(W, L, X) ≈ A * X
        @test mul!(copy(w), L, x, α, β) ≈ A * x * α + w * β
        @test mul!(copy(W), L, X, α, β) ≈ A * X * α + W * β
    end
    @test issymmetric(LinearMap(μ + 0im, (3, 3)))
    @test LinearMap(μ, (M, N)) + LinearMap(α, (M, N)) == LinearMap(μ + α, (M, N))
    @test LinearMap(μ, (M, N)) - LinearMap(α, (M, N)) == LinearMap(μ - α, (M, N))
    @test α*LinearMap(μ, (M, N)) == LinearMap(α * μ, (M, N))
    @test LinearMap(μ, (M, N))*α == LinearMap(μ * α, (M, N))
end