using LinearMaps, LinearAlgebra, Test

@testset "filled maps" begin
    M, N = 2, 3
    μ = rand()
    for λ in (true, false, 3, μ, μ + 2im)
        L = LinearMap(λ, (M, N))
        @test L == LinearMap(λ, M, N)
        @test LinearMaps.MulStyle(L) === LinearMaps.FiveArg()
        A = fill(λ, (M, N))
        x = rand(typeof(λ) <: Real ? Float64 : ComplexF64, 3)
        X = rand(typeof(λ) <: Real ? Float64 : ComplexF64, 3, 4)
        w = similar(x, 2)
        W = similar(X, 2, 4)
        @test size(L) == (M, N)
        @test adjoint(L) == LinearMap(adjoint(λ), (3,2))
        @test transpose(L) == LinearMap(λ, (3,2))
        @test Matrix(L) == A
        @test L * x ≈ A * x
        @test mul!(w, L, x) ≈ A * x
        @test mul!(W, L, X) ≈ A * X
        for α in (true, false, 1, 0, randn()), β in (true, false, 1, 0, randn())
            @test mul!(copy(w), L, x, α, β) ≈ fill(λ * sum(x) * α, M) + w * β
            @test mul!(copy(W), L, X, α, β) ≈ λ * reduce(vcat, sum(X, dims=1) for _ in 1:2) * α + W * β
        end
    end
    @test issymmetric(LinearMap(μ + 1im, (3, 3)))
    @test ishermitian(LinearMap(μ + 0im, (3, 3)))
    @test isposdef(LinearMap(μ, (1,1))) == isposdef(μ)
    @test !isposdef(LinearMap(μ, (3,3)))
    α = rand()
    β = rand()
    @test LinearMap(μ, (M, N)) + LinearMap(α, (M, N)) == LinearMap(μ + α, (M, N))
    @test LinearMap(μ, (M, N)) - LinearMap(α, (M, N)) == LinearMap(μ - α, (M, N))
    @test α*LinearMap(μ, (M, N)) == LinearMap(α * μ, (M, N))
    @test LinearMap(μ, (M, N))*α == LinearMap(μ * α, (M, N))
    @test LinearMap(μ, (M, N))*LinearMap(μ, (N, M)) == LinearMap(μ^2*N, (M, M))
    @test Matrix(LinearMap(μ, (M, N))*LinearMap(μ, (N, M))) == fill(μ, (M, N))*fill(μ, (N, M))
end