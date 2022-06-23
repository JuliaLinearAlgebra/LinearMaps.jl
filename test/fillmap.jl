using LinearMaps, LinearAlgebra, Test

@testset "filled maps" begin
    M, N = 2, 3
    μ = rand()
    for λ in (true, false, 3, μ, μ + 2im)
        L = FillMap(λ, (M, N))
        @test L == FillMap(λ, M, N)
        @test occursin("$M×$N FillMap{$(typeof(λ))} with fill value: $λ", sprint((t, s) -> show(t, "text/plain", s), L))
        @test LinearMaps.MulStyle(L) === LinearMaps.FiveArg()
        A = fill(λ, (M, N))
        x = rand(typeof(λ) <: Real ? Float64 : ComplexF64, 3)
        X = rand(typeof(λ) <: Real ? Float64 : ComplexF64, 3, 4)
        w = similar(x, 2)
        W = similar(X, 2, 4)
        @test size(L) == (M, N)
        @test adjoint(L) == FillMap(adjoint(λ), (3,2))
        @test transpose(L) == FillMap(λ, (3,2))
        @test !issymmetric(L)
        @test !ishermitian(L)
        @test !isposdef(L)
        @test Matrix(L) == A == mul!(copy(A), L, 1) == mul!(copy(A), L, 1, true, false)
        @test mul!(copy(1A), L, 2, true, true) == 3A
        @test L * x ≈ A * x
        @test mul!(w, L, x) ≈ A * x
        @test mul!(W, L, X) ≈ A * X
        for α in (true, false, 1, 0, randn()), β in (true, false, 1, 0, randn())
            @test mul!(copy(w), L, x, α, β) ≈ fill(λ * sum(x) * α, M) + w * β
            @test mul!(copy(W), L, X, α, β) ≈ λ * reduce(vcat, sum(X, dims=1) for _ in 1:2) * α + W * β
        end
    end
    @test issymmetric(FillMap(μ + 1im, (3, 3)))
    @test ishermitian(FillMap(μ + 0im, (3, 3)))
    @test isposdef(FillMap(μ, (1,1))) == isposdef(μ)
    @test !isposdef(FillMap(μ, (3,3)))
    α = rand()
    β = rand()
    @test FillMap(μ, (M, N)) + FillMap(α, (M, N)) == FillMap(μ + α, (M, N))
    @test FillMap(μ, (M, N)) - FillMap(α, (M, N)) == FillMap(μ - α, (M, N))
    @test α*FillMap(μ, (M, N)) == FillMap(α * μ, (M, N))
    @test FillMap(μ, (M, N))*α == FillMap(μ * α, (M, N))
    @test FillMap(μ, (M, N))*FillMap(μ, (N, M)) == FillMap(μ^2*N, (M, M))
    @test Matrix(FillMap(μ, (M, N))*FillMap(μ, (N, M))) ≈ fill(μ, (M, N))*fill(μ, (N, M))
end