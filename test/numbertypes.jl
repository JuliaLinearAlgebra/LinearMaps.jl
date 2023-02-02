using Test, LinearMaps, LinearAlgebra
using Quaternions: Quaternion, quat
using Octonions: Octonion

@testset "noncommutative number type" begin
    x = randn(Quaternion{Float64},10)
    v = rand(10)
    A = randn(Quaternion{Float64}, 10, 10)
    B = randn(Quaternion{Float64}, 10, 10)
    C = similar(A)
    γ = randn(Quaternion{Float64}) # "Number"
    α = UniformScaling(γ)
    β = UniformScaling(randn(Quaternion{Float64}))
    λ = randn(Quaternion{Float64})
    L = LinearMap(A)
    F = LinearMap{eltype(A)}(x -> A*x, y -> A'y, 10)
    @test Array(F) == A
    @test Array(F') == A'
    @test Array(transpose(F)) == transpose(A)
    @test Array(α * F) == α * A
    @test Array(F * α) == A * α
    @test Array(α * F) == α * A
    @test Array(F * α ) == A * α
    @test Array(α * F') == α * A'
    for M in (L, F)
        @test mul!(C, transpose(A), M) ≈ transpose(A)*A
        @test mul!(C, A', M) ≈ A'A
        @test mul!(C, A, M) ≈ A*A
        @test mul!(copy(C), M, A, γ, λ) ≈ A*A*γ + C*λ
        @test mul!(copy(C), A, M, γ, λ) ≈ A*A*γ + C*λ
        @test mul!(copy(C), A, M, γ, 0) ≈ A*A*γ
        @test mul!(copy(C), transpose(A), M, γ, λ) ≈ transpose(A)*A*γ + C*λ
        @test mul!(copy(C), adjoint(A), M, γ, λ) ≈ A'*A*γ + C*λ
    end
    @test Array((α * F')') ≈ (γ * A')' ≈ A * conj(γ)
    @test L * x ≈ A * x
    @test L' * x ≈ A' * x
    @test α * (L * x) ≈ γ * (A * x)
    @test α * L * x ≈ γ * A * x
    @test L * α * x ≈ A * γ * x
    @test 3L * x ≈ 3A * x
    @test 3L' * x ≈ 3A' * x
    @test λ*L isa LinearMaps.CompositeMap
    @test γ * (λ * LinearMap(B)) isa LinearMaps.CompositeMap
    @test (λ * LinearMap(B)) * γ isa LinearMaps.CompositeMap
    @test ((L*L)*λ) isa LinearMaps.CompositeMap
    @test (λ*(L*L)) isa LinearMaps.CompositeMap
    @test λ*L * x ≈ λ*A * x
    @test λ*L' * x ≈ λ*A' * x
    @test α * (3L * x) ≈ γ * (3A * x)
    @test (@inferred α * 3L) * x ≈ γ * 3A * x
    @test (@inferred 3L * α) * x ≈ 3A * γ * x
    @test (α * L') * x ≈ (γ * A') * x
    @test (α * L')' * x ≈ (γ * A')' * x
    @test (α * L')' * v ≈ (γ * A')' * v
    @test Array(@inferred adjoint(α * L * β)) ≈ conj(β) * A' * conj(γ)
    @test Array(@inferred transpose(α * L * β)) ≈ β * transpose(A) * γ
    J = LinearMap(α, 10)
    @test (β * J) * x ≈ LinearMap(β*α, 10) * x ≈ β*γ*x
    @test (J * β) * x ≈ LinearMap(α*β, 10) * x ≈ γ*β*x
    M = β.λ * (γ * L * L)
    @test M == β * (γ * L * L)
    @test length(M.maps) == 3
    @test M.maps[end].λ == β.λ * γ
    @test γ * (β * L * L) == γ * (β.λ * L * L) == α * (β.λ * L * L) == α * (β * L * L)
    @test length(M.maps) == 3
    @test M.maps[end].λ == β.λ * γ
    M = (L * L * γ) * β.λ
    @test M == (L * L * γ) * β == (L * L * α) * β == (L * L * α) * β.λ
    @test length(M.maps) == 3
    @test M.maps[1].λ == γ*β.λ
    @test γ*FillMap(γ, (3, 4)) == FillMap(γ^2, (3, 4)) == FillMap(γ, (3, 4))*γ
    U = LinearMap(quat(1.0)*I, 10)
    for β in (0, 1, rand())
        @test mul!(copy(x), J, x, γ, β) == γ * x * γ + x * β
        @test mul!(copy(x), U, x, γ, β) == x * γ + x * β
    end
    # exercise non-RealOrComplex scalar operations
    @test Array(γ * (L'*L)) ≈ γ * (A'*A) # CompositeMap
    @test Array((L'*L) * γ) ≈ (A'*A) * γ
    @test Array(-L) == -A
    @test Array(γ \ L) ≈ γ \ A
    @test Array(L / γ) ≈ A / γ
    M = randn(10, 10); α = randn();
    y = α^2 * M * x; Y = α * M * A
    @test (α^2 * LinearMap(M)) * x ≈ quat(α)*(α * LinearMap(M)) * x ≈ y
    @test ((α * LinearMap(M))*quat(α)) * x ≈ y
    @test mul!(copy(y), α * LinearMap(M), x) ≈ α * M * x
    @test mul!(copy(y), α * LinearMap(M), x, α, false) ≈ α * M * x * α
    @test mul!(copy(y), α * LinearMap(M), x, quat(α), false) ≈ α * M * x * α
    @test mul!(copy(Y), α * LinearMap(M), A) ≈ α * M * A
    @test mul!(copy(Y), α * LinearMap(M), A, α, false) ≈ α * M * A * α
end

@testset "nonassociative number type" begin
    x = randn(Octonion{Float64}, 10)
    v = rand(10)
    A = randn(Octonion{Float64}, 10, 10)
    α = randn(Octonion{Float64})*I
    β = randn(Octonion{Float64})*I
    L = LinearMap(A)
    @test Array(L) == A
    @test Array(α * L) == α * A
    @test Array(L * α) == A * α
    @test Array(α * L) == α * A
    @test Array(L * α) == A * α
    @test (α * L')' * v ≈ (α * A')' * v
end
