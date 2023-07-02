using LinearMaps, LinearAlgebra, Test

@testset "trace" begin
    for A in (randn(5, 5), randn(ComplexF64, 5, 5))
        @test tr(LinearMap(A)) == tr(A)
        @test tr(transpose(LinearMap(A))) == tr(A)
        @test tr(adjoint(LinearMap(A))) == tr(A')
    end
    @test tr(LinearMap(3I, 10)) == 30
    @test tr(LinearMap{Int}(cumsum, 10)) == 10
    @test tr(LinearMap{Int}(cumsum, reverse∘cumsum∘reverse, 10)') == 10
    @test tr(LinearMap{Complex{Int}}(cumsum, reverse∘cumsum∘reverse, 10)') == 10
    @test tr(LinearMap{Int}(cumsum!, 10)) == 10
    @test tr(2LinearMap{Int}(cumsum!, 10)) == 20
    A = randn(3, 5); B = copy(transpose(A))
    @test tr(A ⊗ B) == tr(kron(A, B))
    @test tr(A ⊗ B ⊗ A ⊗ B) ≈ tr(kron(A, B, A, B))
    A = randn(5, 5); B = copy(transpose(A))
    @test tr(A ⊗ B) ≈ tr(kron(A, B))
    @test tr(A ⊗ B ⊗ A) ≈ tr(kron(A, B, A))
    @test tr(A ⊗ B ⊗ A ⊗ B) ≈ tr(kron(A, B, A, B))
    v = A[:,1]
    @test tr(v ⊗ v') ≈ norm(v)^2
    v = [randn(2,2) for _ in 1:3]
    @test tr(v ⊗ v') ≈ mapreduce(*, +, v, v')
    @test tr(LinearMap{Int}(cumsum!, 10) ⊕ LinearMap{Int}(cumsum!, 10)) == 200
    @test tr(FillMap(true, 5, 5)) == 5
end
