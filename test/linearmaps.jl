using Test, LinearMaps, LinearAlgebra, SparseArrays, BenchmarkTools

@testset "basic functionality" begin
    A = 2 * rand(ComplexF64, (20, 10)) .- 1
    v = rand(ComplexF64, 10)
    w = rand(ComplexF64, 20)
    V = rand(ComplexF64, 10, 3)
    W = rand(ComplexF64, 20, 3)
    α = rand()
    β = rand()
    M = @inferred LinearMap(A)
    N = @inferred LinearMap(M)

    @testset "LinearMaps.jl" begin
        @test eltype(M) == eltype(A)
        @test size(M) == size(A)
        @test size(N) == size(A)
        @test !isreal(M)
        @test ndims(M) == 2
        @test_throws ErrorException size(M, 3)
        @test length(M) == length(A)
        # matrix generation/conversion
        @test Matrix(M) == A
        @test Array(M) == A
        @test convert(Matrix, M) == A
        @test convert(Array, M) == A
        @test Matrix(M') == A'
        @test Matrix(transpose(M)) == copy(transpose(A))
        # sparse matrix generation/conversion
        @test sparse(M) == sparse(Array(M))
        @test convert(SparseMatrixCSC, M) == sparse(Array(M))

        B = copy(A)
        B[rand(1:length(A), 30)] .= 0
        MS = LinearMap(B)
        @test sparse(MS) == sparse(Array(MS)) == sparse(B)

        J = LinearMap(I, 10)
        @test J isa LinearMap{Bool}
        @test sparse(J) == Matrix{Bool}(I, 10, 10)
        @test nnz(sparse(J)) == 10
    end

    Av = A * v
    AV = A * V

    @testset "mul! and *" begin
        @test M * v == Av
        @test N * v == Av
        @test @inferred mul!(copy(w), M, v) == mul!(copy(w), A, v)
        b = @benchmarkable mul!($w, $M, $v)
        @test run(b, samples=3).allocs == 0
        @test @inferred mul!(copy(w), N, v) == Av

        # mat-vec-mul
        @test @inferred mul!(copy(w), M, v, 0, 0) == zero(w)
        @test @inferred mul!(copy(w), M, v, 0, 1) == w
        @test @inferred mul!(copy(w), M, v, 0, β) == β * w
        @test @inferred mul!(copy(w), M, v, 1, 1) ≈ Av + w
        @test @inferred mul!(copy(w), M, v, 1, β) ≈ Av + β * w
        @test @inferred mul!(copy(w), M, v, α, 1) ≈ α * Av + w
        @test @inferred mul!(copy(w), M, v, α, β) ≈ α * Av + β * w
        @test @inferred mul!(copy(w), M, v, α)    ≈ α * Av

        # test mat-mat-mul!
        @test @inferred mul!(copy(W), M, V, α, β) ≈ α * AV + β * W
        @test @inferred mul!(copy(W), M, V, α) ≈ α * AV
        @test @inferred mul!(copy(W), M, V) ≈ AV
        @test typeof(M * V) <: LinearMap
    end
end
