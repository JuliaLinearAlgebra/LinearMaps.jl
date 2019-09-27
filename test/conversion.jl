using Test, LinearMaps, LinearAlgebra, SparseArrays

@testset "conversion" begin
    A = 2 * rand(ComplexF64, (20, 10)) .- 1
    v = rand(ComplexF64, 10)
    w = rand(ComplexF64, 20)
    V = rand(ComplexF64, 10, 3)
    W = rand(ComplexF64, 20, 3)
    α = rand()
    β = rand()
    M = @inferred LinearMap(A)
    N = @inferred LinearMap(M)

    @test Matrix(M) == A
    @test Array(M) == A
    @test convert(AbstractArray, M) == A
    @test convert(AbstractMatrix, M) === A
    @test convert(Matrix, M) === A
    @test convert(Array, M) === A
    @test Matrix(M') == A'
    @test Matrix(transpose(M)) == copy(transpose(A))
    @test convert(AbstractMatrix, M') isa Adjoint
    @test convert(Matrix, M*3I) == A*3

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
