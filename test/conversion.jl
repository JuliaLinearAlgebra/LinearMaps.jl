using Test, LinearMaps, LinearAlgebra, SparseArrays, Quaternions

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
    @test convert(AbstractMatrix, M) == A
    @test convert(Matrix, M) == A
    @test convert(Array, M) == A
    @test Matrix(M') == A'
    @test Matrix(transpose(M)) == copy(transpose(A))
    @test convert(AbstractMatrix, M') isa Adjoint
    @test convert(Matrix, M*3I) == A*3
    @test convert(Matrix, M+M) == A + A

    # UniformScalingMap
    J = LinearMap(α*I, 10)
    JM = convert(AbstractMatrix, J)
    @test JM == Diagonal(fill(α, 10))

    # ScaledMap
    q = rand(ComplexF64)
    A = rand(ComplexF64, 3, 3)
    @test convert(Matrix, q*LinearMap(A)) ≈ q*A
    qAs = convert(SparseMatrixCSC, q*LinearMap(A))
    @test qAs ≈ q*A
    @test qAs isa SparseMatrixCSC

    # CompositeMap of MatrixMap and UniformScalingMap
    q = Quaternion(rand(4)...)
    A = Quaternion.(rand(3,3), rand(3,3), rand(3,3), rand(3,3))
    for mat in (Matrix, SparseMatrixCSC)
        Aqmat = convert(mat, LinearMap(A)*q)
        @test Aqmat ≈ A*q
        @test Aqmat isa mat
        qAmat = convert(mat, q*LinearMap(A))
        @test qAmat ≈ q*A
        @test qAmat isa mat
    end

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
