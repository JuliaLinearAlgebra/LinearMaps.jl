using Test, LinearMaps, LinearAlgebra, SparseArrays

@testset "transpose/adjoint" begin
    A = 2 * rand(ComplexF64, (20, 10)) .- 1
    v = rand(ComplexF64, 10)
    w = rand(ComplexF64, 20)
    V = rand(ComplexF64, 10, 3)
    W = rand(ComplexF64, 20, 3)
    Av = A * v
    AV = A * V
    M = @inferred LinearMap(A)
    N = @inferred LinearMap(M)

    @test @inferred M' * w == A' * w
    @test @inferred mul!(copy(V), adjoint(M), W) ≈ A' * W
    @test @inferred transpose(M) * w == transpose(A) * w
    @test @inferred transpose(LinearMap(transpose(M))) * v == A * v
    @test @inferred LinearMap(M')' * v == A * v
    @test @inferred(transpose(transpose(M))) === M
    @test @inferred(adjoint(adjoint(M))) === M
    Mherm = @inferred LinearMap(A'A)
    @test @inferred ishermitian(Mherm)
    @test @inferred !issymmetric(Mherm)
    @test @inferred !issymmetric(transpose(Mherm))
    @test @inferred ishermitian(transpose(Mherm))
    @test @inferred ishermitian(Mherm')
    @test @inferred isposdef(Mherm)
    @test @inferred isposdef(transpose(Mherm))
    @test @inferred isposdef(adjoint(Mherm))
    @test @inferred !(transpose(M) == adjoint(M))
    @test @inferred !(adjoint(M) == transpose(M))
    @test @inferred transpose(M') * v ≈ transpose(A') * v
    @test @inferred transpose(LinearMap(M')) * v ≈ transpose(A') * v
    @test @inferred LinearMap(transpose(M))' * v ≈ transpose(A)' * v
    @test @inferred transpose(LinearMap(transpose(M))) * v ≈ Av
    @test @inferred adjoint(LinearMap(adjoint(M))) * v ≈ Av

    @test @inferred mul!(copy(w), transpose(LinearMap(M')), v) ≈ transpose(A') * v
    @test @inferred mul!(copy(w), LinearMap(transpose(M))', v) ≈ transpose(A)' * v
    @test @inferred mul!(copy(w), transpose(LinearMap(transpose(M))), v) ≈ Av
    @test @inferred mul!(copy(w), adjoint(LinearMap(adjoint(M))), v) ≈ Av
    @test @inferred mul!(copy(W), transpose(LinearMap(M')), V) ≈ transpose(A') * V
    @test @inferred mul!(copy(W), LinearMap(transpose(M))', V) ≈ transpose(A)' * V
    @test @inferred mul!(copy(W), transpose(LinearMap(transpose(M))), V) ≈ AV
    @test @inferred mul!(copy(W), adjoint(LinearMap(adjoint(M))), V) ≈ AV
    @test @inferred mul!(copy(V), transpose(M), W) ≈ transpose(A) * W
    @test @inferred mul!(copy(V), adjoint(M), W) ≈ A' * W

    B = @inferred LinearMap(Symmetric(rand(10, 10)))
    @test transpose(B) == B
    @test B == transpose(B)

    B = @inferred LinearMap(Hermitian(rand(ComplexF64, 10, 10)))
    @test adjoint(B) == B
    @test B == B'

    CS = @inferred LinearMap{ComplexF64}(cumsum, x -> reverse(cumsum(reverse(x))), 10; ismutating=false)
    for transform in (adjoint, transpose)
        @test transform(CS) != CS
        @test CS != transform(CS)
        @test transform(transform(CS)) == CS
        @test LinearMaps.MulStyle(transform(CS)) === LinearMaps.MulStyle(CS)
    end
    @test !(transpose(CS) == adjoint(CS))
    @test !(adjoint(CS) == transpose(CS))
    M = Matrix(CS)
    @test M == LowerTriangular(ones(10,10))
    x = rand(ComplexF64, 10); w = rand(ComplexF64, 10)
    X = rand(ComplexF64, 10, 3); W = rand(ComplexF64, 10, 3)
    α, β = rand(ComplexF64, 2)
    for transform in (adjoint, transpose)
        @test convert(AbstractMatrix, transform(CS)) == transform(M)
        @test sparse(transform(CS)) == transform(M)
        @test transform(CS) * x ≈ transform(M)*x
        @test mul!(copy(w), transform(CS), x) ≈ transform(M)*x
        @test mul!(copy(W), transform(CS), X) ≈ transform(M)*X
        @test mul!(copy(w), transform(CS), x, α, β) ≈ transform(M)*x*α + w*β
        @test mul!(copy(W), transform(CS), X, α, β) ≈ transform(M)*X*α + W*β
    end
    for transform1 in (adjoint, transpose), transform2 in (adjoint, transpose)
        @test transform2(transform1(CS)) * x ≈ transform2(transform1(M))*x
        @test mul!(copy(w), transform2(transform1(CS)), x) ≈ transform2(transform1(M))*x
        @test mul!(copy(W), transform2(transform1(CS)), X) ≈ transform2(transform1(M))*X
        @test mul!(copy(w), transform2(transform1(CS)), x, α, β) ≈ transform2(transform1(M))*x*α + w*β
        @test mul!(copy(W), transform2(transform1(CS)), X, α, β) ≈ transform2(transform1(M))*X*α + W*β
    end

    id = @inferred LinearMap(identity, identity, 10; issymmetric=true, ishermitian=true, isposdef=true)
    for transform in (adjoint, transpose)
        @test transform(id) == id
        @test id == transform(id)
        for prop in (issymmetric, ishermitian, isposdef)
            @test prop(transform(id))
        end
    end
end
