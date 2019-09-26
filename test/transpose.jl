using Test, LinearMaps, LinearAlgebra

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
    @test @inferred transpose(transpose(M)) == M
    @test (M')' == M
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
        @test transform(transform(CS)) == CS
    end
    @test transpose(CS) != adjoint(CS)
    @test adjoint(CS) != transpose(CS)
    M = Matrix(CS)
    x = rand(ComplexF64, 10)
    for transform1 in (adjoint, transpose), transform2 in (adjoint, transpose)
        @test transform2(LinearMap(transform1(CS))) * x ≈ transform2(transform1(M))*x
    end

    id = @inferred LinearMap(identity, identity, 10; issymmetric=true, ishermitian=true, isposdef=true)
    for transform in (adjoint, transpose)
        @test transform(id) == id
        for prop in (issymmetric, ishermitian, isposdef)
            @test prop(transform(id))
        end
    end
end
