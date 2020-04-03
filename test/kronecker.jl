using Test, LinearMaps, LinearAlgebra, SparseArrays

@testset "kronecker products" begin
    @testset "Kronecker product" begin
        A = rand(ComplexF64, 3, 3)
        B = rand(ComplexF64, 2, 2)
        K = kron(A, B)
        LA = LinearMap(A)
        LB = LinearMap(B)
        LK = @inferred kron(LA, LB)
        @test @inferred size(LK) == size(K)
        @test LinearMaps.MulStyle(LK) === LinearMaps.ThreeArg()
        for i in (1, 2)
            @test @inferred size(LK, i) == size(K, i)
        end
        @test LK isa LinearMaps.KroneckerMap{ComplexF64}

        for transform in (identity, transpose, adjoint)
            @test Matrix(transform(LK)) ≈ transform(Matrix(LK)) ≈ transform(K)
            @test Matrix(kron(transform(LA), transform(LB))) ≈ transform(K)
            @test Matrix(transform(LinearMap(LK))) ≈ transform(Matrix(LK)) ≈ transform(K)
        end
        @test kron(LA, kron(LA, B)) == kron(LA, LA, LB)
        @test kron(kron(LA, LB), kron(LA, LB)) == kron(LA, LB, LA, LB)
        @test kron(A, A, A) ≈ Matrix(@inferred kron(LA, LA, LA)) ≈ Matrix(@inferred LA^⊗(3)) ≈ Matrix(@inferred A^⊗(3))
        LAs = LinearMap(sparse(A))
        K = @inferred kron(A, A, A, LAs)
        @test K isa LinearMaps.KroneckerMap
        @test Matrix(K) ≈ kron(A, A, A, A)
        @test convert(AbstractMatrix, K) isa SparseMatrixCSC
        @test convert(AbstractMatrix, K) ≈ kron(A, A, A, A)
        @test sparse(K) ≈ kron(A, A, A, A)
        @test Matrix(@inferred K*K) ≈ kron(A, A, A, A)*kron(A, A, A, A)
        K4 = @inferred kron(A, B, B, LB)
        # check that matrices don't get Kronecker-multiplied, but that everything is lazy
        @test K4.maps[1].lmap === A
        @test @inferred kron(LA, LB)' == @inferred kron(LA', LB')
        @test (@inferred kron(LA, B)) == (@inferred kron(LA, LB)) == (@inferred kron(A, LB))
        @test @inferred ishermitian(kron(LA'LA, LB'LB))
        A = rand(2, 5); B = rand(4, 2)
        K = @inferred kron(A, LinearMap(B))
        v = rand(size(K, 2))
        @test K*v ≈ kron(A, B)*v
        @test Matrix(K) ≈ kron(A, B)
        K = @inferred kron(LinearMap(B), A)
        v = rand(size(K, 2))
        @test K*v ≈ kron(B, A)*v
        @test Matrix(K) ≈ kron(B, A)
        A = rand(3, 3); B = rand(2, 2); LA = LinearMap(A); LB = LinearMap(B)
        @test @inferred issymmetric(kron(LA'LA, LB'LB))
        @test @inferred ishermitian(kron(LA'LA, LB'LB))
        # use mixed-product rule
        K = kron(LA, LB) * kron(LA, LB) * kron(LA, LB)
        @test Matrix(K) ≈ kron(A, B)^3
        # example that doesn't use mixed-product rule
        A = rand(3, 2); B = rand(2, 3)
        K = @inferred kron(A, LinearMap(B))
        @test Matrix(@inferred K*K) ≈ kron(A, B)*kron(A, B)
        A = rand(3, 2); B = rand(4, 3)
        @test Matrix(kron(LinearMap(A), B, [A A])*kron(LinearMap(A), B, A')) ≈ kron(A, B, [A A])*kron(A, B, A')
    end

    @testset "Kronecker sum" begin
        for elty in (Float64, ComplexF64)
            A = rand(elty, 3, 3)
            B = rand(elty, 2, 2)
            LA = LinearMap(A)
            LB = LinearMap(B)
            KS = @inferred kronsum(LA, B)
            KSmat = kron(A, Matrix(I, 2, 2)) + kron(Matrix(I, 3, 3), B)
            @test Matrix(KS) ≈ Matrix(kron(A, LinearMap(I, 2)) + kron(LinearMap(I, 3), B))
            @test size(KS) == size(kron(A, Matrix(I, 2, 2)))
            for transform in (identity, transpose, adjoint)
                @test Matrix(transform(KS)) ≈ transform(Matrix(KS)) ≈ transform(KSmat)
                @test Matrix(kronsum(transform(LA), transform(LB))) ≈ transform(KSmat)
                @test Matrix(transform(LinearMap(kronsum(LA, LB)))) ≈ Matrix(transform(KS)) ≈ transform(KSmat)
            end
            @test @inferred(kronsum(A, A, LB)) == @inferred(⊕(A, A, B))
            @test Matrix(@inferred LA^⊕(3)) == Matrix(@inferred A^⊕(3)) ≈ Matrix(kronsum(LA, A, A))
            @test @inferred(kronsum(LA, LA, LB)) == @inferred(kronsum(LA, kronsum(LA, LB))) == @inferred(kronsum(A, A, B))
            @test Matrix(@inferred kronsum(A, B, A, B, A, B)) ≈ Matrix(@inferred kronsum(LA, LB, LA, LB, LA, LB))
            T = typeof(kron(Diagonal(rand(elty, 3)), sprand(3, 3, 0.3)))
            @test convert(AbstractMatrix, kronsum(sparse(A), sparse(B), sparse(A))) isa T
            @test convert(AbstractMatrix, kronsum(A, B, A)) == Matrix(kronsum(A, B, A))
            @test sparse(kronsum(sparse(A), B, A)) == Matrix(kronsum(sparse(A), B, A))
        end
    end
end
